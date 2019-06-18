#include <lgr_state.hpp>
#include <lgr_input.hpp>
#include <lgr_adapt.hpp>
#include <lgr_element_specific_inline.hpp>
#include <lgr_print.hpp>
#include <lgr_meshing.hpp>

#include <iostream>
#include <iomanip>

namespace lgr {

HPC_NOINLINE inline void update_bar_quality(state& s) {
  hpc::fill(hpc::device_policy(), s.quality, double(1.0));
}

/* Per:
   Shewchuk, Jonathan Richard.
   "What is a good linear finite element?
    interpolation, conditioning, anisotropy, and quality measures (preprint)."
   University of California at Berkeley 73 (2002): 137.

   A scale-invariant quality measure correlated with matrix conditioning
   (and the stable explicit time step) for triangles is area divided by
   the square of the (root-mean-squared edge length).
   We also use the fact that we already have area computed and that basis
   gradient magnitudes relate to opposite edge lengths.
  */
inline HPC_HOST_DEVICE hpc::dimensionless<double> triangle_quality(
    hpc::array<hpc::basis_gradient<double>, 3> const grad_N,
    hpc::area<double> const area) noexcept {
  decltype(1.0 / hpc::area<double>()) sum_g_i_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    auto const g_i_sq = (grad_N[i] * grad_N[i]);
    sum_g_i_sq += g_i_sq;
  }
  auto const denom = (area * sum_g_i_sq);
  auto const q = 1.0 / denom;
  return q;
}

inline HPC_HOST_DEVICE hpc::dimensionless<double> triangle_quality(hpc::array<hpc::position<double>, 3> const x) noexcept {
  auto const area = triangle_area(x);
  if (area <= 0.0) return -1.0;
  return triangle_quality(triangle_basis_gradients(x, area), area);
}

HPC_NOINLINE inline void update_triangle_quality(state& s) noexcept {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_quality = s.quality.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    hpc::array<hpc::basis_gradient<double>, 3> grad_N;
    for (auto const i : nodes_in_element) {
      grad_N[hpc::weaken(i)] = point_nodes_to_grad_N[point_nodes[i]].load();
    }
    auto const A = points_to_V[point] / hpc::length<double>(1.0);
    auto const fast_quality = triangle_quality(grad_N, A);
    elements_to_quality[element] = fast_quality;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

/* Per:
   Shewchuk, Jonathan Richard.
   "What is a good linear finite element?
    interpolation, conditioning, anisotropy, and quality measures (preprint)."
   University of California at Berkeley 73 (2002): 137.

   A scale-invariant quality measure correlated with matrix conditioning
   (and the stable explicit time step) for tetrahedra is volume divided by
   the (root-mean-squared side area) to the (3/2) power.
   We also use the fact that we already have volume computed and that basis
   gradient magnitudes relate to opposite side areas.

   Furthermore, we raise the entire quantity to the fourth power in order to
   avoid having to compute any roots, since only relative comparison of
   qualities is necessary.

   As such, our "quality" is the inverse of this quality measure to the fourth power
  */
HPC_NOINLINE inline void update_tetrahedron_quality(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_quality = s.quality.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    decltype(1.0 / hpc::area<double>()) sum_g_i_sq = 0.0;
    for (auto const i : nodes_in_element) {
      auto const grad_N = point_nodes_to_grad_N[point_nodes[i]].load();
      auto const g_i_sq = (grad_N * grad_N);
      sum_g_i_sq += g_i_sq;
    }
    auto const V = points_to_V[point];
    elements_to_quality[element] = (V * V) * (sum_g_i_sq * sum_g_i_sq * sum_g_i_sq);
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void update_quality(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_quality(s); break;
    case TRIANGLE: update_triangle_quality(s); break;
    case TETRAHEDRON: update_tetrahedron_quality(s); break;
    case COMPOSITE_TETRAHEDRON: assert(0); break;
  }
}

void update_min_quality(state& s) {
  hpc::dimensionless<double> const init = std::numeric_limits<double>::max();
  s.min_quality = hpc::transform_reduce(
      hpc::device_policy(),
      s.quality,
      init,
      hpc::minimum<hpc::dimensionless<double>>(),
      hpc::identity<hpc::dimensionless<double>>());
}

void initialize_h_adapt(state& s)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_h_adapt = s.h_adapt.begin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const x = nodes_to_x[node].load();
    hpc::area<double> lsq_max = 0.0;
    hpc::area<double> lsq_min = hpc::numeric_limits<double>::max();
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element]; 
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node != node) {
          auto const adj_x = nodes_to_x[adj_node].load();
          auto const lsq = norm_squared(adj_x - x);
          lsq_max = hpc::max(lsq_max, lsq);
          lsq_min = hpc::min(lsq_min, lsq);
        }
      }
    }
    auto const h_min = sqrt(lsq_min);
    auto const h_max = sqrt(lsq_max);
    auto const alpha = sqrt(h_max / h_min);
    auto const h_avg = h_min * alpha;
    nodes_to_h_adapt[node] = h_avg;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

enum cavity_op {
  NONE,
  SWAP,
  SPLIT,
  COLLAPSE,
};

struct adapt_state {
  hpc::device_vector<double, node_index> criteria;
  hpc::device_vector<node_index, node_index> other_node;
  hpc::device_vector<cavity_op, node_index> op;
  hpc::device_vector<element_index, element_index> element_counts;
  hpc::device_vector<node_index, node_index> node_counts;
  hpc::device_vector<element_index, element_index> old_elements_to_new_elements;
  hpc::device_vector<node_index, node_index> old_nodes_to_new_nodes;
  hpc::device_vector<element_index, element_index> new_elements_to_old_elements;
  hpc::device_vector<node_index, node_index> new_nodes_to_old_nodes;
  hpc::device_vector<node_index, element_node_index> new_element_nodes_to_nodes;
  hpc::device_vector<bool, element_index> new_elements_are_same;
  hpc::device_vector<bool, node_index> new_nodes_are_same;
  hpc::device_array_vector<hpc::array<node_index, 2, int>, node_index> interpolate_from;
  hpc::counting_range<element_index> new_elements;
  hpc::counting_range<node_index> new_nodes;
  adapt_state(state const&);
};

adapt_state::adapt_state(state const& s)
  :criteria(s.nodes.size())
  ,other_node(s.nodes.size())
  ,op(s.nodes.size())
  ,element_counts(s.elements.size())
  ,node_counts(s.nodes.size())
  ,old_elements_to_new_elements(s.elements.size() + element_index(1))
  ,old_nodes_to_new_nodes(s.nodes.size() + node_index(1))
  ,new_elements_to_old_elements()
  ,new_nodes_to_old_nodes()
  ,new_element_nodes_to_nodes()
  ,new_elements_are_same()
  ,new_nodes_are_same()
  ,interpolate_from()
  ,new_elements(element_index(0))
  ,new_nodes(node_index(0))
{}

template <std::ptrdiff_t Capacity, class Index>
inline HPC_HOST_DEVICE int find_or_append(
    int& count,
    hpc::array<Index, Capacity>& buffer,
    Index const index)
{
  for (int i = 0; i < count; ++i) {
    if (buffer[i] == index) return i;
  }
  assert(count < Capacity);
  buffer[count] = index;
  return count++;
}

template <int nodes_per_element, int max_shell_elements, int max_shell_nodes>
struct eval_cavity {
  int num_shell_elements;
  int num_shell_nodes;
  hpc::array<node_index, max_shell_nodes> shell_nodes;
  hpc::array<element_index, max_shell_elements> shell_elements;
  hpc::array<hpc::array<int, nodes_per_element>, max_shell_elements> shell_elements_to_shell_nodes;
  hpc::array<material_index, max_shell_elements> shell_elements_to_materials;
  hpc::array<hpc::dimensionless<double>, max_shell_elements> shell_element_qualities;
  hpc::array<int, max_shell_elements> shell_elements_to_node_in_element;
  hpc::array<hpc::position<double>, max_shell_nodes> shell_nodes_to_x;
  hpc::array<hpc::length<double>, max_shell_nodes> shell_nodes_to_h;
  hpc::array<material_set, max_shell_nodes> shell_nodes_to_materials;
};

template <int max_shell_elements, int max_shell_nodes>
inline HPC_DEVICE void evaluate_triangle_swap(
    int const center_node,
    int const edge_node,
    eval_cavity<3, max_shell_elements, max_shell_nodes> const c,
    hpc::dimensionless<double>& best_improvement,
    int& best_swap_edge_node
    ) {
  hpc::array<int, 2> loop_elements;
  loop_elements[0] = loop_elements[1] = -1;
  hpc::array<int, 2> loop_nodes;
  loop_nodes[0] = loop_nodes[1] = -1;
  for (int element = 0; element < c.num_shell_elements; ++element) {
    int const node_in_element = c.shell_elements_to_node_in_element[element];
    if (c.shell_elements_to_shell_nodes[element][(node_in_element + 1) % 3] == edge_node) {
      loop_elements[1] = element;
      loop_nodes[1] = c.shell_elements_to_shell_nodes[element][(node_in_element + 2) % 3];
    }
    if (c.shell_elements_to_shell_nodes[element][(node_in_element + 2) % 3] == edge_node) {
      loop_elements[0] = element;
      loop_nodes[0] = c.shell_elements_to_shell_nodes[element][(node_in_element + 1) % 3];
    }
  }
  if (loop_elements[0] == -1 || loop_elements[1] == -1) {
    return;
  }
  if (c.shell_elements_to_materials[loop_elements[0]] != c.shell_elements_to_materials[loop_elements[1]]) {
    return;
  }
  auto const old_quality1 = c.shell_element_qualities[loop_elements[0]];
  auto const old_quality2 = c.shell_element_qualities[loop_elements[1]];
  auto const quality_before = hpc::min(old_quality1, old_quality2);
  assert(quality_before > 0.0);
  hpc::array<hpc::position<double>, 3> proposed_x;
  proposed_x[0] = c.shell_nodes_to_x[center_node];
  proposed_x[1] = c.shell_nodes_to_x[loop_nodes[0]];
  proposed_x[2] = c.shell_nodes_to_x[loop_nodes[1]];
  auto const new_quality1 = triangle_quality(proposed_x);
  if (new_quality1 <= quality_before) return;
  proposed_x[0] = c.shell_nodes_to_x[edge_node];
  proposed_x[1] = c.shell_nodes_to_x[loop_nodes[1]];
  proposed_x[2] = c.shell_nodes_to_x[loop_nodes[0]];
  auto const new_quality2 = triangle_quality(proposed_x);
  if (new_quality2 <= quality_before) return;
  auto const quality_after = hpc::min(new_quality1, new_quality2);
  auto const improvement = ((quality_after - quality_before) / quality_before);
  if (improvement < 0.05) return;
  if (improvement > best_improvement) {
    best_improvement = improvement;
    best_swap_edge_node = edge_node;
  }
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE hpc::dimensionless<double> measure_edge(
    hpc::length<double> const h_min,
    hpc::length<double> const h_max,
    hpc::length<double> const l) noexcept {
  return l / (0.5 * (h_min + h_max));
}

template <int max_shell_elements, int max_shell_nodes>
inline HPC_DEVICE void evaluate_triangle_split(
    int const center_node,
    int const edge_node,
    eval_cavity<3, max_shell_elements, max_shell_nodes> const c,
    hpc::dimensionless<double>& longest_length,
    int& best_split_edge_node
    ) {
  constexpr double min_acceptable_quality = 0.2;
  auto const h1 = c.shell_nodes_to_h[center_node];
  auto const h2 = c.shell_nodes_to_h[edge_node];
  auto const x1 = c.shell_nodes_to_x[center_node];
  auto const x2 = c.shell_nodes_to_x[edge_node];
  auto const h_min = hpc::min(h1, h2);
  auto const h_max = hpc::max(h1, h2);
  auto const l = norm(x1 - x2);
  auto const lm = measure_edge(h_min, h_max, l);
  if (lm <= std::sqrt(2.0)) return;
  if (lm <= longest_length) return;
  auto const midpoint_x = 0.5 * (x1 + x2);
  for (int element = 0; element < c.num_shell_elements; ++element) {
    hpc::array<hpc::position<double>, 3> parent_x;
    int const center_node_in_element = c.shell_elements_to_node_in_element[element];
    int edge_node_in_element = -1;
    for (int node_in_element = 0; node_in_element < 3; ++node_in_element) {
      int const shell_node = c.shell_elements_to_shell_nodes[element][node_in_element];
      parent_x[node_in_element] = c.shell_nodes_to_x[shell_node];
      if (shell_node == edge_node) edge_node_in_element = node_in_element;
    }
    if (edge_node_in_element == -1) continue;
    hpc::array<hpc::position<double>, 3> child_x = parent_x;
    child_x[center_node_in_element] = midpoint_x;
    auto const new_quality1 = triangle_quality(child_x);
    if (new_quality1 < min_acceptable_quality) return;
    child_x[center_node_in_element] = c.shell_nodes_to_x[center_node];
    child_x[edge_node_in_element] = midpoint_x;
    auto const new_quality2 = triangle_quality(child_x);
    if (new_quality2 < min_acceptable_quality) return;
  }
  longest_length = lm;
  best_split_edge_node = edge_node;
}

template <int max_shell_elements, int max_shell_nodes>
inline HPC_DEVICE void evaluate_triangle_collapse(
    int const center_node,
    int const edge_node,
    eval_cavity<3, max_shell_elements, max_shell_nodes> const c,
    material_set const boundary_materials,
    hpc::dimensionless<double>& shortest_length,
    int& best_collapse_edge_node
    ) {
  if (!c.shell_nodes_to_materials[edge_node].contains(c.shell_nodes_to_materials[center_node])) return;
  constexpr double min_acceptable_quality = 0.2;
  auto const h1 = c.shell_nodes_to_h[center_node];
  auto const h2 = c.shell_nodes_to_h[edge_node];
  auto const x1 = c.shell_nodes_to_x[center_node];
  auto const x2 = c.shell_nodes_to_x[edge_node];
  auto const h_min = hpc::min(h1, h2);
  auto const h_max = hpc::max(h1, h2);
  auto const l = norm(x1 - x2);
  auto const lm = measure_edge(h_min, h_max, l);
  if (lm >= (1.0 / std::sqrt(2.0))) {
    return;
  }
  if (lm >= shortest_length) {
    return;
  }
  auto edge_materials = material_set::none();
  for (int element = 0; element < c.num_shell_elements; ++element) {
    hpc::array<hpc::position<double>, 3> proposed_x;
    int const center_node_in_element = c.shell_elements_to_node_in_element[element];
    int edge_node_in_element = -1;
    for (int node_in_element = 0; node_in_element < 3; ++node_in_element) {
      int const shell_node = c.shell_elements_to_shell_nodes[element][node_in_element];
      proposed_x[node_in_element] = c.shell_nodes_to_x[shell_node];
      if (shell_node == edge_node) edge_node_in_element = node_in_element;
      material_index const element_material = c.shell_elements_to_materials[element];
      edge_materials = edge_materials | material_set(element_material);
    }
    if (edge_node_in_element != -1) continue;
    proposed_x[center_node_in_element] = c.shell_nodes_to_x[edge_node];
    auto const new_quality = triangle_quality(proposed_x);
    if (new_quality < min_acceptable_quality) {
      return;
    }
  }
  auto const center_materials = c.shell_nodes_to_materials[center_node];
  auto const target_materials = c.shell_nodes_to_materials[edge_node];
  if ((center_materials - boundary_materials) == edge_materials && target_materials.contains(center_materials)) {
    shortest_length = lm;
    best_collapse_edge_node = edge_node;
  }
}

HPC_NOINLINE inline void evaluate_triangle_adapt(input const& in, state const& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_node_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const elements_to_materials = s.material.cbegin();
  auto const elements_to_qualities = s.quality.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_h = s.h_adapt.cbegin();
  auto const nodes_to_materials = s.nodal_materials.cbegin();
  auto const nodes_to_criteria = a.criteria.begin();
  auto const nodes_to_other_nodes = a.other_node.begin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_to_op = a.op.begin();
  auto const boundary_materials = material_set::all(in.materials.size() + in.boundaries.size()) - material_set::all(in.materials.size());
  auto functor = [=] HPC_DEVICE (node_index const node) {
    eval_cavity<3, 32, 32> c;
    c.num_shell_nodes = 0;
    c.num_shell_elements = 0;
    int center_node = -1;
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element];
      int const shell_element = c.num_shell_elements++;
      c.shell_elements[shell_element] = element;
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const node2 = element_nodes_to_nodes[element_node];
        int const shell_node = find_or_append(c.num_shell_nodes, c.shell_nodes, node2);
        if (node2 == node) center_node = shell_node;
        if (shell_node + 1 == c.num_shell_nodes) {
          c.shell_nodes_to_x[shell_node] = nodes_to_x[node2].load();
          c.shell_nodes_to_h[shell_node] = nodes_to_h[node2];
          c.shell_nodes_to_materials[shell_node] = nodes_to_materials[node2];
        }
        c.shell_elements_to_shell_nodes[shell_element][hpc::weaken(node_in_element)] = shell_node;
      }
      material_index const material = elements_to_materials[element];
      c.shell_elements_to_materials[shell_element] = material;
      auto const quality = elements_to_qualities[element];
      c.shell_element_qualities[shell_element] = quality;
      node_in_element_index const node_in_element = node_elements_to_node_in_element[node_element];
      c.shell_elements_to_node_in_element[shell_element] = hpc::weaken(node_in_element);
    }
    hpc::dimensionless<double> best_swap_improvement = 0.0;
    int best_swap_edge_node = -1;
    hpc::dimensionless<double> longest_split_edge = 0.0;
    int best_split_edge_node = -1;
    hpc::dimensionless<double> shortest_collapse_edge = 1.0;
    int best_collapse_edge_node = -1;
    for (int edge_node = 0; edge_node < c.num_shell_nodes; ++edge_node) {
      if (edge_node == center_node) continue;
      if (c.shell_nodes[center_node] < c.shell_nodes[edge_node]) {
        // swaps and splits are non-directional, they only need to be
        // examined by one of the nodes
        evaluate_triangle_swap(center_node, edge_node, c,
            best_swap_improvement, best_swap_edge_node);
        evaluate_triangle_split(center_node, edge_node, c,
            longest_split_edge, best_split_edge_node);
      }
      evaluate_triangle_collapse(center_node, edge_node, c,
          boundary_materials,
          shortest_collapse_edge, best_collapse_edge_node);
    }
    if (best_collapse_edge_node != -1) {
      nodes_to_criteria[node] = double(1.0 / shortest_collapse_edge);
      nodes_to_other_nodes[node] = c.shell_nodes[best_collapse_edge_node];
      nodes_to_op[node] = cavity_op::COLLAPSE;
    } else if (best_split_edge_node != -1) {
      nodes_to_criteria[node] = double(longest_split_edge);
      nodes_to_other_nodes[node] = c.shell_nodes[best_split_edge_node];
      nodes_to_op[node] = cavity_op::SPLIT;
    } else if (best_swap_edge_node != -1) {
      nodes_to_criteria[node] = double(best_swap_improvement);
      nodes_to_other_nodes[node] = c.shell_nodes[best_swap_edge_node];
      nodes_to_op[node] = cavity_op::SWAP;
    } else {
      nodes_to_other_nodes[node] = node_index(-1);
      nodes_to_op[node] = cavity_op::NONE;
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

HPC_NOINLINE inline void choose_triangle_adapt(state const& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_criteria = a.criteria.cbegin();
  auto const nodes_to_other_nodes = a.other_node.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  hpc::fill(hpc::device_policy(), a.element_counts, element_index(1));
  hpc::fill(hpc::device_policy(), a.node_counts, node_index(1));
  auto const elements_to_new_counts = a.element_counts.begin();
  auto const nodes_to_new_counts = a.node_counts.begin();
  auto const nodes_to_op = a.op.begin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    cavity_op const op = nodes_to_op[node];
    if (op == cavity_op::NONE) {
      return;
    }
    node_index const target_node = nodes_to_other_nodes[node];
    double const criteria = nodes_to_criteria[node];
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node != node) {
          cavity_op const adj_op = nodes_to_op[adj_node];
          double const adj_criteria = nodes_to_criteria[adj_node];
          if (op < adj_op) {
            nodes_to_op[node] = cavity_op::NONE;
            return;
          }
          if (op > adj_op) continue;
          if (criteria < adj_criteria) {
            nodes_to_op[node] = cavity_op::NONE;
            return;
          }
          if (criteria > adj_criteria) continue;
          if (adj_node < node) {
            nodes_to_op[node] = cavity_op::NONE;
            return;
          }
        }
      }
    }
    element_index edge_element_count(-100);
    if (op == cavity_op::SWAP) {
      edge_element_count = element_index(1);
    } else if (op == cavity_op::SPLIT) {
      edge_element_count = element_index(2);
    } else if (op == cavity_op::COLLAPSE) {
      edge_element_count = element_index(0);
    }
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node == target_node) { // element is adjacent to the edge
          elements_to_new_counts[element] = edge_element_count;
        }
      }
    }
    node_index node_count(-100);
    if (op == cavity_op::SWAP) {
      node_count = node_index(1);
    } else if (op == cavity_op::SPLIT) {
      node_count = node_index(2);
    } else if (op == cavity_op::COLLAPSE) {
      node_count = node_index(0);
    }
    nodes_to_new_counts[node] = node_count;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

struct apply_cavity {
  hpc::range_sum_iterator<node_element_index, node_index> nodes_to_node_elements;
  hpc::pointer_iterator<element_index const, node_element_index> node_elements_to_elements;
  hpc::pointer_iterator<node_in_element_index const, node_element_index> node_elements_to_nodes_in_element;
  hpc::counting_product<hpc::device_layout, element_index, node_in_element_index> elements_to_element_nodes;
  hpc::pointer_iterator<node_index const, element_node_index> old_element_nodes_to_nodes;
  hpc::pointer_iterator<element_index const, element_index> old_elements_to_new_elements;
  hpc::counting_product<hpc::device_layout, element_index, node_in_element_index> new_elements_to_element_nodes;
  hpc::pointer_iterator<node_index const, node_index> old_nodes_to_new_nodes;
  hpc::pointer_iterator<node_index, element_node_index> new_element_nodes_to_nodes;
  hpc::pointer_iterator<bool, element_index> new_elements_are_same;
  hpc::pointer_iterator<node_index const, node_index> nodes_to_other_nodes;
  hpc::counting_range<node_in_element_index> nodes_in_element;
  hpc::pointer_iterator<bool, node_index> new_nodes_are_same;
  hpc::array_vector_iterator<hpc::array<node_index, 2, int>, hpc::device_layout, node_index> interpolate_from;
  apply_cavity(state const& s, adapt_state& a)
    :nodes_to_node_elements(s.nodes_to_node_elements.cbegin())
    ,node_elements_to_elements(s.node_elements_to_elements.cbegin())
    ,node_elements_to_nodes_in_element(s.node_elements_to_nodes_in_element.cbegin())
    ,elements_to_element_nodes(s.elements * s.nodes_in_element)
    ,old_element_nodes_to_nodes(s.elements_to_nodes.cbegin())
    ,old_elements_to_new_elements(a.old_elements_to_new_elements.cbegin())
    ,new_elements_to_element_nodes(a.new_elements * s.nodes_in_element)
    ,old_nodes_to_new_nodes(a.old_nodes_to_new_nodes.cbegin())
    ,new_element_nodes_to_nodes(a.new_element_nodes_to_nodes.begin())
    ,new_elements_are_same(a.new_elements_are_same.begin())
    ,nodes_to_other_nodes(a.other_node.cbegin())
    ,nodes_in_element(s.nodes_in_element)
    ,new_nodes_are_same(a.new_nodes_are_same.begin())
    ,interpolate_from(a.interpolate_from.begin())
  {}
};

inline HPC_DEVICE void apply_triangle_swap(apply_cavity const c,
    node_index const node,
    node_index const target_node) {
  hpc::array<element_index, 2> loop_elements;
  loop_elements[0] = -1;
  loop_elements[1] = -1;
  hpc::array<node_index, 2> loop_nodes;
  loop_nodes[0] = -1;
  loop_nodes[1] = -1;
  for (auto const node_element : c.nodes_to_node_elements[node]) {
    element_index const element = c.node_elements_to_elements[node_element];
    auto const element_nodes = c.elements_to_element_nodes[element];
    node_in_element_index const node_in_element = c.node_elements_to_nodes_in_element[node_element];
    node_in_element_index const plus1 = node_in_element_index((hpc::weaken(node_in_element) + 1) % 3);
    node_in_element_index const plus2 = node_in_element_index((hpc::weaken(node_in_element) + 2) % 3);
    node_index const node1 = c.old_element_nodes_to_nodes[element_nodes[plus1]];
    node_index const node2 = c.old_element_nodes_to_nodes[element_nodes[plus2]];
    if (node1 == target_node) {
      loop_elements[1] = element;
      loop_nodes[1] = node2;
    }
    if (node2 == target_node) {
      loop_elements[0] = element;
      loop_nodes[0] = node1;
    }
  }
  element_index const new_element1 = c.old_elements_to_new_elements[loop_elements[0]];
  element_index const new_element2 = c.old_elements_to_new_elements[loop_elements[1]];
  using l_t = node_in_element_index;
  auto new_element_nodes = c.new_elements_to_element_nodes[new_element1];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(0)]] = c.old_nodes_to_new_nodes[node];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(1)]] = c.old_nodes_to_new_nodes[loop_nodes[0]];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(2)]] = c.old_nodes_to_new_nodes[loop_nodes[1]];
  new_element_nodes = c.new_elements_to_element_nodes[new_element2];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(0)]] = c.old_nodes_to_new_nodes[target_node];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(1)]] = c.old_nodes_to_new_nodes[loop_nodes[1]];
  c.new_element_nodes_to_nodes[new_element_nodes[l_t(2)]] = c.old_nodes_to_new_nodes[loop_nodes[0]];
  c.new_elements_are_same[new_element1] = false;
  c.new_elements_are_same[new_element2] = false;
}

inline HPC_DEVICE void apply_triangle_split(apply_cavity const c,
    node_index const center_node,
    node_index const target_node) {
  node_index const new_center_node = c.old_nodes_to_new_nodes[center_node];
  auto const split_node = new_center_node + node_index(1);
  for (auto const node_element : c.nodes_to_node_elements[center_node]) {
    element_index const element = c.node_elements_to_elements[node_element];
    auto const old_element_nodes = c.elements_to_element_nodes[element];
    node_in_element_index target_node_in_element(-1);
    hpc::array<node_index, 3, node_in_element_index> new_nodes;
    for (auto const node_in_element : c.nodes_in_element) {
      auto const old_element_node = old_element_nodes[node_in_element];
      node_index const old_node = c.old_element_nodes_to_nodes[old_element_node];
      if (old_node == target_node) target_node_in_element = node_in_element;
      auto const new_node = c.old_nodes_to_new_nodes[old_node];
      new_nodes[node_in_element] = new_node;
    }
    if (target_node_in_element == node_in_element_index(-1)) continue;
    node_in_element_index const center_node_in_element = c.node_elements_to_nodes_in_element[node_element];
    element_index const new_element1 = c.old_elements_to_new_elements[element];
    auto const new_element_nodes1 = c.new_elements_to_element_nodes[new_element1];
    new_nodes[center_node_in_element] = split_node;
    for (auto const node_in_element : c.nodes_in_element) {
      auto const new_element_node = new_element_nodes1[node_in_element];
      c.new_element_nodes_to_nodes[new_element_node] = new_nodes[node_in_element];
    }
    element_index const new_element2 = new_element1 + element_index(1);
    auto const new_element_nodes2 = c.new_elements_to_element_nodes[new_element2];
    new_nodes[center_node_in_element] = new_center_node;
    new_nodes[target_node_in_element] = split_node;
    for (auto const node_in_element : c.nodes_in_element) {
      auto const new_element_node = new_element_nodes2[node_in_element];
      c.new_element_nodes_to_nodes[new_element_node] = new_nodes[node_in_element];
    }
    c.new_elements_are_same[new_element1] = false;
    c.new_elements_are_same[new_element2] = false;
  }
  c.new_nodes_are_same[split_node] = false;
  hpc::array<node_index, 2, int> interpolate_from;
  interpolate_from[0] = center_node;
  interpolate_from[1] = target_node;
  c.interpolate_from[split_node] = interpolate_from;
}

inline HPC_DEVICE void apply_triangle_collapse(apply_cavity const c,
    node_index const center_node,
    node_index const target_node) {
  node_index const new_target_node = c.old_nodes_to_new_nodes[target_node];
  for (auto const node_element : c.nodes_to_node_elements[center_node]) {
    element_index const element = c.node_elements_to_elements[node_element];
    auto const old_element_nodes = c.elements_to_element_nodes[element];
    node_in_element_index target_node_in_element(-1);
    hpc::array<node_index, 3, node_in_element_index> new_nodes;
    for (auto const node_in_element : c.nodes_in_element) {
      auto const old_element_node = old_element_nodes[node_in_element];
      node_index const old_node = c.old_element_nodes_to_nodes[old_element_node];
      if (old_node == target_node) target_node_in_element = node_in_element;
      auto const new_node = c.old_nodes_to_new_nodes[old_node];
      new_nodes[node_in_element] = new_node;
    }
    if (target_node_in_element != node_in_element_index(-1)) continue;
    node_in_element_index const center_node_in_element = c.node_elements_to_nodes_in_element[node_element];
    element_index const new_element = c.old_elements_to_new_elements[element];
    auto const new_element_nodes = c.new_elements_to_element_nodes[new_element];
    new_nodes[center_node_in_element] = new_target_node;
    for (auto const node_in_element : c.nodes_in_element) {
      auto const new_element_node = new_element_nodes[node_in_element];
      c.new_element_nodes_to_nodes[new_element_node] = new_nodes[node_in_element];
    }
    c.new_elements_are_same[new_element] = false;
  }
}

HPC_NOINLINE inline void apply_triangle_adapt(state const& s, adapt_state& a)
{
  apply_cavity c(s, a);
  hpc::fill(hpc::device_policy(), a.new_elements_are_same, true);
  hpc::fill(hpc::device_policy(), a.new_nodes_are_same, true);
  c.new_elements_are_same = a.new_elements_are_same.begin();
  auto const nodes_to_op = a.op.cbegin();
  auto const nodes_to_other_nodes = a.other_node.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    cavity_op const op = nodes_to_op[node];
    if (cavity_op::NONE == op) return;
    node_index const target_node = nodes_to_other_nodes[node];
    if (cavity_op::SWAP == op) apply_triangle_swap(c, node, target_node);
    else if (cavity_op::SPLIT == op) apply_triangle_split(c, node, target_node);
    else if (cavity_op::COLLAPSE == op) apply_triangle_collapse(c, node, target_node);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

template <class Index>
HPC_NOINLINE void project(
    hpc::counting_range<Index> const old_things,
    hpc::device_vector<Index, Index> const& old_things_to_new_things_in,
    hpc::device_vector<Index, Index>& new_things_to_old_things_in) {
  auto const old_things_to_new_things = old_things_to_new_things_in.cbegin();
  auto const new_things_to_old_things = new_things_to_old_things_in.begin();
  auto functor = [=] HPC_DEVICE (Index const old_thing) {
    Index first = old_things_to_new_things[old_thing];
    Index const last = old_things_to_new_things[old_thing + Index(1)];
    for (; first < last; ++first) {
      new_things_to_old_things[first] = old_thing;
    }
  };
  hpc::for_each(hpc::device_policy(), old_things, functor);
}

HPC_NOINLINE inline void transfer_same_connectivity(state const& s, adapt_state& a) {
  auto const new_elements_to_element_nodes = a.new_elements * s.nodes_in_element;
  auto const old_elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const old_element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const new_element_nodes_to_nodes = a.new_element_nodes_to_nodes.begin();
  auto const new_elements_are_same = a.new_elements_are_same.cbegin();
  auto const old_nodes_to_new_nodes = a.old_nodes_to_new_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (element_index const new_element) {
    if (new_elements_are_same[new_element]) {
      auto const new_element_nodes = new_elements_to_element_nodes[new_element];
      element_index const old_element = new_elements_to_old_elements[new_element];
      auto const old_element_nodes = old_elements_to_element_nodes[old_element];
      for (auto const node_in_element : nodes_in_element) {
        auto const new_element_node = new_element_nodes[node_in_element];
        auto const old_element_node = old_element_nodes[node_in_element];
        node_index const old_node = node_index(old_element_nodes_to_nodes[old_element_node]);
        node_index const new_node = old_nodes_to_new_nodes[old_node];
        new_element_nodes_to_nodes[new_element_node] = new_node;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), a.new_elements, functor);
}

HPC_NOINLINE inline void transfer_element_materials(adapt_state& a,
    hpc::device_vector<material_index, element_index>& data) {
  hpc::device_vector<material_index, element_index> new_data(a.new_elements.size());
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const old_elements_to_T = data.cbegin();
  auto const new_elements_to_T = new_data.begin();
  auto functor = [=] HPC_DEVICE (element_index const new_element) {
    element_index const old_element = new_elements_to_old_elements[new_element];
    new_elements_to_T[new_element] =
      material_index(old_elements_to_T[old_element]);
  };
  hpc::for_each(hpc::device_policy(), a.new_elements, functor);
  data = std::move(new_data);
}

template <class Range>
HPC_NOINLINE void transfer_point_data(state const& s, adapt_state const& a,
    Range& data) {
  auto const points_in_element = s.points_in_element;
  using value_type = typename Range::value_type;
  Range new_data(a.new_elements.size() * points_in_element.size());
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const old_points_to_T = data.cbegin();
  auto const new_points_to_T = new_data.begin();
  auto const old_elements_to_points = s.elements * points_in_element;
  auto const new_elements_to_points = a.new_elements * points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const new_element) {
    element_index const old_element = new_elements_to_old_elements[new_element];
    auto const new_element_points = new_elements_to_points[new_element];
    auto const old_element_points = old_elements_to_points[old_element];
    for (auto const point_in_element : points_in_element) {
      auto const new_point = new_element_points[point_in_element];
      auto const old_point = old_element_points[point_in_element];
      auto const old_value = value_type(old_points_to_T[old_point]);
      new_points_to_T[new_point] = old_value;
    }
  };
  hpc::for_each(hpc::device_policy(), a.new_elements, functor);
  data = std::move(new_data);
}

HPC_NOINLINE inline void transfer_nodal_energy(input const& in, adapt_state const& a, state& s) {
  auto const new_nodes_to_old_nodes = a.new_nodes_to_old_nodes.cbegin();
  for (auto const material : in.materials) {
    if (!in.enable_nodal_energy[material]) continue;
    hpc::device_vector<hpc::specific_energy<double>, node_index>& old_data = s.e_h[material];
    hpc::device_vector<hpc::specific_energy<double>, node_index> new_data(a.new_nodes.size());
    auto const old_nodes_to_T = old_data.cbegin();
    auto const new_nodes_to_T = new_data.begin();
    auto functor = [=] HPC_DEVICE (node_index const new_node) {
      auto const old_node = new_nodes_to_old_nodes[new_node];
      auto const old_value = old_nodes_to_T[old_node];
      assert(old_value > 0.0);
      new_nodes_to_T[new_node] = old_value;
    };
    // FIXME: this could be run over just the new nodes touching this material,
    // but we can't do that right now because we don't compute that set of nodes
    // until after all the transfers
    hpc::for_each(hpc::device_policy(), a.new_nodes, functor);
    old_data = std::move(new_data);
  }
}

template <class Range>
HPC_NOINLINE void interpolate_nodal_data(adapt_state const& a, Range& data) {
  using value_type = typename Range::value_type;
  Range new_data(a.new_nodes.size());
  auto const old_nodes_to_data = data.cbegin();
  auto const new_nodes_to_data = new_data.begin();
  auto const new_nodes_to_old_nodes = a.new_nodes_to_old_nodes.cbegin();
  auto const new_nodes_are_same = a.new_nodes_are_same.cbegin();
  auto const interpolate_from = a.interpolate_from.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const new_node) {
    if (new_nodes_are_same[new_node]) {
      node_index const old_node = new_nodes_to_old_nodes[new_node];
      new_nodes_to_data[new_node] = value_type(old_nodes_to_data[old_node]);
    } else {
      auto const pair = interpolate_from[new_node].load();
      node_index const left = pair[0];
      node_index const right = pair[1];
      new_nodes_to_data[new_node] = 0.5 * (
          value_type(old_nodes_to_data[left])
        + value_type(old_nodes_to_data[right]));
    }
  };
  hpc::for_each(hpc::device_policy(), a.new_nodes, functor);
  data = std::move(new_data);
}

bool adapt(input const& in, state& s) {
  adapt_state a(s);
  evaluate_triangle_adapt(in, s, a);
  choose_triangle_adapt(s, a);
  auto const num_chosen = hpc::transform_reduce(hpc::device_policy(), a.op, int(0), hpc::plus<int>(),
      [] HPC_DEVICE (cavity_op const op) { return op == cavity_op::NONE ? 0 : 1; });
  if (num_chosen == 0) return false;
  if (in.output_to_command_line) {
    std::cout << "adapting " << num_chosen << " cavities\n";
  }
  auto const num_new_elements = hpc::reduce(hpc::device_policy(), a.element_counts, element_index(0));
  auto const num_new_nodes = hpc::reduce(hpc::device_policy(), a.node_counts, node_index(0));
  hpc::offset_scan(hpc::device_policy(), a.element_counts, a.old_elements_to_new_elements);
  hpc::offset_scan(hpc::device_policy(), a.node_counts, a.old_nodes_to_new_nodes);
  a.new_elements.resize(num_new_elements);
  a.new_nodes.resize(num_new_nodes);
  a.new_elements_to_old_elements.resize(num_new_elements);
  a.new_nodes_to_old_nodes.resize(num_new_nodes);
  a.new_element_nodes_to_nodes.resize(num_new_elements * s.nodes_in_element.size());
  a.new_elements_are_same.resize(num_new_elements);
  a.new_nodes_are_same.resize(num_new_nodes);
  a.interpolate_from.resize(num_new_nodes);
  project(s.elements, a.old_elements_to_new_elements, a.new_elements_to_old_elements);
  project(s.nodes, a.old_nodes_to_new_nodes, a.new_nodes_to_old_nodes);
  apply_triangle_adapt(s, a);
  transfer_same_connectivity(s, a);
  transfer_element_materials(a, s.material);
  transfer_point_data(s, a, s.rho);
  if (!hpc::all_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    transfer_point_data(s, a, s.e);
  } else {
    transfer_nodal_energy(in, a, s);
  }
  transfer_point_data(s, a, s.F_total);
  interpolate_nodal_data(a, s.x);
  interpolate_nodal_data(a, s.v);
  interpolate_nodal_data(a, s.h_adapt);
  s.elements = a.new_elements;
  s.nodes = a.new_nodes;
  s.elements_to_nodes = std::move(a.new_element_nodes_to_nodes);
  propagate_connectivity(s);
  compute_nodal_materials(in, s);
  return true;
}

}
