#include <lgr_macros.hpp>
#include <lgr_state.hpp>
#include <lgr_input.hpp>
#include <lgr_adapt.hpp>
#include <lgr_array.hpp>
#include <lgr_element_specific_inline.hpp>
#include <lgr_print.hpp>
#include <lgr_reduce.hpp>
#include <lgr_meshing.hpp>

#include <iostream>
#include <iomanip>

namespace lgr {

static void LGR_NOINLINE update_bar_quality(state& s) {
  fill(s.quality, double(1.0));
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
inline double triangle_quality(array<vector3<double>, 3> const grad_N, double const area) {
  double sum_g_i_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    auto const g_i_sq = (grad_N[i] * grad_N[i]);
    sum_g_i_sq += g_i_sq;
  }
  double const denom = (area * sum_g_i_sq);
  double const q = 1.0 / denom;
  return q;
}

inline double triangle_quality(array<vector3<double>, 3> const x) {
  double const area = triangle_area(x);
  if (area <= 0.0) return area;
  return triangle_quality(triangle_basis_gradients(x, area), area);
}

static void LGR_NOINLINE update_triangle_quality(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_quality = s.quality.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    array<vector3<double>, 3> grad_N;
    for (auto const i : nodes_in_element) {
      grad_N[int(i)] = point_nodes_to_grad_N[point_nodes[i]];
    }
    auto const A = points_to_V[point];
    double const fast_quality = triangle_quality(grad_N, A);
    elements_to_quality[element] = fast_quality;
  };
  lgr::for_each(s.elements, functor);
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
static void LGR_NOINLINE update_tetrahedron_quality(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_quality = s.quality.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    auto sum_g_i_sq = 0.0;
    for (auto const i : nodes_in_element) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_nodes[i]];
      auto const g_i_sq = (grad_N * grad_N);
      sum_g_i_sq += g_i_sq;
    }
    auto const V = points_to_V[point];
    elements_to_quality[element] = (V * V) * (sum_g_i_sq * sum_g_i_sq * sum_g_i_sq);
  };
  lgr::for_each(s.elements, functor);
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
  s.min_quality = transform_reduce(s.quality,
      std::numeric_limits<double>::max(),
      minimum<double>(),
      identity<double>());
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
  auto functor = [=] (node_index const node) {
    vector3<double> const x = nodes_to_x[node];
    double lsq_max = 0.0;
    double lsq_min = std::numeric_limits<double>::max();
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element]; 
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node != node) {
          vector3<double> const adj_x = nodes_to_x[adj_node];
          auto const lsq = norm_squared(adj_x - x);
          lsq_max = max(lsq_max, lsq);
          lsq_min = min(lsq_min, lsq);
        }
      }
    }
    double const h_min = std::sqrt(lsq_min);
    double const h_max = std::sqrt(lsq_max);
    double const alpha = std::sqrt(h_max / h_min);
    double const h_avg = h_min * alpha;
    nodes_to_h_adapt[node] = h_avg;
  };
  for_each(s.nodes, functor);
}

enum cavity_op {
  NONE,
  SWAP,
  SPLIT,
};

struct adapt_state {
  device_vector<double, node_index> criteria;
  device_vector<node_index, node_index> other_node;
  device_vector<bool, node_index> chosen;
  device_vector<cavity_op, node_index> op;
  device_vector<element_index, element_index> element_counts;
  device_vector<node_index, node_index> node_counts;
  device_vector<element_index, element_index> old_elements_to_new_elements;
  device_vector<node_index, node_index> old_nodes_to_new_nodes;
  device_vector<element_index, element_index> new_elements_to_old_elements;
  device_vector<node_index, node_index> new_nodes_to_old_nodes;
  device_vector<node_index, element_node_index> new_element_nodes_to_nodes;
  device_vector<bool, element_index> new_elements_are_same;
  device_vector<bool, node_index> new_nodes_are_same;
  device_vector<array<node_index, 2>, node_index> interpolate_from;
  counting_range<element_index> new_elements;
  counting_range<node_index> new_nodes;
  adapt_state(state const&);
};

adapt_state::adapt_state(state const& s)
  :criteria(s.nodes.size(), s.devpool)
  ,other_node(s.nodes.size(), s.devpool)
  ,chosen(s.nodes.size(), s.devpool)
  ,op(s.nodes.size(), s.devpool)
  ,element_counts(s.elements.size(), s.devpool)
  ,node_counts(s.nodes.size(), s.devpool)
  ,old_elements_to_new_elements(s.elements.size() + element_index(1), s.devpool)
  ,old_nodes_to_new_nodes(s.nodes.size() + node_index(1), s.devpool)
  ,new_elements_to_old_elements(s.devpool)
  ,new_nodes_to_old_nodes(s.devpool)
  ,new_element_nodes_to_nodes(s.devpool)
  ,new_elements_are_same(s.devpool)
  ,new_nodes_are_same(s.devpool)
  ,interpolate_from(s.devpool)
  ,new_elements(element_index(0))
  ,new_nodes(node_index(0))
{}

template <int Capacity, class Index>
static inline int find_or_append(
    int& count,
    array<Index, Capacity>& buffer,
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
struct cavity {
  int num_shell_elements;
  int num_shell_nodes;
  array<node_index, max_shell_nodes> shell_nodes;
  array<element_index, max_shell_elements> shell_elements;
  array<array<int, nodes_per_element>, max_shell_elements> shell_elements_to_shell_nodes;
  array<material_index, max_shell_elements> shell_elements_to_materials;
  array<double, max_shell_elements> shell_element_qualities;
  array<int, max_shell_elements> shell_elements_to_node_in_element;
  array<vector3<double>, max_shell_nodes> shell_nodes_to_x;
  array<double, max_shell_nodes> shell_nodes_to_h;
};

template <int max_shell_elements, int max_shell_nodes>
static inline void evaluate_triangle_swap(
    int const center_node,
    int const edge_node,
    cavity<3, max_shell_elements, max_shell_nodes> const c,
    double& best_improvement,
    int& best_swap_edge_node
    ) {
  array<int, 2> loop_elements;
  loop_elements[0] = loop_elements[1] = -1;
  array<int, 2> loop_nodes;
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
  double const old_quality1 = c.shell_element_qualities[loop_elements[0]];
  double const old_quality2 = c.shell_element_qualities[loop_elements[1]];
  double const quality_before = min(old_quality1, old_quality2);
  assert(quality_before > 0.0);
  array<vector3<double>, 3> proposed_x;
  proposed_x[0] = c.shell_nodes_to_x[center_node];
  proposed_x[1] = c.shell_nodes_to_x[loop_nodes[0]];
  proposed_x[2] = c.shell_nodes_to_x[loop_nodes[1]];
  double const new_quality1 = triangle_quality(proposed_x);
  if (new_quality1 <= quality_before) return;
  proposed_x[0] = c.shell_nodes_to_x[edge_node];
  proposed_x[1] = c.shell_nodes_to_x[loop_nodes[1]];
  proposed_x[2] = c.shell_nodes_to_x[loop_nodes[0]];
  double const new_quality2 = triangle_quality(proposed_x);
  if (new_quality2 <= quality_before) return;
  double const quality_after = min(new_quality1, new_quality2);
  double const improvement = ((quality_after - quality_before) / quality_before);
  if (improvement < 0.05) return;
  if (improvement > best_improvement) {
    best_improvement = improvement;
    best_swap_edge_node = edge_node;
  }
}

static inline double measure_edge(double const h_min, double const h_max, double const l) {
  return l / (0.5 * (h_min + h_max));
}

template <int max_shell_elements, int max_shell_nodes>
static inline void evaluate_triangle_split(
    int const center_node,
    int const edge_node,
    cavity<3, max_shell_elements, max_shell_nodes> const c,
    double& longest_length,
    int& best_split_edge_node
    ) {
  constexpr double min_acceptable_quality = 0.2;
  auto const h1 = c.shell_nodes_to_h[center_node];
  auto const h2 = c.shell_nodes_to_h[edge_node];
  auto const x1 = c.shell_nodes_to_x[center_node];
  auto const x2 = c.shell_nodes_to_x[edge_node];
  auto const h_min = min(h1, h2);
  auto const h_max = max(h1, h2);
  auto const l = norm(x1 - x2);
  auto const lm = measure_edge(h_min, h_max, l);
  if (lm < std::sqrt(2.0)) return;
  if (lm < longest_length) return;
  auto const midpoint_x = 0.5 * (x1 + x2);
  for (int element = 0; element < c.num_shell_elements; ++element) {
    array<vector3<double>, 3> parent_x;
    int const center_node_in_element = c.shell_elements_to_node_in_element[element];
    int edge_node_in_element = -1;
    for (int node_in_element = 0; node_in_element < 3; ++node_in_element) {
      int const shell_node = c.shell_elements_to_shell_nodes[element][node_in_element];
      parent_x[node_in_element] = c.shell_nodes_to_x[shell_node];
      if (shell_node == edge_node) edge_node_in_element = node_in_element;
    }
    if (edge_node_in_element == -1) continue;
    array<vector3<double>, 3> child_x = parent_x;
    child_x[center_node_in_element] = midpoint_x;
    double const new_quality1 = triangle_quality(child_x);
    if (new_quality1 < min_acceptable_quality) return;
    child_x[center_node_in_element] = c.shell_nodes_to_x[center_node];
    child_x[edge_node_in_element] = midpoint_x;
    double const new_quality2 = triangle_quality(child_x);
    if (new_quality2 < min_acceptable_quality) return;
  }
  longest_length = lm;
  best_split_edge_node = edge_node;
}

static LGR_NOINLINE void evaluate_triangle_adapt(state const& s, adapt_state& a)
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
  auto const nodes_to_criteria = a.criteria.begin();
  auto const nodes_to_other_nodes = a.other_node.begin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_to_op = a.op.begin();
  auto functor = [=] (node_index const node) {
    cavity<3, 32, 32> c;
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
          c.shell_nodes_to_x[shell_node] = nodes_to_x[node2];
          c.shell_nodes_to_h[shell_node] = nodes_to_h[node2];
        }
        c.shell_elements_to_shell_nodes[shell_element][int(node_in_element)] = shell_node;
      }
      material_index const material = elements_to_materials[element];
      c.shell_elements_to_materials[shell_element] = material;
      double const quality = elements_to_qualities[element];
      c.shell_element_qualities[shell_element] = quality;
      node_in_element_index const node_in_element = node_elements_to_node_in_element[node_element];
      c.shell_elements_to_node_in_element[shell_element] = int(node_in_element);
    }
    double best_swap_improvement = 0.0;
    int best_swap_edge_node = -1;
    double longest_split_edge = 0.0;
    int best_split_edge_node = -1;
    for (int edge_node = 0; edge_node < c.num_shell_nodes; ++edge_node) {
      if (edge_node == center_node) continue;
      // only examine edges once, the smaller node examines it
      if (c.shell_nodes[edge_node] < c.shell_nodes[center_node]) continue;
      evaluate_triangle_swap(center_node, edge_node, c,
          best_swap_improvement, best_swap_edge_node);
      evaluate_triangle_split(center_node, edge_node, c,
          longest_split_edge, best_split_edge_node);
    }
    if (best_swap_edge_node == -1) {
      nodes_to_other_nodes[node] = node_index(-1);
      nodes_to_op[node] = cavity_op::NONE;
    } else {
      nodes_to_criteria[node] = best_swap_improvement;
      nodes_to_other_nodes[node] = c.shell_nodes[best_swap_edge_node];
      nodes_to_op[node] = cavity_op::SWAP;
    }
  };
  for_each(s.nodes, functor);
}

static LGR_NOINLINE void choose_triangle_adapt(state const& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_criteria = a.criteria.cbegin();
  auto const nodes_to_other_nodes = a.other_node.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  fill(a.chosen, false);
  auto const nodes_are_chosen = a.chosen.begin();
  fill(a.element_counts, element_index(1));
  fill(a.node_counts, node_index(1));
  auto const elements_to_new_counts = a.element_counts.begin();
  auto const nodes_to_new_counts = a.node_counts.begin();
  auto const nodes_to_op = a.op.cbegin();
  auto functor = [=] (node_index const node) {
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
          if (op < adj_op) return;
          if (op > adj_op) continue;
          if (criteria < adj_criteria) return;
          if (criteria > adj_criteria) continue;
          if (adj_node < node) return;
        }
      }
    }
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node == target_node) {
          element_index count(-100);
          if (op == cavity_op::SWAP) {
            count = element_index(1);
          } else if (op == cavity_op::SPLIT) {
            count = element_index(2);
          }
          elements_to_new_counts[element] = count;
        }
      }
    }
    if (op == cavity_op::SPLIT) {
      nodes_to_new_counts[node] = node_index(2);
    }
    nodes_are_chosen[node] = true;
  };
  for_each(s.nodes, functor);
}

static LGR_NOINLINE void apply_triangle_adapt(state const& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const old_element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_other_nodes = a.other_node.cbegin();
  auto const nodes_are_chosen = a.chosen.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const old_elements_to_new_elements = a.old_elements_to_new_elements.cbegin();
  auto const new_elements_to_element_nodes = a.new_elements * s.nodes_in_element;
  auto const old_nodes_to_new_nodes = a.old_nodes_to_new_nodes.cbegin();
  auto const new_element_nodes_to_nodes = a.new_element_nodes_to_nodes.begin();
  fill(a.new_elements_are_same, true);
  fill(a.new_nodes_are_same, true);
  auto const new_elements_are_same = a.new_elements_are_same.begin();
  auto functor = [=] (node_index const node) {
    if (!nodes_are_chosen[node]) return;
    node_index const target_node = nodes_to_other_nodes[node];
    array<element_index, 2> loop_elements;
    array<node_index, 2> loop_nodes;
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      node_in_element_index const node_in_element = node_elements_to_nodes_in_element[node_element];
      node_in_element_index const plus1 = node_in_element_index((int(node_in_element) + 1) % 3);
      node_in_element_index const plus2 = node_in_element_index((int(node_in_element) + 2) % 3);
      node_index const node1 = old_element_nodes_to_nodes[element_nodes[plus1]];
      node_index const node2 = old_element_nodes_to_nodes[element_nodes[plus2]];
      if (node1 == target_node) {
        loop_elements[1] = element;
        loop_nodes[1] = node2;
      }
      if (node2 == target_node) {
        loop_elements[0] = element;
        loop_nodes[0] = node1;
      }
    }
    element_index const new_element1 = old_elements_to_new_elements[loop_elements[0]];
    element_index const new_element2 = old_elements_to_new_elements[loop_elements[1]];
    using l_t = node_in_element_index;
    auto new_element_nodes = new_elements_to_element_nodes[new_element1];
    new_element_nodes_to_nodes[new_element_nodes[l_t(0)]] = old_nodes_to_new_nodes[node];
    new_element_nodes_to_nodes[new_element_nodes[l_t(1)]] = old_nodes_to_new_nodes[loop_nodes[0]];
    new_element_nodes_to_nodes[new_element_nodes[l_t(2)]] = old_nodes_to_new_nodes[loop_nodes[1]];
    new_element_nodes = new_elements_to_element_nodes[new_element2];
    new_element_nodes_to_nodes[new_element_nodes[l_t(0)]] = old_nodes_to_new_nodes[target_node];
    new_element_nodes_to_nodes[new_element_nodes[l_t(1)]] = old_nodes_to_new_nodes[loop_nodes[1]];
    new_element_nodes_to_nodes[new_element_nodes[l_t(2)]] = old_nodes_to_new_nodes[loop_nodes[0]];
    new_elements_are_same[new_element1] = false;
    new_elements_are_same[new_element2] = false;
  };
  for_each(s.nodes, functor);
}

template <class Index>
static LGR_NOINLINE void project(
    counting_range<Index> const old_things,
    device_vector<Index, Index> const& old_things_to_new_things_in,
    device_vector<Index, Index>& new_things_to_old_things_in) {
  auto const old_things_to_new_things = old_things_to_new_things_in.cbegin();
  auto const new_things_to_old_things = new_things_to_old_things_in.begin();
  auto functor = [=] (Index const old_thing) {
    Index first = old_things_to_new_things[old_thing];
    Index const last = old_things_to_new_things[old_thing + Index(1)];
    for (; first < last; ++first) {
      new_things_to_old_things[first] = old_thing;
    }
  };
  for_each(old_things, functor);
}

static LGR_NOINLINE void transfer_same_element_node(state const& s, adapt_state& a,
    device_vector<node_index, element_node_index> const& old_data,
    device_vector<node_index, element_node_index>& new_data) {
  auto const new_elements_to_element_nodes = a.new_elements * s.nodes_in_element;
  auto const old_elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const old_element_nodes_to_T = old_data.cbegin();
  auto const new_element_nodes_to_T = new_data.begin();
  auto const new_elements_are_same = a.new_elements_are_same.cbegin();
  auto functor = [=] (element_index const new_element) {
    if (new_elements_are_same[new_element]) {
      auto const new_element_nodes = new_elements_to_element_nodes[new_element];
      element_index const old_element = new_elements_to_old_elements[new_element];
      auto const old_element_nodes = old_elements_to_element_nodes[old_element];
      for (auto const node_in_element : nodes_in_element) {
        auto const new_element_node = new_element_nodes[node_in_element];
        auto const old_element_node = old_element_nodes[node_in_element];
        node_index const node = node_index(old_element_nodes_to_T[old_element_node]);
        new_element_nodes_to_T[new_element_node] = node;
      }
    }
  };
  for_each(a.new_elements, functor);
}

static LGR_NOINLINE void transfer_same_connectivity(state const& s, adapt_state& a) {
  transfer_same_element_node(s, a, s.elements_to_nodes, a.new_element_nodes_to_nodes);
}

template <class T>
static LGR_NOINLINE void transfer_element_data(adapt_state& a,
    device_vector<T, element_index>& data) {
  device_vector<T, element_index> new_data(a.new_elements.size(), data.get_allocator());
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const old_elements_to_T = data.cbegin();
  auto const new_elements_to_T = new_data.begin();
  auto functor = [=] (element_index const new_element) {
    element_index const old_element = new_elements_to_old_elements[new_element];
    new_elements_to_T[new_element] =
      T(old_elements_to_T[old_element]);
  };
  for_each(a.new_elements, functor);
  data = std::move(new_data);
}

template <class T>
static LGR_NOINLINE void transfer_point_data(state const& s, adapt_state const& a,
    device_vector<T, point_index>& data) {
  auto const points_in_element = s.points_in_element;
  device_vector<T, point_index> new_data(a.new_elements.size() * points_in_element.size(), data.get_allocator());
  auto const new_elements_to_old_elements = a.new_elements_to_old_elements.cbegin();
  auto const old_points_to_T = data.cbegin();
  auto const new_points_to_T = new_data.begin();
  auto const old_elements_to_points = s.elements * points_in_element;
  auto const new_elements_to_points = a.new_elements * points_in_element;
  auto functor = [=] (element_index const new_element) {
    element_index const old_element = new_elements_to_old_elements[new_element];
    auto const new_element_points = new_elements_to_points[new_element];
    auto const old_element_points = old_elements_to_points[old_element];
    for (auto const point_in_element : points_in_element) {
      auto const new_point = new_element_points[point_in_element];
      auto const old_point = old_element_points[point_in_element];
      new_points_to_T[new_point] =
        T(old_points_to_T[old_point]);
    }
  };
  for_each(a.new_elements, functor);
  data = std::move(new_data);
}

template <class T>
static LGR_NOINLINE void interpolate_nodal_data(adapt_state const& a, device_vector<T, node_index>& data) {
  device_vector<T, node_index> new_data(a.new_nodes.size(), data.get_allocator());
  auto const old_nodes_to_data = data.cbegin();
  auto const new_nodes_to_data = new_data.begin();
  auto const new_nodes_to_old_nodes = a.new_nodes_to_old_nodes.cbegin();
  auto const new_nodes_are_same = a.new_nodes_are_same.cbegin();
  auto const interpolate_from = a.interpolate_from.cbegin();
  auto functor = [=] (node_index const new_node) {
    if (new_nodes_are_same[new_node]) {
      node_index const old_node = new_nodes_to_old_nodes[new_node];
      new_nodes_to_data[new_node] = T(old_nodes_to_data[old_node]);
    } else {
      array<node_index, 2> const pair = interpolate_from[new_node];
      node_index const left = pair[0];
      node_index const right = pair[1];
      new_nodes_to_data[new_node] = 0.5 * (
          T(old_nodes_to_data[left])
        + T(old_nodes_to_data[right]));
    }
  };
  for_each(a.new_nodes, functor);
  data = std::move(new_data);
}

bool adapt(input const& in, state& s) {
  adapt_state a(s);
  evaluate_triangle_adapt(s, a);
  choose_triangle_adapt(s, a);
  auto const num_chosen = reduce(a.chosen, int(0));
  if (num_chosen == 0) return false;
  if (in.output_to_command_line) {
    std::cout << "adapting " << num_chosen << " cavities\n";
  }
  auto const num_new_elements = reduce(a.element_counts, element_index(0));
  auto const num_new_nodes = reduce(a.node_counts, node_index(0));
  offset_scan(a.element_counts, a.old_elements_to_new_elements);
  offset_scan(a.node_counts, a.old_nodes_to_new_nodes);
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
  transfer_element_data<material_index>(a, s.material);
  transfer_point_data<double>(s, a, s.rho);
  if (!in.enable_nodal_energy) transfer_point_data<double>(s, a, s.e);
  transfer_point_data<matrix3x3<double>>(s, a, s.F_total);
  interpolate_nodal_data<vector3<double>>(a, s.x);
  interpolate_nodal_data<vector3<double>>(a, s.v);
  s.elements = a.new_elements;
  s.nodes = a.new_nodes;
  s.elements_to_nodes = std::move(a.new_element_nodes_to_nodes);
  invert_connectivity(s);
  return true;
}

}
