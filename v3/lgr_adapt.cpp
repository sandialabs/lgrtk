#include <lgr_macros.hpp>
#include <lgr_state.hpp>
#include <lgr_input.hpp>
#include <lgr_adapt.hpp>
#include <lgr_array.hpp>
#include <lgr_element_specific_inline.hpp>
#include <lgr_print.hpp>
#include <lgr_reduce.hpp>

#include <iostream>
#include <iomanip>

namespace lgr {

static void LGR_NOINLINE update_bar_Q(state& s) {
  fill(s.Q, double(1.0));
}

inline double triangle_quality(array<vector3<double>, 3> const grad_N, double const area) {
  double sum_g_i_sq = 0.0;
  for (int i = 0; i < 3; ++i) {
    auto const g_i_sq = (grad_N[i] * grad_N[i]);
    sum_g_i_sq += g_i_sq;
  }
  return (area * sum_g_i_sq);
}

inline double triangle_quality(array<vector3<double>, 3> const x) {
  double const area = triangle_area(x);
  return triangle_quality(triangle_basis_gradients(x, area), area);
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
static void LGR_NOINLINE update_triangle_Q(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_Q = s.Q.begin();
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
    double const fast_Q = triangle_quality(grad_N, A);
    elements_to_Q[element] = fast_Q;
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
  */
static void LGR_NOINLINE update_tetrahedron_Q(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_Q = s.Q.begin();
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
    elements_to_Q[element] = (V * V) * (sum_g_i_sq * sum_g_i_sq * sum_g_i_sq);
  };
  lgr::for_each(s.elements, functor);
}

void update_Q(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_Q(s); break;
    case TRIANGLE: update_triangle_Q(s); break;
    case TETRAHEDRON: update_tetrahedron_Q(s); break;
    case COMPOSITE_TETRAHEDRON: assert(0); break;
  }
}

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

struct adapt_state {
  device_vector<double, node_index> improvement;
  device_vector<node_index, node_index> other_node;
  device_vector<int, node_index> chosen;
  device_vector<element_index, element_index> element_counts;
  device_vector<element_index, element_index> old_elements_to_new_elements;
  adapt_state(state&);
};

adapt_state::adapt_state(state& s)
  :improvement(s.nodes.size(), s.devpool)
  ,other_node(s.nodes.size(), s.devpool)
  ,chosen(s.nodes.size(), s.devpool)
  ,element_counts(s.elements.size(), s.devpool)
  ,old_elements_to_new_elements(s.elements.size() + element_index(1), s.devpool)
{}

static LGR_NOINLINE void evaluate_triangle_adapt(state& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_node_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const elements_to_materials = s.material.cbegin();
  auto const elements_to_badnesses = s.Q.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_improvement = a.improvement.begin();
  auto const nodes_to_other_nodes = a.other_node.begin();
  auto functor = [=] (node_index const node) {
    int num_shell_nodes = 0;
    int num_shell_elements = 0;
    constexpr int max_shell_elements = 32;
    constexpr int max_shell_nodes = 32;
    constexpr int nodes_per_element = 3;
    array<node_index, max_shell_nodes> shell_nodes;
    array<element_index, max_shell_elements> shell_elements;
    array<array<int, nodes_per_element>, max_shell_elements> shell_elements_to_shell_nodes;
    array<material_index, max_shell_elements> shell_elements_to_materials;
    array<double, max_shell_elements> shell_element_badnesses;
    array<int, max_shell_elements> shell_elements_to_node_in_element;
    array<vector3<double>, max_shell_nodes> shell_nodes_to_x;
    int center_node = -1;
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element]; 
      int const shell_element = num_shell_elements++;
      shell_elements[shell_element] = element;
      auto const element_nodes = elements_to_element_nodes[element];
      for (node_in_element_index node_in_element(0);
          node_in_element < node_in_element_index(nodes_per_element);
          ++node_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const node2 = element_nodes_to_nodes[element_node];
        int const shell_node = find_or_append(num_shell_nodes, shell_nodes, node2);
        if (node2 == node) center_node = shell_node;
        if (shell_node + 1 == num_shell_nodes) {
          shell_nodes_to_x[shell_node] = nodes_to_x[node2];
        }
        shell_elements_to_shell_nodes[shell_element][int(node_in_element)] = shell_node;
      }
      material_index const material = elements_to_materials[element];
      shell_elements_to_materials[shell_element] = material;
      double const badness = elements_to_badnesses[element];
      shell_element_badnesses[shell_element] = badness;
      node_in_element_index const node_in_element = node_elements_to_node_in_element[node_element];
      shell_elements_to_node_in_element[shell_element] = int(node_in_element);
    }
    double best_improvement = 0.0;
    int best_swap_edge_node = -1;
    for (int edge_node = 0; edge_node < num_shell_nodes; ++edge_node) {
      if (edge_node == center_node) continue;
      // only examine edges once, the smaller node examines it
      if (shell_nodes[edge_node] < shell_nodes[center_node]) continue;
      array<int, 2> loop_elements;
      loop_elements[0] = loop_elements[1] = -1;
      array<int, 2> loop_nodes;
      loop_nodes[0] = loop_nodes[1] = -1;
      for (int element = 0; element < num_shell_elements; ++element) {
        int const node_in_element = shell_elements_to_node_in_element[element];
        if (shell_elements_to_shell_nodes[element][(node_in_element + 1) % 3] == edge_node) {
          loop_elements[1] = element;
          loop_nodes[1] = shell_elements_to_shell_nodes[element][(node_in_element + 2) % 3];
        }
        if (shell_elements_to_shell_nodes[element][(node_in_element + 2) % 3] == edge_node) {
          loop_elements[0] = element;
          loop_nodes[0] = shell_elements_to_shell_nodes[element][(node_in_element + 1) % 3];
        }
      }
      if (loop_elements[0] == -1 || loop_elements[1] == -1) {
        continue;
      }
      if (shell_elements_to_materials[loop_elements[0]] != shell_elements_to_materials[loop_elements[1]]) {
        continue;
      }
      double const old_badness1 = shell_element_badnesses[loop_elements[0]];
      double const old_badness2 = shell_element_badnesses[loop_elements[1]];
      double const badness_before = lgr::max(old_badness1, old_badness2);
      array<vector3<double>, 3> proposed_x;
      proposed_x[0] = shell_nodes_to_x[center_node];
      proposed_x[1] = shell_nodes_to_x[loop_nodes[0]];
      proposed_x[2] = shell_nodes_to_x[loop_nodes[1]];
      double const new_badness1 = triangle_quality(proposed_x);
      proposed_x[0] = shell_nodes_to_x[edge_node];
      proposed_x[1] = shell_nodes_to_x[loop_nodes[1]];
      proposed_x[2] = shell_nodes_to_x[loop_nodes[0]];
      double const new_badness2 = triangle_quality(proposed_x);
      double const badness_after = lgr::max(new_badness1, new_badness2);
      if (badness_after > badness_before) continue;
      double const improvement = ((badness_before - badness_after) / badness_after);
      if (improvement < 0.05) continue;
      if (improvement > best_improvement) {
        best_improvement = improvement;
        best_swap_edge_node = edge_node;
      }
    }
    nodes_to_improvement[node] = best_improvement;
    nodes_to_other_nodes[node] = (best_swap_edge_node == -1) ? node_index(-1) : shell_nodes[best_swap_edge_node];
  };
  for_each(s.nodes, functor);
}

static LGR_NOINLINE void choose_triangle_adapt(state& s, adapt_state& a)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_improvement = a.improvement.cbegin();
  auto const nodes_to_other_nodes = a.other_node.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_are_chosen = a.chosen.begin();
  fill(a.element_counts, element_index(1));
  auto const elements_to_new_counts = a.element_counts.begin();
  auto functor = [=] (node_index const node) {
    node_index const target_node = nodes_to_other_nodes[node];
    if (target_node == node_index(-1)) return;
    double const improvement = nodes_to_improvement[node];
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element]; 
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node != node) {
          double const adj_improvement = nodes_to_improvement[adj_node];
          if ((adj_improvement > improvement) ||
              ((adj_improvement == improvement) &&
               (adj_node < node))) {
            nodes_are_chosen[node] = 0;
            return;
          }
        }
      }
    }
    bool is_first = true;
    for (auto const node_element : nodes_to_node_elements[node]) {
      element_index const element = node_elements_to_elements[node_element]; 
      auto const element_nodes = elements_to_element_nodes[element];
      for (auto const node_in_element : nodes_in_element) {
        element_node_index const element_node = element_nodes[node_in_element];
        node_index const adj_node = element_nodes_to_nodes[element_node];
        if (adj_node == target_node) {
          elements_to_new_counts[element] = is_first ? element_index(2) : element_index(0);
          is_first = false;
        }
      }
    }
    nodes_are_chosen[node] = 1;
  };
  for_each(s.nodes, functor);
}

void adapt(state& s) {
  adapt_state a(s);
  evaluate_triangle_adapt(s, a);
  choose_triangle_adapt(s, a);
  element_index const num_new_elements = transform_reduce(a.element_counts, element_index(0),
      plus<element_index>(), identity<element_index>());
  std::cout << int(num_new_elements) << " new elements\n";
}

}
