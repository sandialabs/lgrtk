#include <iostream>

#include <lgr_state.hpp>
#include <lgr_run.hpp>
#include <lgr_int_range.hpp>
#include <lgr_host_vector.hpp>
#include <lgr_int_range_sum.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>
#include <lgr_print.hpp>
#include <lgr_vtk.hpp>
#include <lgr_for_each.hpp>
#include <lgr_int_range_product.hpp>
#include <lgr_reduce.hpp>
#include <lgr_fill.hpp>
#include <lgr_copy.hpp>
#include <lgr_element_specific.hpp>
#include <lgr_meshing.hpp>

namespace lgr {

static void LGR_NOINLINE advance_time(
    input const& in,
    double const max_stable_dt,
    double const next_file_output_time,
    double* time,
    double* dt) {
  auto const old_time = *time;
  auto new_time = next_file_output_time;
  new_time = std::min(new_time, old_time + (max_stable_dt * in.CFL));
  *time = new_time;
  *dt = new_time - old_time;
}

static void LGR_NOINLINE update_u(
    int_range const nodes,
    double const dt,
    host_vector<vector3<double>> const& v_vector,
    host_vector<vector3<double>>* u_vector) {
  auto const u_iterator = u_vector->begin();
  auto const v_iterator = v_vector.cbegin();
  auto functor = [=] (int const node) {
    vector3<double> const old_u = u_iterator[node];
    vector3<double> const v = v_iterator[node];
    u_iterator[node] = (dt * v) - old_u;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_V_h_incr(
    int_range const nodes,
    double const dt,
    host_vector<double> const& V_h_dot_vector,
    host_vector<double>* V_h_incr_vector) {
  auto const nodes_to_V_h_incr = V_h_incr_vector->begin();
  auto const nodes_to_V_h_dot = V_h_dot_vector.cbegin();
  auto functor = [=] (int const node) {
    double const old_V_h_incr = nodes_to_V_h_incr[node];
    double const V_h_dot = nodes_to_V_h_dot[node];
    nodes_to_V_h_incr[node] = (dt * V_h_dot) - old_V_h_incr;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_v(state& s, double const dt, host_vector<vector3<double>> const& old_v_vector) {
  auto const nodes_to_v = s.v.begin();
  auto const nodes_to_old_v = old_v_vector.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto functor = [=] (int const node) {
    vector3<double> const old_v = nodes_to_old_v[node];
    vector3<double> const a = nodes_to_a[node];
    vector3<double> const v = old_v + dt * a;
    nodes_to_v[node] = v;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_p_h(
    int_range const nodes,
    double const dt,
    host_vector<double> const& p_h_dot_vector,
    host_vector<double> const& old_p_h_vector,
    host_vector<double>* p_h_vector) {
  auto const nodes_to_p_h = p_h_vector->begin();
  auto const nodes_to_old_p_h = old_p_h_vector.cbegin();
  auto const nodes_to_p_h_dot = p_h_dot_vector.cbegin();
  auto functor = [=] (int const node) {
    double const old_p_h = nodes_to_old_p_h[node];
    double const p_h_dot = nodes_to_p_h_dot[node];
    double const p_h = old_p_h + dt * p_h_dot;
    nodes_to_p_h[node] = p_h;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_sigma_with_p_h(
    int_range const elements,
    int_range const nodes_in_element,
    host_vector<int> const& elements_to_nodes_vector,
    host_vector<double> const& p_h_vector,
    host_vector<symmetric3x3<double>>* sigma_vector) {
  auto const elements_to_element_nodes = elements * nodes_in_element;
  auto const element_nodes_to_nodes = elements_to_nodes_vector.cbegin();
  auto const nodes_to_p_h = p_h_vector.cbegin();
  auto const elements_to_sigma = sigma_vector->begin();
  auto functor = [=] (int const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    double element_p_h = 0.0;
    for (auto const element_node : element_nodes) {
      auto const node = element_nodes_to_nodes[element_node];
      double const p_h = nodes_to_p_h[node];
      element_p_h = element_p_h + p_h;
    }
    element_p_h = element_p_h / double(element_nodes.size());
    symmetric3x3<double> const old_sigma = elements_to_sigma[element];
    auto const new_sigma = deviator(old_sigma) - element_p_h;
    elements_to_sigma[element] = new_sigma;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE update_a(
    int_range const nodes,
    host_vector<vector3<double>> const& f_vector,
    host_vector<double> const& m_vector,
    host_vector<vector3<double>>* a_vector) {
  auto const nodes_to_f = f_vector.cbegin();
  auto const nodes_to_m = m_vector.cbegin();
  auto const nodes_to_a = a_vector->begin();
  auto functor = [=] (int const node) {
    vector3<double> const f = nodes_to_f[node];
    double const m = nodes_to_m[node];
    vector3<double> const a = f / m;
    nodes_to_a[node] = a;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_x(
    int_range const nodes,
    host_vector<vector3<double>> const& u_vector,
    decltype(state::x)* x_vector) {
  auto const nodes_to_u = u_vector.cbegin();
  auto const nodes_to_x = x_vector->begin();
  auto functor = [=] (int const node) {
    vector3<double> const old_x = nodes_to_x[node];
    vector3<double> const u = nodes_to_u[node];
    vector3<double> const new_x = old_x + u;
    nodes_to_x[node] = new_x;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_p(
    int_range const elements,
    host_vector<symmetric3x3<double>> const& sigma_vector,
    host_vector<double>* p_vector) {
  auto const elements_to_sigma = sigma_vector.cbegin();
  auto const elements_to_p = p_vector->begin();
  auto functor = [=] (int const element) {
    symmetric3x3<double> const sigma = elements_to_sigma[element];
    auto const p = -(1.0 / 3.0) * trace(sigma);
    elements_to_p[element] = p;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE update_reference(
    int_range const elements,
    int_range const nodes_in_element,
    host_vector<int> const& elements_to_nodes_vector,
    host_vector<vector3<double>> const& u_vector,
    host_vector<matrix3x3<double>>* F_total_vector,
    host_vector<vector3<double>>* grad_N_vector,
    host_vector<double>* V_vector,
    host_vector<double>* rho_vector) {
  auto const elements_to_element_nodes = elements * nodes_in_element;
  auto const element_nodes_to_nodes = elements_to_nodes_vector.cbegin();
  auto const nodes_to_u = u_vector.cbegin();
  auto const elements_to_F_total = F_total_vector->begin();
  auto const element_nodes_to_grad_N = grad_N_vector->begin();
  auto const elements_to_V = V_vector->begin();
  auto const elements_to_rho = rho_vector->begin();
  auto functor = [=] (int const element) {
    auto F_incr = matrix3x3<double>::identity();
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      auto const node = element_nodes_to_nodes[element_node];
      vector3<double> const u = nodes_to_u[node];
      vector3<double> const old_grad_N = element_nodes_to_grad_N[element_node];
      F_incr = F_incr + outer_product(u, old_grad_N);
    }
    auto const F_inverse_transpose = transpose(inverse(F_incr));
    for (auto const element_node : element_nodes) {
      vector3<double> const old_grad_N = element_nodes_to_grad_N[element_node];
      auto const new_grad_N = F_inverse_transpose * old_grad_N;
      element_nodes_to_grad_N[element_node] = new_grad_N;
    }
    matrix3x3<double> const old_F_total = elements_to_F_total[element];
    matrix3x3<double> const new_F_total = F_incr * old_F_total;
    elements_to_F_total[element] = new_F_total;
    auto const J = determinant(F_incr);
    assert(J > 0.0);
    double const old_V = elements_to_V[element];
    auto const new_V = J * old_V;
    assert(new_V > 0.0);
    elements_to_V[element] = new_V;
    auto const old_rho = elements_to_rho[element];
    auto const new_rho = old_rho / J;
    elements_to_rho[element] = new_rho;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE update_nodal_reference(
    int_range const nodes,
    host_vector<double> const& V_h_incr_vector,
    host_vector<double>* V_h_vector,
    host_vector<double>* J_h_vector) {
  auto const nodes_to_V_h_incr = V_h_incr_vector.cbegin();
  auto const nodes_to_V_h = V_h_vector->begin();
  auto const nodes_to_J_h = J_h_vector->begin();
  auto functor = [=] (int const node) {
    double const old_V_h = nodes_to_V_h[node];
    double const V_h_incr = nodes_to_V_h_incr[node];
    auto const V_h = old_V_h + V_h_incr;
    assert(V_h > 0.0);
    auto const J_incr = V_h / old_V_h;
    assert(J_incr > 0.0);
    nodes_to_V_h[node] = V_h;
    auto const old_J_h = nodes_to_J_h[node];
    auto const J_h = J_incr * old_J_h;
    nodes_to_J_h[node] = J_h;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE update_c(state& s)
{
  auto const elements_to_rho = s.rho.cbegin();
  auto const elements_to_K = s.K.cbegin();
  auto const elements_to_G = s.G.cbegin();
  auto const elements_to_c = s.c.begin();
  auto functor = [=] (int const element) {
    double const rho = elements_to_rho[element];
    double const K = elements_to_K[element];
    double const G = elements_to_G[element];
    auto const M = K + (4.0 / 3.0) * G;
    auto const c = std::sqrt(M / rho);
    elements_to_c[element] = c;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_element_dt(state& s) {
  auto const elements_to_c = s.c.cbegin();
  auto const elements_to_h_min = s.h_min.cbegin();
  auto const elements_to_dt = s.element_dt.begin();
  auto functor = [=] (int const element) {
    double const h_min = elements_to_h_min[element];
    auto const c = elements_to_c[element];
    auto const element_dt = h_min / c;
    assert(element_dt > 0.0);
    elements_to_dt[element] = element_dt;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE find_max_stable_dt(
    host_vector<double> const& element_dt_vector,
    double* max_stable_dt
    )
{
  double const init = std::numeric_limits<double>::max();
  *max_stable_dt = lgr::reduce(
      element_dt_vector.begin(), element_dt_vector.end(), init, lgr::minimum<double>());
  //print(std::cerr, "max_stable_dt ", *max_stable_dt, "\n");
}

static void LGR_NOINLINE update_v_prime(input const& in, state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_dt = s.element_dt.cbegin();
  auto const elements_to_rho = s.rho.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto const nodes_to_p_h = s.p_h.cbegin();
  auto const elements_to_v_prime = s.v_prime.begin();
  auto const c_tau = in.c_tau;
  auto const inv_nodes_per_element = 1.0 / double(s.nodes_in_element.size());
  auto functor = [=] (int const element) {
    double const dt = elements_to_dt[element];
    auto const tau = c_tau * dt;
    auto grad_p = vector3<double>::zero();
    auto const element_nodes = elements_to_element_nodes[element];
    auto a = vector3<double>::zero();
    for (auto const element_node : element_nodes) {
      int const node = element_nodes_to_nodes[element_node];
      double const p_h = nodes_to_p_h[node];
      vector3<double> const grad_N = element_nodes_to_grad_N[element_node];
      grad_p = grad_p + (grad_N * p_h);
      vector3<double> const a_of_node = nodes_to_a[node];
      a = a + a_of_node;
    }
    a = a * inv_nodes_per_element;
    double const rho = elements_to_rho[element];
    auto const v_prime = -(tau / rho) * (rho * a + grad_p);
    elements_to_v_prime[element] = v_prime;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_p_h_W(state& s)
{
  auto const elements_to_K = s.K.cbegin();
  auto const elements_to_v_prime = s.v_prime.cbegin();
  auto const elements_to_V = s.V.cbegin();
  auto const elements_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const element_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(s.nodes_in_element.size());
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto functor = [=] (int const element) {
    symmetric3x3<double> symm_grad_v = elements_to_symm_grad_v[element];
    double const div_v = trace(symm_grad_v);
    double const K = elements_to_K[element];
    double const V = elements_to_V[element];
    vector3<double> const v_prime = elements_to_v_prime[element];
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      vector3<double> const grad_N = element_nodes_to_grad_N[element_node];
      double const p_h_dot =
        -(N * (K * div_v)) + (grad_N * (K * v_prime));
      double const W = p_h_dot * V;
      element_nodes_to_W[element_node] = W;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_V_h_W(state& s)
{
  auto const elements_to_v_prime = s.v_prime.cbegin();
  auto const elements_to_V = s.V.cbegin();
  auto const elements_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const element_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(s.nodes_in_element.size());
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto functor = [=] (int const element) {
    symmetric3x3<double> symm_grad_v = elements_to_symm_grad_v[element];
    double const div_v = trace(symm_grad_v);
    double const V = elements_to_V[element];
    vector3<double> const v_prime = elements_to_v_prime[element];
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      vector3<double> const grad_N = element_nodes_to_grad_N[element_node];
      double const V_h_dot = (N * div_v) - (grad_N * v_prime);
      double const W = V_h_dot * V;
      element_nodes_to_W[element_node] = W;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_any_h_dot(
    int_range const nodes,
    int_range const elements,
    int_range const nodes_in_element,
    int_range_sum<host_allocator<int>> const& nodes_to_node_elements_vector,
    host_vector<int> const& node_elements_to_elements_vector,
    host_vector<int> const& node_elements_to_nodes_in_element_vector,
    host_vector<double> const& W_vector,
    host_vector<double> const& V_vector,
    host_vector<double>* any_h_dot_vector)
{
  auto const nodes_to_node_elements = nodes_to_node_elements_vector.cbegin();
  auto const node_elements_to_elements = node_elements_to_elements_vector.cbegin();
  auto const node_elements_to_nodes_in_element = node_elements_to_nodes_in_element_vector.cbegin();
  auto const element_nodes_to_W = W_vector.cbegin();
  auto const elements_to_V = V_vector.cbegin();
  auto const nodes_to_any_h_dot = any_h_dot_vector->begin();
  auto const elements_to_element_nodes = elements * nodes_in_element;
  auto functor = [=] (int const node) {
    double node_W = 0.0;
    double node_V = 0.0;
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      auto const element_node = element_nodes[node_in_element];
      double const W = element_nodes_to_W[element_node];
      double const V = elements_to_V[element];
      node_W = node_W + W;
      node_V = node_V + V;
    }
    auto const any_h_dot = node_W / node_V;
    nodes_to_any_h_dot[node] = any_h_dot;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE neo_Hookean(
    int_range const elements,
    double const K0,
    double const G0,
    host_vector<matrix3x3<double>> const& F_total_vector,
    host_vector<symmetric3x3<double>>* sigma_vector,
    host_vector<double>* K_vector,
    host_vector<double>* G_vector
    ) {
  auto const elements_to_F_total = F_total_vector.cbegin();
  auto const elements_to_sigma = sigma_vector->begin();
  auto const elements_to_K = K_vector->begin();
  auto const elements_to_G = G_vector->begin();
  auto functor = [=] (int const element) {
    matrix3x3<double> const F = elements_to_F_total[element];
    auto const J = determinant(F);
    auto const Jinv = 1.0 / J;
    auto const half_K0 = 0.5 * K0;
    auto const Jm13 = 1.0 / std::cbrt(J);
    auto const Jm23 = Jm13 * Jm13;
    auto const Jm53 = (Jm23 * Jm23) * Jm13;
    auto const B = self_times_transpose(F);
    auto const devB = deviator(B);
    //print(std::cerr, "p[", element, "] = ", (half_K0 * (J - Jinv)), "\n");
    auto const sigma = half_K0 * (J - Jinv) + (G0 * Jm53) * devB;
    elements_to_sigma[element] = sigma;
    auto const K = half_K0 * (J + Jinv);
    elements_to_K[element] = K;
    elements_to_G[element] = G0;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE nodal_neo_Hookean(
    int_range const nodes,
    double const K0,
    host_vector<double> const& J_h_vector,
    host_vector<double>* p_h_vector
    ) {
  auto const nodes_to_J_h = J_h_vector.cbegin();
  auto const nodes_to_p_h = p_h_vector->begin();
  auto functor = [=] (int const node) {
    auto const J = nodes_to_J_h[node];
    auto const Jinv = 1.0 / J;
    auto const half_K0 = 0.5 * K0;
    auto const p = -half_K0 * (J - Jinv);
    //print(std::cerr, "p_h[", node, "] = ", p, "\n");
    nodes_to_p_h[node] = p;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE ideal_gas(
    int_range const elements,
    double const gamma,
    host_vector<double> const& rho_vector,
    host_vector<double> const& e_vector,
    host_vector<symmetric3x3<double>>* sigma_vector,
    host_vector<double>* K_vector
    ) {
  auto const elements_to_rho = rho_vector.cbegin();
  auto const elements_to_e = e_vector.cbegin();
  auto const elements_to_sigma = sigma_vector->begin();
  auto const elements_to_K = K_vector->begin();
  auto functor = [=] (int const element) {
    double const rho = elements_to_rho[element];
    assert(rho > 0.0);
    double const e = elements_to_e[element];
    assert(e > 0.0);
    auto const p = (gamma - 1.0) * (rho * e);
    assert(p > 0.0);
    symmetric3x3<double> const old_sigma = elements_to_sigma[element];
    auto const new_sigma = deviator(old_sigma) - p;
    elements_to_sigma[element] = new_sigma;
    auto const K = gamma * p;
    assert(K > 0.0);
    elements_to_K[element] = K;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE update_element_force(
    int_range const elements,
    int_range const nodes_in_element,
    host_vector<symmetric3x3<double>> const& sigma_vector,
    host_vector<double> const& V_vector,
    host_vector<vector3<double>> const& grad_N_vector,
    host_vector<vector3<double>>* element_f_vector
    )
{
  auto const sigma_iterator = sigma_vector.cbegin();
  auto const V_iterator = V_vector.cbegin();
  auto const element_nodes_to_grad_N = grad_N_vector.cbegin();
  auto const element_nodes_to_f = element_f_vector->begin();
  auto const elements_to_element_nodes = elements * nodes_in_element;
  auto functor = [=] (int const element) {
    symmetric3x3<double> const sigma = sigma_iterator[element];
    double const V = V_iterator[element];
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      vector3<double> const grad_N = element_nodes_to_grad_N[element_node];
      auto const element_f = -(sigma * grad_N) * V;
      element_nodes_to_f[element_node] = element_f;
    }
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE update_nodal_force(state& s) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const element_nodes_to_f = s.element_f.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto functor = [=] (int const node) {
    auto node_f = vector3<double>::zero();
    auto const range = nodes_to_node_elements[node];
    for (auto const node_element : range) {
      auto const element = node_elements_to_elements[node_element];
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      auto const element_node = element_nodes[node_in_element];
      vector3<double> const element_f = element_nodes_to_f[element_node];
      node_f = node_f + element_f;
    }
    nodes_to_f[node] = node_f;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_nodal_mass(
    int_range const nodes,
    int_range const nodes_in_element,
    int_range_sum<host_allocator<int>> const& nodes_to_node_elements_vector,
    host_vector<int> const& node_elements_to_elements_vector,
    host_vector<double> const& rho_vector,
    host_vector<double> const& V_vector,
    host_vector<double>* m_vector) {
  auto const nodes_to_node_elements = nodes_to_node_elements_vector.cbegin();
  auto const node_elements_to_elements = node_elements_to_elements_vector.cbegin();
  auto const elements_to_rho = rho_vector.cbegin();
  auto const elements_to_V = V_vector.cbegin();
  auto const nodes_to_m = m_vector->begin();
  auto const lumping_factor = 1.0 / double(nodes_in_element.size());
  auto functor = [=] (int const node) {
    double m(0.0);
    auto const range = nodes_to_node_elements[node];
    for (auto const node_element : range) {
      auto const element = node_elements_to_elements[node_element];
      auto const rho = elements_to_rho[element];
      auto const V = elements_to_V[element];
      m = m + (rho * V) * lumping_factor;
    }
    nodes_to_m[node] = m;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE initialize_V_h(
    int_range const nodes,
    int_range const nodes_in_element,
    int_range_sum<host_allocator<int>> const& nodes_to_node_elements_vector,
    host_vector<int> const& node_elements_to_elements_vector,
    host_vector<double> const& V_vector,
    host_vector<double>* V_h_vector) {
  auto const nodes_to_node_elements = nodes_to_node_elements_vector.cbegin();
  auto const node_elements_to_elements = node_elements_to_elements_vector.cbegin();
  auto const elements_to_V = V_vector.cbegin();
  auto const nodes_to_V_h = V_h_vector->begin();
  auto const lumping_factor = 1.0 / double(nodes_in_element.size());
  auto functor = [=] (int const node) {
    double V_h(0.0);
    auto const range = nodes_to_node_elements[node];
    for (auto const node_element : range) {
      auto const element = node_elements_to_elements[node_element];
      auto const V = elements_to_V[element];
      V_h = V_h + V * lumping_factor;
    }
    nodes_to_V_h[node] = V_h;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE collect_domain_entities(
    int_range const nodes,
    domain const& domain,
    decltype(state::x) const& x_vector,
    host_vector<int>* entities)
{
  host_vector<int> is_on(nodes.size());
  lgr::fill(is_on, int(0));
  domain.mark(x_vector, int(1), &is_on);
  host_vector<int> offsets(nodes.size());
  std::partial_sum(is_on.cbegin(), is_on.cend(), offsets.begin());
  int const domain_size = std::accumulate(is_on.cbegin(), is_on.cend(), 0);
  entities->resize(domain_size);
  auto const domain_ents_to_ents = entities->begin();
  auto const nodes_to_offsets = offsets.cbegin();
  auto const nodes_are_on = is_on.cbegin();
  auto functor2 = [=] (int const node) {
    if (nodes_are_on[node]) {
      domain_ents_to_ents[nodes_to_offsets[node] - 1] = node;
    }
  };
  lgr::for_each(nodes, functor2);
}

static void LGR_NOINLINE zero_acceleration(
    host_vector<int> const& domain,
    vector3<double> const axis,
    host_vector<vector3<double>>* a_vector) {
  auto const nodes_to_a = a_vector->begin();
  auto functor = [=] (int const node) {
    vector3<double> const old_a = nodes_to_a[node];
    auto const new_a = old_a - axis * (old_a * axis);
    nodes_to_a[node] = new_a;
  };
  lgr::for_each(domain, functor);
}

static void LGR_NOINLINE update_symm_grad_v(state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const nodes_to_v = s.v.cbegin();
  auto const elements_to_symm_grad_v = s.symm_grad_v.begin();
  auto functor = [=] (int const element) {
    auto grad_v = matrix3x3<double>::zero();
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      auto const node = element_nodes_to_nodes[element_node];
      vector3<double> const v = nodes_to_v[node];
      vector3<double> const grad_N = element_nodes_to_grad_N[element_node];
      grad_v = grad_v + outer_product(v, grad_N);
    }
    symmetric3x3<double> const symm_grad_v(grad_v);
    elements_to_symm_grad_v[element] = symm_grad_v;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_e(state& s, double const dt)
{
  auto const elements_to_sigma = s.sigma.cbegin();
  auto const elements_to_rho = s.rho.cbegin();
  auto const elements_to_old_e = s.old_e.cbegin();
  auto const elements_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const elements_to_e = s.e.begin();
  auto functor = [=] (int const element) {
    symmetric3x3<double> const symm_grad_v = elements_to_symm_grad_v[element];
    symmetric3x3<double> const sigma = elements_to_sigma[element];
    auto const rho_e_dot = inner_product(sigma, symm_grad_v);
    double const rho = elements_to_rho[element];
    auto const e_dot = rho_e_dot / rho;
    double const old_e = elements_to_old_e[element];
    auto const e = old_e + dt * e_dot;
    elements_to_e[element] = e;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_nu_art(input const& in, state& s) {
  auto const elements_to_nu_art = s.nu_art.begin();
  auto const elements_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const elements_to_h_art = s.h_art.cbegin();
  auto const elements_to_c = s.c.cbegin();
  auto const c1 = in.linear_artificial_viscosity;
  auto const c2 = in.quadratic_artificial_viscosity;
  auto functor = [=] (int const element) {
    symmetric3x3<double> const symm_grad_v = elements_to_symm_grad_v[element];
    double const div_v = trace(symm_grad_v);
    if (div_v >= 0.0) {
      elements_to_nu_art[element] = 0.0;
      return;
    }
    double const h_art = elements_to_h_art[element];
    double const c = elements_to_c[element];
    double const nu_art = c1 * ((-div_v) * (h_art * h_art)) + c2 * c * h_art;
    elements_to_nu_art[element] = nu_art;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE resize_physics(input const& in, state& s) {
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.old_v.resize(s.nodes.size());
  s.V.resize(s.elements.size());
  s.grad_N.resize(s.elements.size() * s.nodes_in_element.size());
  s.F_total.resize(s.elements.size());
  s.sigma.resize(s.elements.size());
  s.symm_grad_v.resize(s.elements.size());
  s.p.resize(s.elements.size());
  s.K.resize(s.elements.size());
  s.G.resize(s.elements.size());
  s.c.resize(s.elements.size());
  s.element_f.resize(s.elements.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.elements.size());
  s.e.resize(s.elements.size());
  s.old_e.resize(s.elements.size());
  s.m.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.elements.size());
  s.element_dt.resize(s.elements.size());
  if (in.enable_nodal_pressure) {
    s.p_h.resize(s.nodes.size());
    s.p_h_dot.resize(s.nodes.size());
    s.old_p_h.resize(s.nodes.size());
    s.v_prime.resize(s.elements.size());
    s.W.resize(s.elements.size() * s.nodes_in_element.size());
  }
  if (in.enable_nodal_volume) {
    s.p_h.resize(s.nodes.size());
    s.J_h.resize(s.nodes.size());
    s.V_h.resize(s.nodes.size());
    s.V_h_dot.resize(s.nodes.size());
    s.V_h_incr.resize(s.nodes.size());
    s.v_prime.resize(s.elements.size());
    s.W.resize(s.elements.size() * s.nodes_in_element.size());
  }
}

static void LGR_NOINLINE update_material_state(input const& in, state& s) {
  if (in.enable_neo_Hookean) {
    neo_Hookean(s.elements, in.K0, in.G0, s.F_total, &s.sigma, &s.K, &s.G);
    if (in.enable_nodal_volume) {
      nodal_neo_Hookean(s.nodes, in.K0, s.J_h, &s.p_h);
    }
  }
  else {
    lgr::fill(s.sigma, symmetric3x3<double>::zero());
    lgr::fill(s.K, double(0.0));
    lgr::fill(s.G, double(0.0));
  }
  if (in.enable_ideal_gas) {
    ideal_gas(s.elements, in.gamma, s.rho, s.e, &s.sigma, &s.K);
  }
}

static void LGR_NOINLINE update_a_from_material_state(input const& in, state& s) {
  if (in.enable_nodal_pressure) {
    update_sigma_with_p_h(s.elements, s.nodes_in_element,
        s.elements_to_nodes, s.p_h, &s.sigma);
  }
  update_element_force(s.elements, s.nodes_in_element, s.sigma, s.V, s.grad_N, &s.element_f);
  update_nodal_force(s);
  update_a(s.nodes, s.f, s.m, &s.a);
  for (auto const& cond : in.zero_acceleration_conditions) {
    zero_acceleration(s.node_sets[cond.node_set_name], cond.axis, &s.a);
  }
}

static void LGR_NOINLINE update_p_h_dot_from_a(input const& in, state& s) {
  if (in.enable_nodal_pressure) {
    update_v_prime(in, s);
    update_p_h_W(s);
    update_any_h_dot(s.nodes, s.elements, s.nodes_in_element,
        s.nodes_to_node_elements, s.node_elements_to_elements,
        s.node_elements_to_nodes_in_element,
        s.W, s.V, &s.p_h_dot);
  }
}

static void LGR_NOINLINE update_V_h_dot_from_a(input const& in, state& s) {
  if (in.enable_nodal_volume) {
    update_v_prime(in, s);
    update_V_h_W(s);
    update_any_h_dot(s.nodes, s.elements, s.nodes_in_element,
        s.nodes_to_node_elements, s.node_elements_to_elements,
        s.node_elements_to_nodes_in_element,
        s.W, s.V, &s.V_h_dot);
  }
}

static void LGR_NOINLINE midpoint_predictor_corrector_step(input const& in, state& s) {
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  if (in.enable_nodal_volume) {
    lgr::fill(s.V_h_incr, double(0.0));
  }
  lgr::copy(s.v, s.old_v);
  lgr::copy(s.e, s.old_e);
  if (in.enable_nodal_pressure) lgr::copy(s.p_h, s.old_p_h);
  constexpr int npc = 2;
  for (int pc = 0; pc < npc; ++pc) {
    if (in.enable_nodal_pressure) {
      update_p_h(s.nodes, (s.dt / 2.0), s.p_h_dot, s.old_p_h, &s.p_h);
    }
    if (pc == 0) advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
    update_v(s, s.dt / 2.0, s.old_v);
    update_symm_grad_v(s);
    bool const last_pc = (pc == (npc - 1));
    auto const half_dt = last_pc ? s.dt : s.dt / 2.0;
    update_e(s, half_dt);
    update_u(s.nodes, half_dt, s.v, &s.u);
    if (in.enable_nodal_volume) {
      update_V_h_incr(s.nodes, half_dt, s.V_h_dot, &s.V_h_incr);
    }
    if (last_pc) update_v(s, s.dt, s.old_v);
    update_x(s.nodes, s.u, &s.x);
    update_reference(s.elements, s.nodes_in_element, s.elements_to_nodes,
        s.u, &s.F_total, &s.grad_N, &s.V, &s.rho);
    if (in.enable_nodal_volume) {
      update_nodal_reference(s.nodes, s.V_h_incr, &s.V_h, &s.J_h);
    }
    if (in.enable_viscosity) update_h_art(in, s);
    update_symm_grad_v(s);
    update_h_min(in, s);
    update_material_state(in, s);
    update_c(s);
    if (in.enable_viscosity) update_nu_art(in, s);
    if (last_pc) update_element_dt(s);
    if (last_pc) find_max_stable_dt(s.element_dt, &s.max_stable_dt);
    update_a_from_material_state(in, s);
    update_p_h_dot_from_a(in, s);
    update_V_h_dot_from_a(in, s);
    if (last_pc) update_p(s.elements, s.sigma, &s.p);
  }
}

static void LGR_NOINLINE velocity_verlet_step(input const& in, state& s) {
  advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
  update_v(s, s.dt / 2.0, s.v);
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  update_u(s.nodes, s.dt, s.v, &s.u);
  update_x(s.nodes, s.u, &s.x);
  update_reference(s.elements, s.nodes_in_element, s.elements_to_nodes,
      s.u, &s.F_total, &s.grad_N, &s.V, &s.rho);
  update_h_min(in, s);
  update_material_state(in, s);
  update_c(s);
  update_element_dt(s);
  find_max_stable_dt(s.element_dt, &s.max_stable_dt);
  update_a_from_material_state(in, s);
  update_p_h_dot_from_a(in, s);
  update_p(s.elements, s.sigma, &s.p);
  update_v(s, s.dt / 2.0, s.v);
}

static void LGR_NOINLINE time_integrator_step(input const& in, state& s) {
  switch (in.time_integrator) {
    case MIDPOINT_PREDICTOR_CORRECTOR:
      midpoint_predictor_corrector_step(in, s);
      break;
    case VELOCITY_VERLET:
      velocity_verlet_step(in, s);
      break;
  }
}

void run(input const& in) {
  auto const num_file_outputs = in.num_file_outputs;
  double const file_output_period = num_file_outputs ? in.end_time / num_file_outputs : 0.0;
  state s;
  build_mesh(in, s);
  if (in.x_transform) in.x_transform(&s.x);
  for (auto const& pair : in.node_sets) {
    auto const& domain_name = pair.first;
    auto const& domain_ptr = pair.second;
    collect_domain_entities(s.nodes, *domain_ptr, s.x, &s.node_sets[domain_name]);
  }
  resize_physics(in, s);
  lgr::fill(s.rho, in.rho0);
  lgr::fill(s.e, in.e0);
  in.initial_v(s.nodes, s.x, &s.v);
  initialize_V(in, s);
  if (in.enable_viscosity) update_h_art(in, s);
  update_nodal_mass(s.nodes, s.nodes_in_element, s.nodes_to_node_elements, s.node_elements_to_elements, s.rho, s.V, &s.m);
  if (in.enable_nodal_volume) {
    initialize_V_h(s.nodes, s.nodes_in_element, s.nodes_to_node_elements, s.node_elements_to_elements, s.V, &s.V_h);
    lgr::fill(s.J_h, double(1.0));
  }
  initialize_grad_N(in, s);
  lgr::fill(s.F_total, matrix3x3<double>::identity());
  update_symm_grad_v(s);
  update_h_min(in, s);
  update_material_state(in, s);
  update_c(s);
  lgr::fill(s.nu_art, double(0.0));
  update_element_dt(s);
  find_max_stable_dt(s.element_dt, &s.max_stable_dt);
  update_a_from_material_state(in, s);
  update_p_h_dot_from_a(in, s);
  update_V_h_dot_from_a(in, s);
  update_p(s.elements, s.sigma, &s.p);
  file_writer output_file(in.name);
  s.next_file_output_time = num_file_outputs ? 0.0 : in.end_time;
  int file_output_index = 0;
  while (s.time < in.end_time) {
    if (num_file_outputs) {
      if (in.output_to_command_line) {
        print(std::cout, "outputting file n ", file_output_index, " time ", s.time, "\n");
      }
      output_file(in, file_output_index, s);
      ++file_output_index;
      s.next_file_output_time = file_output_index * file_output_period;
      s.next_file_output_time = std::min(s.next_file_output_time, in.end_time);
    }
    while (s.time < s.next_file_output_time) {
      if (in.output_to_command_line) {
        print(std::cout, "step ", s.n, " time ", s.time, " dt ", s.max_stable_dt, "\n");
      }
      time_integrator_step(in, s);
      ++s.n;
    }
  }
  if (num_file_outputs) {
    if (in.output_to_command_line) {
      print(std::cout, "outputting last file n ", file_output_index, " time ", s.time, "\n");
    }
    output_file(in, file_output_index, s);
  }
  if (in.output_to_command_line) {
    print(std::cout, "final time ", s.time, "\n");
  }
}

}

