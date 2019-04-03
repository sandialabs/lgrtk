#include <iostream>
#include <cassert>
#include <iomanip>

#include <lgr_state.hpp>
#include <lgr_run.hpp>
#include <lgr_counting_range.hpp>
#include <lgr_physics_types.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>
#include <lgr_print.hpp>
#include <lgr_vtk.hpp>
#include <lgr_for_each.hpp>
#include <lgr_reduce.hpp>
#include <lgr_fill.hpp>
#include <lgr_copy.hpp>
#include <lgr_element_specific.hpp>
#include <lgr_meshing.hpp>
#include <lgr_input.hpp>

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

static void LGR_NOINLINE update_u(state& s, double const dt) {
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.cbegin();
  auto functor = [=] (node_index const node) {
    vector3<double> const old_u = nodes_to_u[node];
    vector3<double> const v = nodes_to_v[node];
    nodes_to_u[node] = (dt * v) - old_u;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_v(state& s, double const dt, device_vector<vector3<double>, node_index> const& old_v_vector) {
  auto const nodes_to_v = s.v.begin();
  auto const nodes_to_old_v = old_v_vector.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto functor = [=] (node_index const node) {
    vector3<double> const old_v = nodes_to_old_v[node];
    vector3<double> const a = nodes_to_a[node];
    vector3<double> const v = old_v + dt * a;
    nodes_to_v[node] = v;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_p_h(state& s, double const dt) {
  auto const nodes_to_p_h = s.p_h.begin();
  auto const nodes_to_old_p_h = s.old_p_h.cbegin();
  auto const nodes_to_p_h_dot = s.p_h_dot.cbegin();
  auto functor = [=] (node_index const node) {
    double const old_p_h = nodes_to_old_p_h[node];
    double const p_h_dot = nodes_to_p_h_dot[node];
    double const p_h = old_p_h + dt * p_h_dot;
    nodes_to_p_h[node] = p_h;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_sigma_with_p_h(state& s) {
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_element_points = s.elements * s.points_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_to_p_h = s.p_h.cbegin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_sigma = s.sigma.begin();
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    auto const element_points = elements_to_element_points[element];
    for (auto const point : element_points) {
      double point_p_h = 0.0;
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const node = element_nodes_to_nodes[element_node];
        double const p_h = nodes_to_p_h[node];
        point_p_h = point_p_h + N * p_h;
      }
      symmetric3x3<double> const old_sigma = points_to_sigma[point];
      auto const new_sigma = deviator(old_sigma) - point_p_h;
      points_to_sigma[point] = new_sigma;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_a(state& s) {
  auto const nodes_to_f = s.f.cbegin();
  auto const nodes_to_m = s.m.cbegin();
  auto const nodes_to_a = s.a.begin();
  auto functor = [=] (node_index const node) {
    vector3<double> const f = nodes_to_f[node];
    double const m = nodes_to_m[node];
    vector3<double> const a = f / m;
    nodes_to_a[node] = a;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_x(state& s) {
  auto const nodes_to_u = s.u.cbegin();
  auto const nodes_to_x = s.x.begin();
  auto functor = [=] (node_index const node) {
    vector3<double> const old_x = nodes_to_x[node];
    vector3<double> const u = nodes_to_u[node];
    vector3<double> const new_x = old_x + u;
    nodes_to_x[node] = new_x;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_p(state& s) {
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_p = s.p.begin();
  auto functor = [=] (point_index const point) {
    symmetric3x3<double> const sigma = points_to_sigma[point];
    auto const p = -(1.0 / 3.0) * trace(sigma);
    points_to_p[point] = p;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_reference(state& s) {
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_element_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_u = s.u.cbegin();
  auto const points_to_F_total = s.F_total.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_V = s.V.begin();
  auto const points_to_rho = s.rho.begin();
  auto const nodes_in_element = s.nodes_in_element;
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    auto const element_points = elements_to_element_points[element];
    for (auto const point : element_points) {
      auto const point_nodes = points_to_point_nodes[point];
      auto F_incr = matrix3x3<double>::identity();
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        auto const node = element_nodes_to_nodes[element_node];
        vector3<double> const u = nodes_to_u[node];
        vector3<double> const old_grad_N = point_nodes_to_grad_N[point_node];
        F_incr = F_incr + outer_product(u, old_grad_N);
      }
      auto const F_inverse_transpose = transpose(inverse(F_incr));
      for (auto const point_node : point_nodes) {
        vector3<double> const old_grad_N = point_nodes_to_grad_N[point_node];
        auto const new_grad_N = F_inverse_transpose * old_grad_N;
        point_nodes_to_grad_N[point_node] = new_grad_N;
      }
      matrix3x3<double> const old_F_total = points_to_F_total[point];
      matrix3x3<double> const new_F_total = F_incr * old_F_total;
      points_to_F_total[point] = new_F_total;
      auto const J = determinant(F_incr);
      assert(J > 0.0);
      double const old_V = points_to_V[point];
      auto const new_V = J * old_V;
      assert(new_V > 0.0);
      points_to_V[point] = new_V;
      auto const old_rho = points_to_rho[point];
      auto const new_rho = old_rho / J;
      points_to_rho[point] = new_rho;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_c(state& s)
{
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_K = s.K.cbegin();
  auto const points_to_G = s.G.cbegin();
  auto const points_to_c = s.c.begin();
  auto functor = [=] (point_index const point) {
    double const rho = points_to_rho[point];
    double const K = points_to_K[point];
    double const G = points_to_G[point];
    auto const M = K + (4.0 / 3.0) * G;
    auto const c = std::sqrt(M / rho);
    points_to_c[point] = c;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_element_dt(state& s) {
  auto const points_to_c = s.c.cbegin();
  auto const elements_to_h_min = s.h_min.cbegin();
  auto const points_to_nu_art = s.nu_art.cbegin();
  auto const points_to_dt = s.element_dt.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double const h_min = elements_to_h_min[element];
    for (auto const point : elements_to_points[element]) {
      auto const c = points_to_c[point];
      auto const nu_art = points_to_nu_art[point];
      auto const h_sq = h_min * h_min;
      auto const c_sq = c * c;
      auto const nu_art_sq = nu_art * nu_art;
      auto const dt = h_sq / (nu_art + std::sqrt(nu_art_sq + (c_sq * h_sq)));
      assert(dt > 0.0);
      points_to_dt[point] = dt;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE find_max_stable_dt(state& s)
{
  double const init = std::numeric_limits<double>::max();
  s.max_stable_dt = lgr::transform_reduce(s.element_dt, init, lgr::minimum<double>(), lgr::identity<double>());
}

static void LGR_NOINLINE update_v_prime(input const& in, state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_dt = s.element_dt.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto const nodes_to_p_h = s.p_h.cbegin();
  auto const points_to_v_prime = s.v_prime.begin();
  auto const c_tau = in.c_tau;
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      auto const point_nodes = points_to_point_nodes[point];
      double const dt = points_to_dt[point];
      auto const tau = c_tau * dt;
      auto grad_p = vector3<double>::zero();
      auto a = vector3<double>::zero();
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
        double const p_h = nodes_to_p_h[node];
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        grad_p = grad_p + (grad_N * p_h);
        vector3<double> const a_of_node = nodes_to_a[node];
        a = a + a_of_node;
      }
      a = a * N;
      double const rho = points_to_rho[point];
      auto const v_prime = -(tau / rho) * (rho * a + grad_p);
      points_to_v_prime[point] = v_prime;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_q(input const& in, state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_dt = s.element_dt.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto const nodes_to_p_h = s.p_h.cbegin();
  auto const points_to_q = s.q.begin();
  auto const c_tau = in.c_tau;
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double const dt = points_to_dt[point];
      auto const tau = c_tau * dt;
      auto grad_p = vector3<double>::zero();
      auto a = vector3<double>::zero();
      double p_h = 0.0;
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
        double const p_h_of_node = nodes_to_p_h[node];
        p_h = p_h + p_h_of_node;
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        grad_p = grad_p + (grad_N * p_h_of_node);
        vector3<double> const a_of_node = nodes_to_a[node];
        a = a + a_of_node;
      }
      a = a * N;
      p_h = p_h * N;
      double const rho = points_to_rho[point];
      auto const v_prime = -(tau / rho) * (rho * a + grad_p);
      auto const q = p_h * v_prime;
      points_to_q[point] = q;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_p_h_W(state& s)
{
  auto const points_to_K = s.K.cbegin();
  auto const points_to_v_prime = s.v_prime.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const points_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const point_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] (point_index const point) {
    symmetric3x3<double> symm_grad_v = points_to_symm_grad_v[point];
    double const div_v = trace(symm_grad_v);
    double const K = points_to_K[point];
    double const V = points_to_V[point];
    vector3<double> const v_prime = points_to_v_prime[point];
    auto const point_nodes = points_to_point_nodes[point];
    for (auto const point_node : point_nodes) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
      double const p_h_dot =
        -(N * (K * div_v)) + (grad_N * (K * v_prime));
      double const W = p_h_dot * V;
      point_nodes_to_W[point_node] = W;
    }
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_e_h_W(state& s)
{
  auto const points_to_q = s.q.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const points_to_rho_e_dot = s.rho_e_dot.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const point_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] (point_index const point) {
    double const rho_e_dot = points_to_rho_e_dot[point];
    double const V = points_to_V[point];
    vector3<double> const q = points_to_q[point];
    auto const point_nodes = points_to_point_nodes[point];
    for (auto const point_node : point_nodes) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
      double const rho_e_h_dot = (N * rho_e_dot) + (grad_N * q);
      double const W = rho_e_h_dot * V;
      point_nodes_to_W[point_node] = W;
    }
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_p_h_dot(state& s)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const point_nodes_to_W = s.W.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_p_h_dot = s.p_h_dot.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (node_index const node) {
    double node_W = 0.0;
    double node_V = 0.0;
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const point_nodes = points_to_point_nodes[point];
        auto const point_node = point_nodes[node_in_element];
        double const W = point_nodes_to_W[point_node];
        double const V = points_to_V[point];
        node_W = node_W + W;
        node_V = node_V + (N * V);
      }
    }
    auto const p_h_dot = node_W / node_V;
    nodes_to_p_h_dot[node] = p_h_dot;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_e_h_dot(state& s)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const point_nodes_to_W = s.W.cbegin();
  auto const nodes_to_e_h_dot = s.e_h_dot.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_to_m = s.m.cbegin();
  auto functor = [=] (node_index const node) {
    double node_W = 0.0;
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const point_nodes = points_to_point_nodes[point];
        auto const point_node = point_nodes[node_in_element];
        double const W = point_nodes_to_W[point_node];
        node_W = node_W + W;
      }
    }
    double const m = nodes_to_m[node];
    auto const e_h_dot = node_W / m;
    nodes_to_e_h_dot[node] = e_h_dot;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE neo_Hookean(input const& in, state& s) {
  auto const points_to_F_total = s.F_total.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const points_to_K = s.K.begin();
  auto const points_to_G = s.G.begin();
  auto const K0 = in.K0;
  auto const G0 = in.G0;
  auto functor = [=] (point_index const point) {
    matrix3x3<double> const F = points_to_F_total[point];
    auto const J = determinant(F);
    auto const Jinv = 1.0 / J;
    auto const half_K0 = 0.5 * K0;
    auto const Jm13 = 1.0 / std::cbrt(J);
    auto const Jm23 = Jm13 * Jm13;
    auto const Jm53 = (Jm23 * Jm23) * Jm13;
    auto const B = self_times_transpose(F);
    auto const devB = deviator(B);
    auto const sigma = half_K0 * (J - Jinv) + (G0 * Jm53) * devB;
    points_to_sigma[point] = sigma;
    auto const K = half_K0 * (J + Jinv);
    points_to_K[point] = K;
    points_to_G[point] = G0;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE ideal_gas(input const& in, state& s) {
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_e = s.e.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const points_to_K = s.K.begin();
  auto const gamma = in.gamma;
  auto functor = [=] (point_index const point) {
    double const rho = points_to_rho[point];
    assert(rho > 0.0);
    double const e = points_to_e[point];
    assert(e > 0.0);
    auto const p = (gamma - 1.0) * (rho * e);
    assert(p > 0.0);
    symmetric3x3<double> const old_sigma = points_to_sigma[point];
    auto const new_sigma = deviator(old_sigma) - p;
    points_to_sigma[point] = new_sigma;
    auto const K = gamma * p;
    assert(K > 0.0);
    points_to_K[point] = K;
//  if (point < point_index(6)) std::cout << "element " << int(point) << " rho " << rho << " e " << e << " p " << p << " K " << K << '\n';
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE nodal_ideal_gas(input const& in, state& s) {
  auto const nodes_to_rho = s.rho_h.cbegin();
  auto const nodes_to_e = s.e_h.cbegin();
  auto const nodes_to_p = s.p_h.begin();
  auto const gamma = in.gamma;
  auto functor = [=] (node_index const node) {
    double const rho = nodes_to_rho[node];
    assert(rho > 0.0);
    double const e = nodes_to_e[node];
    assert(e > 0.0);
    auto const p = (gamma - 1.0) * (rho * e);
    assert(p > 0.0);
    nodes_to_p[node] = p;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_element_force(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const point_nodes_to_f = s.element_f.begin();
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] (point_index const point) {
    symmetric3x3<double> const sigma = points_to_sigma[point];
    double const V = points_to_V[point];
    auto const point_nodes = points_to_point_nodes[point];
    for (auto const point_node : point_nodes) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
      auto const f = -(sigma * grad_N) * V;
      point_nodes_to_f[point_node] = f;
    }
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_nodal_force(state& s) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const point_nodes_to_f = s.element_f.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (node_index const node) {
    auto node_f = vector3<double>::zero();
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const point_nodes = points_to_point_nodes[point];
        auto const point_node = point_nodes[node_in_element];
        vector3<double> const point_f = point_nodes_to_f[point_node];
        node_f = node_f + point_f;
      }
    }
    nodes_to_f[node] = node_f;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_nodal_mass(state& s) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_m = s.m.begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (node_index const node) {
    double m(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      for (auto const point : elements_to_points[element]) {
        double const rho = points_to_rho[point];
        double const V = points_to_V[point];
        m = m + (rho * V) * N;
      }
    }
    nodes_to_m[node] = m;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE update_nodal_density(state& s)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_m = s.m.cbegin();
  auto const nodes_to_rho_h = s.rho_h.begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (node_index const node) {
    double node_V(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const V = points_to_V[point];
        node_V = node_V + (N * V);
      }
    }
    double const m = nodes_to_m[node];
    nodes_to_rho_h[node] = m / node_V;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE zero_acceleration(
    device_vector<node_index, int> const& domain,
    vector3<double> const axis,
    device_vector<vector3<double>, node_index>* a_vector) {
  auto const nodes_to_a = a_vector->begin();
  auto functor = [=] (node_index const node) {
    vector3<double> const old_a = nodes_to_a[node];
    auto const new_a = old_a - axis * (old_a * axis);
    nodes_to_a[node] = new_a;
  };
  lgr::for_each(domain, functor);
}

static void LGR_NOINLINE update_symm_grad_v(state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const nodes_to_v = s.v.cbegin();
  auto const points_to_symm_grad_v = s.symm_grad_v.begin();
  auto const nodes_in_element = s.nodes_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      auto grad_v = matrix3x3<double>::zero();
//    if (point < point_index(6)) std::cout << "element " << int(point) << " symm_grad_v calc:\n";
      auto const element_nodes = elements_to_element_nodes[element];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
//      if (point < point_index(6)) std::cout << "node " << int(node_in_element) << " ID " << int(node) << '\n'; 
        vector3<double> const v = nodes_to_v[node];
//      if (point < point_index(6)) std::cout << "node " << int(node_in_element) << " v " << v << '\n'; 
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
//      if (point < point_index(6)) std::cout << "node " << int(node_in_element) << " grad_N " << grad_N << '\n'; 
        grad_v = grad_v + outer_product(v, grad_N);
      }
//    if (point < point_index(6)) std::cout << "grad_v:\n" << grad_v;
      symmetric3x3<double> const symm_grad_v(grad_v);
//    if (point < point_index(6)) std::cout << "symm_grad_v:\n" << symm_grad_v;
      points_to_symm_grad_v[point] = symm_grad_v;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_rho_e_dot(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const points_to_rho_e_dot = s.rho_e_dot.begin();
  auto functor = [=] (point_index const point) {
    symmetric3x3<double> const symm_grad_v = points_to_symm_grad_v[point];
//  if (point < point_index(6)) std::cout << "element " << int(point) << " rho_e_dot calc:\n";
//  if (point < point_index(6)) std::cout << "element " << int(point) << " symm_grad_v\n" << symm_grad_v;
    symmetric3x3<double> const sigma = points_to_sigma[point];
//  if (point < point_index(6)) std::cout << "element " << int(point) << " sigma\n" << sigma;
//  if (point < point_index(6)) std::cout << "diagonal inner product " <<
//    (sigma(0,0) * symm_grad_v(0,0) + sigma(1,1) * symm_grad_v(1,1) + sigma(2,2) * symm_grad_v(2,2)) << '\n';
//  if (point < point_index(6)) std::cout << "off-diagonal inner product " <<
//    (sigma(0,1) * symm_grad_v(0,1) + sigma(1,2) * symm_grad_v(1,2) + sigma(0,2) * symm_grad_v(0,2)) << '\n';
//  if (point < point_index(6)) std::cout << "expected inner_product " <<
//    ((sigma(0,0) * symm_grad_v(0,0) + sigma(1,1) * symm_grad_v(1,1) + sigma(2,2) * symm_grad_v(2,2)) + 2.0 *
//     (sigma(0,1) * symm_grad_v(0,1) + sigma(1,2) * symm_grad_v(1,2) + sigma(0,2) * symm_grad_v(0,2))) << '\n';
    auto const rho_e_dot = inner_product(sigma, symm_grad_v);
//  if (point < point_index(6)) std::cout << "element " << int(point) << " rho_e_dot " << rho_e_dot << '\n';
    points_to_rho_e_dot[point] = rho_e_dot;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_e(state& s, double const dt)
{
  auto const points_to_rho_e_dot = s.rho_e_dot.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_old_e = s.old_e.cbegin();
  auto const points_to_e = s.e.begin();
  auto functor = [=] (point_index const point) {
    auto const rho_e_dot = points_to_rho_e_dot[point];
    double const rho = points_to_rho[point];
    auto const e_dot = rho_e_dot / rho;
    double const old_e = points_to_old_e[point];
    auto const e = old_e + dt * e_dot;
    points_to_e[point] = e;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE interpolate_e(state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_e_h = s.e_h.cbegin();
  auto const points_to_e = s.e.begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double e = 0.0;
      for (auto const element_node : element_nodes) {
        node_index const node = element_nodes_to_nodes[element_node];
        double const e_h = nodes_to_e_h[node];
        e = e + e_h;
      }
      e = e * N;
      points_to_e[point] = e;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE interpolate_rho(state& s)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const nodes_to_rho_h = s.rho_h.cbegin();
  auto const points_to_rho = s.rho.begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double rho = 0.0;
      for (auto const element_node : element_nodes) {
        node_index const node = element_nodes_to_nodes[element_node];
        double const rho_h = nodes_to_rho_h[node];
        rho = rho + rho_h;
      }
      rho = rho * N;
      points_to_rho[point] = rho;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_e_h(state& s, double const dt)
{
  auto const nodes_to_e_h_dot = s.e_h_dot.cbegin();
  auto const nodes_to_old_e_h = s.old_e_h.cbegin();
  auto const nodes_to_e_h = s.e_h.begin();
  auto functor = [=] (node_index const node) {
    auto const e_h_dot = nodes_to_e_h_dot[node];
    double const old_e_h = nodes_to_old_e_h[node];
    auto const e_h = old_e_h + dt * e_h_dot;
    nodes_to_e_h[node] = e_h;
  };
  lgr::for_each(s.nodes, functor);
}

static void LGR_NOINLINE apply_viscosity(input const& in, state& s) {
  auto const points_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const elements_to_h_art = s.h_art.cbegin();
  auto const points_to_c = s.c.cbegin();
  auto const c1 = in.quadratic_artificial_viscosity;
  auto const c2 = in.linear_artificial_viscosity;
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const points_to_nu_art = s.nu_art.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double const h_art = elements_to_h_art[element];
    for (auto const point : elements_to_points[element]) {
      symmetric3x3<double> const symm_grad_v = points_to_symm_grad_v[point];
      double const div_v = trace(symm_grad_v);
      if (div_v >= 0.0) {
        points_to_nu_art[point] = 0.0;
      } else {
        double const c = points_to_c[point];
        double const nu_art = c1 * ((-div_v) * (h_art * h_art)) + c2 * c * h_art;
        points_to_nu_art[point] = nu_art;
        double const rho = points_to_rho[point];
        auto const sigma_art = (rho * nu_art) * symm_grad_v;
        symmetric3x3<double> const sigma = points_to_sigma[point];
        auto const sigma_tilde = sigma + sigma_art;
        points_to_sigma[point] = sigma_tilde;
      }
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE volume_average_J(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const points_to_F = s.F_total.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double total_V0 = 0.0;
    double total_V = 0.0;
    for (auto const point : elements_to_points[element]) {
      matrix3x3<double> const F = points_to_F[point];
      auto const J = determinant(F);
      double const V = points_to_V[point];
      auto const V0 = V / J;
      total_V0 += V0;
      total_V += V;
    }
    auto const average_J = total_V / total_V0;
    for (auto const point : elements_to_points[element]) {
      matrix3x3<double> const old_F = points_to_F[point];
      auto const old_J = determinant(old_F);
      auto const new_F = std::cbrt(average_J / old_J) * old_F;
      points_to_F[point] = new_F;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE resize_physics(input const& in, state& s) {
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.old_v.resize(s.nodes.size());
  s.V.resize(s.points.size());
  s.grad_N.resize(s.points.size() * s.nodes_in_element.size());
  s.F_total.resize(s.points.size());
  s.sigma.resize(s.points.size());
  s.symm_grad_v.resize(s.points.size());
  s.p.resize(s.points.size());
  s.K.resize(s.points.size());
  s.G.resize(s.points.size());
  s.c.resize(s.points.size());
  s.element_f.resize(s.points.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.points.size());
  s.e.resize(s.points.size());
  s.old_e.resize(s.points.size());
  s.rho_e_dot.resize(s.points.size());
  s.m.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.points.size());
  s.element_dt.resize(s.points.size());
  if (in.enable_nodal_pressure) {
    s.p_h.resize(s.nodes.size());
    s.p_h_dot.resize(s.nodes.size());
    s.old_p_h.resize(s.nodes.size());
    s.v_prime.resize(s.points.size());
    s.W.resize(s.points.size() * s.nodes_in_element.size());
  }
  if (in.enable_nodal_energy) {
    s.p_h.resize(s.nodes.size());
    s.e_h.resize(s.nodes.size());
    s.old_e_h.resize(s.nodes.size());
    s.e_h_dot.resize(s.nodes.size());
    s.rho_h.resize(s.nodes.size());
    s.q.resize(s.points.size());
    s.W.resize(s.points.size() * s.nodes_in_element.size());
  }
}

static void LGR_NOINLINE update_material_state(input const& in, state& s) {
  if (in.enable_neo_Hookean) {
    neo_Hookean(in, s);
  }
  else {
    lgr::fill(s.sigma, symmetric3x3<double>::zero());
    lgr::fill(s.K, double(0.0));
    lgr::fill(s.G, double(0.0));
  }
  if (in.enable_ideal_gas) {
    ideal_gas(in, s);
    if (in.enable_nodal_energy) {
      nodal_ideal_gas(in, s);
    }
  }
  if (in.enable_nodal_pressure || in.enable_nodal_energy) {
    update_sigma_with_p_h(s);
  }
}

static void LGR_NOINLINE update_a_from_material_state(input const& in, state& s) {
  update_element_force(s);
  update_nodal_force(s);
  update_a(s);
  for (auto const& cond : in.zero_acceleration_conditions) {
    zero_acceleration(s.node_sets.find(cond.node_set_name)->second, cond.axis, &s.a);
  }
}

static void LGR_NOINLINE update_p_h_dot_from_a(input const& in, state& s) {
  if (in.enable_nodal_pressure) {
    update_v_prime(in, s);
    update_p_h_W(s);
    update_p_h_dot(s);
  }
}

static void LGR_NOINLINE update_e_h_dot_from_a(input const& in, state& s) {
  update_q(in, s);
  update_e_h_W(s);
  update_e_h_dot(s);
}

static void LGR_NOINLINE midpoint_predictor_corrector_step(input const& in, state& s) {
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  lgr::copy(s.v, s.old_v);
  lgr::copy(s.e, s.old_e);
  if (in.enable_nodal_pressure) lgr::copy(s.p_h, s.old_p_h);
  if (in.enable_nodal_energy) lgr::copy(s.e_h, s.old_e_h);
  constexpr int npc = 2;
  for (int pc = 0; pc < npc; ++pc) {
    if (pc == 0) advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
    update_v(s, s.dt / 2.0, s.old_v);
    update_symm_grad_v(s);
    bool const last_pc = (pc == (npc - 1));
    auto const half_dt = last_pc ? s.dt : s.dt / 2.0;
    if (in.enable_nodal_pressure) {
      update_p_h(s, half_dt);
    }
    update_rho_e_dot(s);
    if (in.enable_nodal_energy) {
      update_e_h_dot_from_a(in, s);
      update_e_h(s, half_dt);
      interpolate_e(s);
    } else {
      update_e(s, half_dt);
    }
    update_u(s, half_dt);
    if (last_pc) update_v(s, s.dt, s.old_v);
    update_x(s);
    update_reference(s);
    if (in.enable_J_averaging) volume_average_J(s);
    if (in.enable_nodal_energy) {
      update_nodal_density(s);
      interpolate_rho(s);
    }
    update_symm_grad_v(s);
    update_h_min(in, s);
    if (in.enable_viscosity) update_h_art(in, s);
    update_material_state(in, s);
    update_c(s);
    if (in.enable_viscosity) apply_viscosity(in, s);
    if (last_pc) update_element_dt(s);
    if (last_pc) find_max_stable_dt(s);
    update_a_from_material_state(in, s);
    update_p_h_dot_from_a(in, s);
    if (last_pc) update_p(s);
  }
}

static void LGR_NOINLINE velocity_verlet_step(input const& in, state& s) {
  advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
  update_v(s, s.dt / 2.0, s.v);
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  update_u(s, s.dt);
  update_x(s);
  update_reference(s);
  if (in.enable_J_averaging) volume_average_J(s);
  update_h_min(in, s);
  update_material_state(in, s);
  update_c(s);
  update_element_dt(s);
  find_max_stable_dt(s);
  update_a_from_material_state(in, s);
  update_p_h_dot_from_a(in, s);
  update_p(s);
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
  std::cout << std::scientific << std::setprecision(17);
  auto const num_file_outputs = in.num_file_outputs;
  double const file_output_period = num_file_outputs ? in.end_time / num_file_outputs : 0.0;
  state s;
  build_mesh(in, s);
  if (in.x_transform) in.x_transform(&s.x);
  for (auto const& pair : in.node_sets) {
    auto const& domain_name = pair.first;
    auto const& domain_ptr = pair.second;
    s.node_sets.emplace(domain_name, s.mempool);
    collect_domain_entities(s.nodes, *domain_ptr, s.x, &(s.node_sets.find(domain_name)->second));
  }
  resize_physics(in, s);
  lgr::fill(s.rho, in.rho0);
  lgr::fill(s.e, in.e0);
  lgr::fill(s.p_h, double(0.0));
  lgr::fill(s.e_h, in.e0);
  assert(in.initial_v);
  in.initial_v(s.nodes, s.x, &s.v);
  initialize_V(in, s);
  if (in.enable_viscosity) update_h_art(in, s);
  update_nodal_mass(s);
  if (in.enable_nodal_energy) update_nodal_density(s);
  initialize_grad_N(in, s);
  lgr::fill(s.F_total, matrix3x3<double>::identity());
  update_symm_grad_v(s);
  update_h_min(in, s);
  update_material_state(in, s);
  update_c(s);
  if (in.enable_viscosity) apply_viscosity(in, s);
  else lgr::fill(s.nu_art, double(0.0));
  update_element_dt(s);
  find_max_stable_dt(s);
  update_a_from_material_state(in, s);
  update_p_h_dot_from_a(in, s);
  update_p(s);
  file_writer output_file(in.name);
  s.next_file_output_time = num_file_outputs ? 0.0 : in.end_time;
  int file_output_index = 0;
  while (s.time < in.end_time) {
    if (num_file_outputs) {
      if (in.output_to_command_line) {
        std::cout << "outputting file n " << file_output_index << " time " << s.time << "\n";
      }
      output_file(in, file_output_index, s);
      ++file_output_index;
      s.next_file_output_time = file_output_index * file_output_period;
      s.next_file_output_time = std::min(s.next_file_output_time, in.end_time);
    }
    while (s.time < s.next_file_output_time) {
      if (in.output_to_command_line) {
        std::cout << "step " << s.n << " time " << s.time << " dt " << s.max_stable_dt << "\n";
      }
      time_integrator_step(in, s);
      ++s.n;
    }
  }
  if (num_file_outputs) {
    if (in.output_to_command_line) {
      std::cout << "outputting last file n " << file_output_index << " time " << s.time << "\n";
    }
    output_file(in, file_output_index, s);
  }
  if (in.output_to_command_line) {
    std::cout << "final time " << s.time << "\n";
  }
}

}

