#include <iostream>
#include <cassert>
#include <iomanip>

#include <lgr_state.hpp>
#include <lgr_physics.hpp>
#include <lgr_counting_range.hpp>
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
#include <lgr_stabilized.hpp>
#include <lgr_adapt.hpp>

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

static void LGR_NOINLINE update_a(state& s) {
  auto const nodes_to_f = s.f.cbegin();
  auto const nodes_to_m = s.mass.cbegin();
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

static void LGR_NOINLINE update_p(state& s, material_index const material) {
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_p = s.p.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      symmetric3x3<double> const sigma = points_to_sigma[point];
      auto const p = -(1.0 / 3.0) * trace(sigma);
      points_to_p[point] = p;
    }
  };
  for_each(s.element_sets[material], functor);
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

static void LGR_NOINLINE neo_Hookean(input const& in, state& s, material_index const material) {
  auto const points_to_F_total = s.F_total.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const points_to_K = s.K.begin();
  auto const points_to_G = s.G.begin();
  auto const K0 = in.K0[material];
  auto const G0 = in.G0[material];
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
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
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE ideal_gas(input const& in, state& s, material_index const material) {
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_e = s.e.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const points_to_K = s.K.begin();
  auto const gamma = in.gamma[material];
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
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
    }
  };
  lgr::for_each(s.element_sets[material], functor);
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
      auto const element_nodes = elements_to_element_nodes[element];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
        vector3<double> const v = nodes_to_v[node];
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        grad_v = grad_v + outer_product(v, grad_N);
      }
      symmetric3x3<double> const symm_grad_v(grad_v);
      points_to_symm_grad_v[point] = symm_grad_v;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE stress_power(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const points_to_rho_e_dot = s.rho_e_dot.begin();
  auto functor = [=] (point_index const point) {
    symmetric3x3<double> const symm_grad_v = points_to_symm_grad_v[point];
    symmetric3x3<double> const sigma = points_to_sigma[point];
    auto const rho_e_dot = inner_product(sigma, symm_grad_v);
    points_to_rho_e_dot[point] = rho_e_dot;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE update_e(state& s, double const dt,
    material_index const material,
    device_vector<double, point_index> const& old_e_vector)
{
  auto const points_to_rho_e_dot = s.rho_e_dot.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_old_e = old_e_vector.cbegin();
  auto const points_to_e = s.e.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      auto const rho_e_dot = points_to_rho_e_dot[point];
      double const rho = points_to_rho[point];
      auto const e_dot = rho_e_dot / rho;
      double const old_e = points_to_old_e[point];
      auto const e = old_e + dt * e_dot;
      points_to_e[point] = e;
    }
  };
  for_each(s.element_sets[material], functor);
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

static void LGR_NOINLINE volume_average_rho(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const points_to_rho = s.rho.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double mass = 0.0;
    double total_V = 0.0;
    for (auto const point : elements_to_points[element]) {
      double const rho = points_to_rho[point];
      double const V = points_to_V[point];
      mass += V * rho;
      total_V += V;
    }
    auto const average_rho = mass / total_V;
    for (auto const point : elements_to_points[element]) {
      points_to_rho[point] = average_rho;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE volume_average_e(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_e = s.e.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double energy = 0.0;
    double mass = 0.0;
    for (auto const point : elements_to_points[element]) {
      double const rho = points_to_rho[point];
      double const e = points_to_e[point];
      double const V = points_to_V[point];
      energy += V * (rho * e);
      mass += V * rho;
    }
    auto const average_e = energy / mass;
    for (auto const point : elements_to_points[element]) {
      points_to_e[point] = average_e;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE volume_average_p(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const points_to_sigma = s.sigma.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    double total_V = 0.0;
    double average_p = 0.0;
    for (auto const point : elements_to_points[element]) {
      symmetric3x3<double> const sigma = points_to_sigma[point];
      double const p = -(1.0 / 3.0) * trace(sigma);
      double const V = points_to_V[point];
      average_p += V * p;
      total_V += V;
    }
    average_p /= total_V;
    for (auto const point : elements_to_points[element]) {
      symmetric3x3<double> const old_sigma = points_to_sigma[point];
      auto const new_sigma = deviator(old_sigma) - average_p;
      points_to_sigma[point] = new_sigma;
    }
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_single_material_state(input const& in, state& s, material_index const material,
    double const dt,
    device_vector<double, node_index> const& old_p_h) {
  if (in.enable_neo_Hookean[material]) {
    neo_Hookean(in, s, material);
  }
  if (in.enable_ideal_gas[material]) {
    if (in.enable_nodal_energy[material]) {
      nodal_ideal_gas(in, s, material);
    } else {
      ideal_gas(in, s, material);
    }
  }
  if (in.enable_nodal_pressure[material] || in.enable_nodal_energy[material]) {
    if (in.enable_p_prime[material]) {
      update_sigma_with_p_h_p_prime(in, s, material, dt, old_p_h);
    } else {
      update_sigma_with_p_h(s, material);
    }
  }
}

static void LGR_NOINLINE update_material_state(input const& in, state& s,
    double const dt,
    host_vector<device_vector<double, node_index>, material_index> const& old_p_h) {
  lgr::fill(s.sigma, symmetric3x3<double>::zero());
  lgr::fill(s.G, double(0.0));
  for (auto const material : in.materials) {
    update_single_material_state(in, s, material, dt, old_p_h[material]);
  }
}

static void LGR_NOINLINE update_a_from_material_state(input const& in, state& s) {
  update_element_force(s);
  update_nodal_force(s);
  update_a(s);
  for (auto const& cond : in.zero_acceleration_conditions) {
    zero_acceleration(s.node_sets[cond.boundary], cond.axis, &s.a);
  }
}

static void LGR_NOINLINE midpoint_predictor_corrector_step(input const& in, state& s) {
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  device_vector<vector3<double>, node_index> old_v(s.nodes.size(), s.devpool);
  lgr::copy(s.v, old_v);
  device_vector<double, point_index> old_e(s.points.size(), s.devpool);
  lgr::copy(s.e, old_e);
  host_vector<device_vector<double, node_index>, material_index> old_p_h(in.materials.size(), s.devpool);
  host_vector<device_vector<double, node_index>, material_index> old_e_h(in.materials.size(), s.devpool);
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      old_p_h[material].resize(s.nodes.size());
      lgr::copy(s.p_h[material], old_p_h[material]);
    }
    if (in.enable_nodal_energy[material]) {
      if (in.enable_p_prime[material]) {
        old_p_h[material].resize(s.nodes.size());
        lgr::copy(s.p_h[material], old_p_h[material]);
      }
      old_e_h[material].resize(s.nodes.size());
      lgr::copy(s.e_h[material], old_e_h[material]);
    }
  }
  constexpr int npc = 2;
  for (int pc = 0; pc < npc; ++pc) {
    if (pc == 0) advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
    update_v(s, s.dt / 2.0, old_v);
    update_symm_grad_v(s);
    bool const last_pc = (pc == (npc - 1));
    auto const half_dt = last_pc ? s.dt : s.dt / 2.0;
    for (auto const material : in.materials) {
      if (in.enable_nodal_pressure[material]) {
        update_p_h(s, half_dt, material, old_p_h[material]);
      }
    }
    stress_power(s);
    for (auto const material : in.materials) {
      if (in.enable_nodal_energy[material]) {
        update_e_h_dot_from_a(in, s, material);
        update_e_h(s, half_dt, material, old_e_h[material]);
      } else {
        update_e(s, half_dt, material, old_e);
      }
    }
    if (in.enable_e_averaging) volume_average_e(s);
    update_u(s, half_dt);
    if (last_pc) update_v(s, s.dt, old_v);
    update_x(s);
    update_reference(s);
    if (in.enable_J_averaging) volume_average_J(s);
    if (in.enable_rho_averaging) volume_average_rho(s);
    for (auto const material : in.materials) {
      if (in.enable_nodal_energy[material]) {
        update_nodal_density(s, material);
        interpolate_rho(s, material);
      }
    }
    if (in.enable_adapt) {
      update_quality(in, s);
      update_min_quality(s);
    }
    update_symm_grad_v(s);
    update_h_min(in, s);
    if (in.enable_viscosity) update_h_art(in, s);
    update_material_state(in, s, half_dt, old_p_h);
    for (auto const material : in.materials) {
      if (in.enable_nodal_energy[material]) {
        interpolate_K(s, material);
      }
    }
    update_c(s);
    if (in.enable_viscosity) apply_viscosity(in, s);
    if (in.enable_p_averaging) volume_average_p(s);
    if (last_pc) update_element_dt(s);
    if (last_pc) find_max_stable_dt(s);
    update_a_from_material_state(in, s);
    for (auto const material : in.materials) {
      if (in.enable_nodal_pressure[material]) {
        update_p_h_dot_from_a(in, s, material);
      }
      if (!(in.enable_nodal_pressure[material] || in.enable_nodal_energy[material])) {
        update_p(s, material);
      }
    }
  }
}

static void LGR_NOINLINE velocity_verlet_step(input const& in, state& s) {
  host_vector<device_vector<double, node_index>, material_index> old_p_h(in.materials.size(), s.devpool);
  advance_time(in, s.max_stable_dt, s.next_file_output_time, &s.time, &s.dt);
  update_v(s, s.dt / 2.0, s.v);
  lgr::fill(s.u, vector3<double>(0.0, 0.0, 0.0));
  update_u(s, s.dt);
  update_x(s);
  update_reference(s);
  if (in.enable_J_averaging) volume_average_J(s);
  update_h_min(in, s);
  update_material_state(in, s, s.dt, old_p_h);
  update_c(s);
  update_element_dt(s);
  find_max_stable_dt(s);
  update_a_from_material_state(in, s);
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      update_p_h_dot_from_a(in, s, material);
    } else {
      update_p(s, material);
    }
  }
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

static void LGR_NOINLINE initialize_material_scalar(
    double const scalar,
    state& s,
    material_index const material,
    device_vector<double, point_index>& out) {
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_scalar = out.begin();
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      points_to_scalar[point] = scalar;
    }
  };
  for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE common_initialization(input const& in, state& s) {
  host_vector<device_vector<double, node_index>, material_index> old_p_h(in.materials.size(), s.devpool);
  initialize_V(in, s);
  if (in.enable_viscosity) update_h_art(in, s);
  update_nodal_mass(in, s);
  for (auto const material : in.materials) {
    if (in.enable_nodal_energy[material]) {
      update_nodal_density(s, material);
    }
  }
  initialize_grad_N(in, s);
  if (in.enable_adapt) {
    update_quality(in, s);
    update_min_quality(s);
  }
  update_symm_grad_v(s);
  update_h_min(in, s);
  update_material_state(in, s, 0.0, old_p_h);
  for (auto const material : in.materials) {
    if (in.enable_nodal_energy[material]) {
      interpolate_K(s, material);
    }
  }
  update_c(s);
  if (in.enable_viscosity) apply_viscosity(in, s);
  else lgr::fill(s.nu_art, double(0.0));
  update_element_dt(s);
  find_max_stable_dt(s);
  update_a_from_material_state(in, s);
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      update_p_h_dot_from_a(in, s, material);
    }
    if (!(in.enable_nodal_pressure[material] || in.enable_nodal_energy[material])) {
      update_p(s, material);
    }
  }
}

void run(input const& in) {
  std::cout << std::scientific << std::setprecision(17);
  auto const num_file_outputs = in.num_file_outputs;
  double const file_output_period = num_file_outputs ? in.end_time / num_file_outputs : 0.0;
  state s;
  build_mesh(in, s);
  if (in.x_transform) in.x_transform(&s.x);
  resize_state(in, s);
  assign_element_materials(in, s);
  compute_nodal_materials(in, s);
  collect_node_sets(in, s);
  collect_element_sets(in, s);
  for (auto const material : in.materials) {
    initialize_material_scalar(in.rho0[material], s, material, s.rho);
    if (in.enable_nodal_pressure[material]) {
      lgr::fill(s.p_h[material], double(0.0));
    }
    if (in.enable_nodal_energy[material]) {
      lgr::fill(s.e_h[material], in.e0[material]);
    } else {
      initialize_material_scalar(in.e0[material], s, material, s.e);
    }
  }
  assert(in.initial_v);
  in.initial_v(s.nodes, s.x, &s.v);
  lgr::fill(s.F_total, matrix3x3<double>::identity());
  common_initialization(in, s);
  if (in.enable_adapt) initialize_h_adapt(s);
  file_writer output_file(in.name);
  s.next_file_output_time = num_file_outputs ? 0.0 : in.end_time;
  int file_output_index = 0;
  int file_period_index = 0;
  while (s.time < in.end_time) {
    if (num_file_outputs) {
      if (in.output_to_command_line) {
        std::cout << "outputting file n " << file_output_index << " time " << s.time << "\n";
      }
      output_file(in, file_output_index, s);
      ++file_output_index;
      ++file_period_index;
      s.next_file_output_time = file_period_index * file_output_period;
      s.next_file_output_time = std::min(s.next_file_output_time, in.end_time);
    }
    while (s.time < s.next_file_output_time) {
      if (in.output_to_command_line) {
        std::cout << "step " << s.n << " time " << s.time << " dt " << s.max_stable_dt << "\n";
      }
      time_integrator_step(in, s);
      if (in.enable_adapt && (s.n % 10 == 0)) {
//      output_file(in, file_output_index, s);
//      ++file_output_index;
        for (int i = 0; i < 4; ++i) {
          adapt(in, s);
          resize_state(in, s);
          collect_element_sets(in, s);
          collect_node_sets(in, s);
          common_initialization(in, s);
//        output_file(in, file_output_index, s);
//        ++file_output_index;
        }
      }
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

