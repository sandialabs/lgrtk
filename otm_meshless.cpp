#include <bitset>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_math.hpp>
#include <hpc_vector3.hpp>
#include <lgr_domain.hpp>
#include <lgr_element_specific_inline.hpp>
#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_physics_util.hpp>
#include <lgr_state.hpp>
#include <otm_tetrahedron_util.hpp>
#include <otm_distance.hpp>
#include <otm_materials.hpp>
#include <otm_meshless.hpp>
#include <otm_search_util.hpp>
#include <otm_search.hpp>
#include <otm_tet2meshless.hpp>
#include <otm_util.hpp>
#include <j2/hardening.hpp>
#include <otm_vtk.hpp>

namespace lgr {

void otm_mark_boundary_domains(input const& in, state& s)
{
  s.boundaries = in.boundaries;
  for (auto const boundary : s.boundaries) {
    auto const& domain = in.domains[boundary];
    domain->mark(s.x, boundary, &s.nodal_materials);
  }
}

void otm_initialize_displacement(state& s)
{
  auto const x = std::acos(-1.0);
  auto const y = std::exp(1.0);
  auto const z = std::sqrt(2.0);
  hpc::fill(hpc::device_policy(), s.u, hpc::position<double>(x, y, z));
}

void otm_uniaxial_patch_test_ics(state& s)
{
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const top_vel = hpc::speed<double>(10.0);
  auto functor = [=] HPC_DEVICE (point_index const node) {
    auto const x = nodes_to_x[node].load();
    auto const vz = top_vel * x(2);
    nodes_to_u[node] = hpc::position<double>(0.0, 0.0, 0.0);
    nodes_to_v[node] = hpc::velocity<double>(0.0, 0.0, vz);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_initialize_point_volume(state& s) {
  auto const num_points = s.points.size();
  s.V.resize(num_points);
  auto const points_per_element = s.points_in_element.size();
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_to_V = s.V.begin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = points_to_point_nodes[point];
    hpc::array<hpc::position<double>, 4> x;
    assert(point_nodes.size() == 4);
    for (auto i = 0; i < 4; ++i)
    {
      auto const node = point_nodes_to_nodes[point_nodes[i]];
      x[i] = nodes_to_x[node].load();
    }
    auto const volume = tetrahedron_volume(x);
    assert(volume > 0.0);
    point_to_V[point] = volume / points_per_element;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_update_shape_functions(state& s) {
  hpc::adimensional<double> gamma(1.5);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xp = s.xp.cbegin();
  auto const points_to_h = s.h_otm.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = points_to_point_nodes[point];
    auto const h = points_to_h[point];
    auto const beta = gamma / h / h;
    auto const xp = points_to_xp[point].load();
    // Newton's algorithm
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    auto const eps = 1024 * hpc::machine_epsilon<double>();
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto const max_iter = 16;
    for (auto iter = 0; iter < max_iter; ++iter) {
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto point_node : point_nodes) {
        auto const node = point_nodes_to_nodes[point_node];
        auto const xn = nodes_to_x[node].load();
        auto const r = xn - xp;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const error = hpc::norm(dmu) / hpc::norm(mu);
      converged = error <= eps;
      if (converged == true) {
        J = dRdmu;
        break;
      }
    }
    assert(converged == true);
    auto Z = 0.0;
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xp;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      point_nodes_to_N[point_node] = boltzmann_factor;
    }
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xp;
      auto const Jinvr = hpc::solve_full_pivot(J, r);
      auto const NZ = point_nodes_to_N[point_node];
      point_nodes_to_N[point_node] = NZ / Z;
      point_nodes_to_grad_N[point_node] = -NZ * Jinvr;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}
inline void otm_assemble_internal_force(state& s)
{
  auto const points_to_sigma = s.sigma_full.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const sigma = points_to_sigma[point].load();
      auto const V = points_to_V[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const grad_N = point_nodes_to_grad_N[point_node].load();
      auto const f = -(sigma * grad_N) * V;
      node_f = node_f + f;
    }
    auto const f_old = nodes_to_f[node].load();
    auto const f_new = f_old + node_f;
    nodes_to_f[node] = f_new;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

inline void otm_assemble_external_force(state& s)
{
  auto const points_to_body_acce = s.b.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_N = s.N.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const body_acce = points_to_body_acce[point].load();
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const N = point_nodes_to_N[point_node];
      auto const m = N * rho * V;
      auto const f = m * body_acce;
      node_f = node_f + f;
    }
    auto const f_old = nodes_to_f[node].load();
    auto const f_new = f_old + node_f;
    nodes_to_f[node] = f_new;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_nodal_force(state& s) {
  hpc::fill(hpc::device_policy(), s.f, hpc::force<double>::zero());
  otm_assemble_internal_force(s);
  otm_assemble_external_force(s);
}

void otm_update_nodal_mass(state& s) {
  auto const node_to_mass = s.mass.begin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  hpc::fill(hpc::device_policy(), s.mass, 0.0);
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_m = 0.0;
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const N = point_nodes_to_N[point_node];
      auto const m = N * rho * V;
      node_m += m;
    }
    node_to_mass[node] += node_m;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_reference(state& s) {
  otm_update_nodal_position(s);
  otm_update_point_position(s);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const nodes_to_u = s.u.cbegin();
  auto const points_to_F_total = s.F_total.begin();
  auto const points_to_V = s.V.begin();
  auto const points_to_rho = s.rho.begin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const point_nodes = points_to_point_nodes[point];
    auto F_incr = hpc::deformation_gradient<double>::identity();
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const u = nodes_to_u[node].load();
      auto const old_grad_N = point_nodes_to_grad_N[point_node].load();
      F_incr = F_incr + outer_product(u, old_grad_N);
    }
    // TODO: Verify this is also true for OTM
    auto const F_inverse_transpose = transpose(inverse(F_incr));
    for (auto const point_node : point_nodes) {
      auto const old_grad_N = point_nodes_to_grad_N[point_node].load();
      auto const new_grad_N = F_inverse_transpose * old_grad_N;
      point_nodes_to_grad_N[point_node] = new_grad_N;
    }
    auto const old_F_total = points_to_F_total[point].load();
    auto const new_F_total = F_incr * old_F_total;
    points_to_F_total[point] = new_F_total;
    auto const J = determinant(F_incr);
    assert(J > 0.0);
    auto const old_V = points_to_V[point];
    auto const new_V = J * old_V;
    assert(new_V > 0.0);
    points_to_V[point] = new_V;
    auto const old_rho = points_to_rho[point];
    auto const new_rho = old_rho / J;
    points_to_rho[point] = new_rho;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_update_material_state(input const& in, state& s, material_index const material)
{
  auto const dt = s.dt;
  auto const points_to_F_total = s.F_total.cbegin();
  auto const points_to_sigma = s.sigma_full.begin();
  auto const points_to_K = s.K.begin();
  auto const points_to_G = s.G.begin();
  auto const points_to_W = s.potential_density.begin();
  auto const points_to_Fp = s.Fp_total.begin();
  auto const points_to_ep = s.ep.begin();
  auto const K = in.K0[material];
  auto const G = in.G0[material];
  auto const Y0 = in.Y0[material];
  auto const n = in.n[material];
  auto const eps0 = in.eps0[material];
  auto const Svis0 = in.Svis0[material];
  auto const m = in.m[material];
  auto const eps_dot0 = in.eps_dot0[material];
  auto const is_neo_hookean = in.enable_neo_Hookean[material];
  auto const is_variational_J2 = in.enable_variational_J2[material];
  auto functor = [=] HPC_DEVICE (point_index const point) {
      auto const F = points_to_F_total[point].load();
      auto sigma = hpc::stress<double>::zero();
      auto Keff = hpc::pressure<double>(0.0);
      auto Geff = hpc::pressure<double>(0.0);
      auto W = hpc::energy_density<double>(0.0);
      if (is_neo_hookean == true) {
        neo_Hookean_point(F, K, G, sigma, Keff, Geff, W);
      }
      if (is_variational_J2 == true) {
        j2::Properties props{K, G, Y0, n, eps0, Svis0, m, eps_dot0};
        auto Fp = points_to_Fp[point].load();
        auto ep = points_to_ep[point];
        variational_J2_point(F, props, dt, sigma, Keff, Geff, W, Fp, ep);
      }
      points_to_sigma[point] = sigma;
      points_to_K[point] = Keff;
      points_to_G[point] = Geff;
      points_to_W[point] = W;
  };
  hpc::for_each(hpc::host_policy(), s.points, functor);
}

void otm_update_nodal_momentum(state& s) {
  auto const nodes_to_lm = s.lm.begin();
  auto const nodes_to_v = s.v.cbegin();
  auto const nodes_to_mass = s.mass.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const v = nodes_to_v[node].load();
    auto lm = nodes_to_lm[node];
    lm = m * v;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_nodal_position(state& s) {
  auto const dt = s.dt;
  auto const dt_old = s.dt_old;
  auto const dt_avg = 0.5 * (dt + dt_old);
  auto const nodes_to_mass = s.mass.cbegin();
  auto const nodes_to_f = s.f.cbegin();
  auto const nodes_to_lm = s.lm.cbegin();
  auto const nodes_to_x = s.x.begin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const nodes_to_domain = s.nodal_materials.cbegin();
  auto const boundary_indices = s.boundaries;
  auto const boundary_to_prescribed_v = s.prescribed_v.cbegin();
  auto const boundary_to_prescribed_dof = s.prescribed_dof.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const lm = nodes_to_lm[node].load();
    auto const f = nodes_to_f[node].load();
    auto disp = (dt / m) * (lm + dt_avg * f);
    auto const domain_set = nodes_to_domain[node];
    for (auto boundary_index : boundary_indices) {
      material_set const boundary(boundary_index);
      if (domain_set.contains(boundary)) {
        auto const v = boundary_to_prescribed_v[boundary_index].load();
        auto const dof = boundary_to_prescribed_dof[boundary_index].load();
        if (dof(0) == 1) {disp(0) = v(0) * dt;}
        if (dof(1) == 1) {disp(1) = v(1) * dt;}
        if (dof(2) == 1) {disp(2) = v(2) * dt;}
      }
    }
    auto const velo = disp / dt;
    auto x_old = nodes_to_x[node].load();
    nodes_to_u[node] = disp;
    auto x_new = x_old + disp;
    nodes_to_x[node] = x_new;
    nodes_to_v[node] = velo;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_point_position(state& s)
{
  auto const point_nodes_to_N = s.N.cbegin();
  auto const nodes_to_u = s.u.cbegin();
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const points_to_xp = s.xp.begin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = points_to_point_nodes[point];
    auto up = hpc::position<double>::zero();
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const u = nodes_to_u[node].load();
      auto const N = point_nodes_to_N[point_node];
      up += N * u;
    }
    auto xp = points_to_xp[point].load();
    xp += up;
    points_to_xp[point] = xp;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_allocate_state(input const& in, state& s) {
  auto const num_points = s.points.size();
  auto const num_nodes = s.nodes.size();
  auto const num_elements = s.elements.size();
  auto const num_materials = in.materials.size();
  auto const num_boundaries = in.boundaries.size();
  auto const support_size = s.points_to_point_nodes.size();
  s.u.resize(num_nodes);
  s.v.resize(num_nodes);
  s.V.resize(num_points);
  s.grad_N.resize(num_points * support_size);
  s.N.resize(num_points * support_size);
  s.F_total.resize(num_points);
  s.sigma_full.resize(num_points);
  s.symm_grad_v.resize(num_points);
  s.K.resize(num_points);
  s.G.resize(num_points);
  s.potential_density.resize(num_points);
  s.c.resize(num_points);
  s.Fp_total.resize(num_points);
  s.ep.resize(num_points);
  s.element_f.resize(num_points * s.nodes_in_element.size());
  s.lm.resize(num_nodes);
  s.f.resize(num_nodes);
  s.rho.resize(num_points);
  s.b.resize(num_points);
  s.nearest_neighbor_dist.resize(num_points);
  s.material_mass.resize(num_materials);
  for (auto& mm : s.material_mass) mm.resize(num_nodes);
  s.mass.resize(num_nodes);
  s.a.resize(num_nodes);
  s.h_min.resize(num_elements);
  if (in.enable_viscosity) {
    s.h_art.resize(num_elements);
  }
  s.nu_art.resize(num_points);
  s.element_dt.resize(num_points);
  s.p_h.resize(num_materials);
  s.p_h_dot.resize(num_materials);
  s.e_h.resize(num_materials);
  s.e_h_dot.resize(num_materials);
  s.rho_h.resize(num_materials);
  s.K_h.resize(num_materials);
  s.dp_de_h.resize(num_materials);
  s.temp.resize(num_materials);
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      s.p_h[material].resize(num_nodes);
      s.p_h_dot[material].resize(num_nodes);
      s.v_prime.resize(num_points);
      s.W.resize(num_points * s.nodes_in_element.size());
    }
    if (in.enable_nodal_energy[material]) {
      s.p_h[material].resize(num_nodes);
      s.e_h[material].resize(num_nodes);
      s.e_h_dot[material].resize(num_nodes);
      s.rho_h[material].resize(num_nodes);
      s.K_h[material].resize(num_nodes);
      s.q.resize(num_points);
      s.W.resize(num_points * s.nodes_in_element.size());
      s.dp_de_h[material].resize(num_nodes);
      if (in.enable_p_prime[material]) {
        s.p_prime.resize(num_points);
      }
    }
  }
  s.material.resize(num_elements);
  if (in.enable_adapt) {
    s.quality.resize(num_elements);
    s.h_adapt.resize(num_nodes);
  }
  s.nodal_materials.resize(num_nodes);
  s.prescribed_v.resize(num_boundaries + num_materials);
  s.prescribed_dof.resize(num_boundaries + num_materials);
}

void otm_initialize(input& in, state& s)
{
  otm_mark_boundary_domains(in, s);
  otm_initialize_point_volume(s);
  otm_update_shape_functions(s);
  for (auto material : in.materials) {
    otm_update_material_state(in, s, material);
  }
  otm_update_nodal_mass(s);
}

HPC_NOINLINE inline void compute_min_neighbor_dist(state& s) {
  search_util::point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 1);
  search_util::compute_point_neighbor_squared_distances(s, n, s.nearest_neighbor_dist);
  assert( s.nearest_neighbor_dist.size() == s.points.size() );
  auto const nearest_neighbors = n.entities_to_neighbors.cbegin();
  auto const neighbor_distances = s.nearest_neighbor_dist.begin();
  auto dist_func = [=] HPC_DEVICE (point_index const point) {
                     auto const neighbor = nearest_neighbors[point];
                     if (point < neighbor)
                     {
                       auto const dist = std::sqrt(neighbor_distances[point]);
                       neighbor_distances[point] = dist;
                       neighbor_distances[neighbor] = dist;
                     }
                   };
  hpc::for_each(hpc::device_policy(), s.points, dist_func);
}

HPC_NOINLINE inline void update_point_dt(state& s) {
  compute_min_neighbor_dist(s);
  auto const points_to_c = s.c.cbegin();
  auto const points_to_dt = s.element_dt.begin();
  auto const points_to_neighbor_dist = s.nearest_neighbor_dist.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
                   auto const h_min = points_to_neighbor_dist[point];
                   auto const c = points_to_c[point];
                   auto const dt = h_min / c;
                   assert(dt > 0.0);
                   points_to_dt[point] = dt;
                 };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_update_time_step(state& s)
{
  update_c(s);
  update_point_dt(s);
  find_max_stable_dt(s);
}

void otm_update_time(input const& in, state& s)
{
  if (in.use_constant_dt == true) {
    s.max_stable_dt = s.dt = in.constant_dt;
  } else {
    otm_update_time_step(s);
  }
  auto const old_time = s.time;
  auto const new_time = std::min(s.next_file_output_time, old_time + (s.max_stable_dt * in.CFL));
  s.time = new_time;
  s.dt_old = s.dt;
  s.dt = new_time - old_time;
}

void otm_time_integrator_step(input const& in, state& s)
{
  otm_update_nodal_mass(s);
  otm_update_nodal_momentum(s);
  otm_update_nodal_force(s);
  otm_update_reference(s);
  for (auto material : in.materials) {
    otm_update_material_state(in, s, material);
  }
  otm_update_shape_functions(s);
  otm_update_time(in, s);
}

void otm_debug_output(state& s)
{
  HPC_DUMP("TIME : " << s.time << '\n');
  auto const nodes_to_x = s.x.cbegin();
  auto print_x = [=] HPC_DEVICE (lgr::node_index const node) {
    auto const x = nodes_to_x[node].load();
    HPC_DUMP("node: " << node << ", x:" << x);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_x);

  auto const nodes_to_v = s.v.cbegin();
  auto print_v = [=] HPC_DEVICE (lgr::node_index const node) {
    auto const v = nodes_to_v[node].load();
    HPC_DUMP("node: " << node << ", v:" << v);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_v);

  auto const nodes_to_u = s.u.cbegin();
  auto print_u = [=] HPC_DEVICE (lgr::node_index const node) {
    auto const u = nodes_to_u[node].load();
    HPC_DUMP("node: " << node << ", u:" << u);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_u);

  auto const points_to_xp = s.xp.cbegin();
  auto print_xp = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const xp = points_to_xp[point].load();
    HPC_DUMP("point: " << point << ", xp:" << xp);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_xp);

  auto const points_to_F = s.F_total.cbegin();
  auto print_F = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const F = points_to_F[point].load();
    HPC_DUMP("point: " << point << ", F:\n" << F);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_F);

  auto const points_to_sigma = s.sigma_full.cbegin();
  auto const points_to_K = s.K.cbegin();
  auto const points_to_G = s.G.cbegin();
  auto print_sigma = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const sigma = points_to_sigma[point].load();
    auto const K = points_to_K[point];
    auto const G = points_to_G[point];
    HPC_DUMP("point: " << point << ", K: " << K << ", G: " << G << ", sigma:\n" << sigma);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_sigma);
}

void otm_run(input const& in, state& s)
{
  lgr::search::initialize_otm_search();
  lgr::otm_file_writer output_file(in.name);
  std::cout << std::scientific << std::setprecision(17);
  auto const num_file_output_periods = in.num_file_output_periods;
  auto const file_output_period = num_file_output_periods != 0 ? in.end_time / double(num_file_output_periods) : hpc::time<double>(0.0);
  auto file_output_index = 0;
  s.next_file_output_time = num_file_output_periods != 0 ? 0.0 : in.end_time;
  while (s.time < in.end_time) {
    if (num_file_output_periods != 0) {
      if (in.output_to_command_line == true) {
        std::cout << "outputting file " << file_output_index << " time " << double(s.time) << "\n";
      }
      if (in.debug_output == true) {
        otm_debug_output(s);
      }
      output_file.capture(s);
      output_file.write(file_output_index);
      ++file_output_index;
      s.next_file_output_time = double(file_output_index) * file_output_period;
      s.next_file_output_time = std::min(s.next_file_output_time, in.end_time);
    }
    while (s.time < s.next_file_output_time) {
      if (in.output_to_command_line == true) {
        std::cout << "step " << s.n << " time " << double(s.time) << " dt " << double(s.max_stable_dt) << "\n";
      }
      otm_time_integrator_step(in, s);
      if (in.enable_adapt && (s.n % 10 == 0)) {
      }
      ++s.n;
    }
  }
  lgr::search::finalize_otm_search();
}

void otm_j2_uniaxial_patch_test()
{
  material_index num_materials(1);
  material_index num_boundaries(4);
  input in(num_materials, num_boundaries);
  state s;
  std::string const filename{"cube.g"};
  auto const points_in_element = 1;
  s.points_in_element.resize(point_in_element_index(points_in_element));
  in.otm_material_points_to_add_per_element = points_in_element;
  lgr::tet_nodes_to_points point_interpolator(points_in_element);
  in.xp_transform = std::ref(point_interpolator);
  auto const err_code = lgr::read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  in.name = "OTM";
  in.end_time = 0.001;
  in.num_file_output_periods = 100;
  in.debug_output = true;
  auto const rho = hpc::density<double>(7.8e+03);
  auto const nu = hpc::adimensional<double>(0.00);
  auto const E = hpc::pressure<double>(200.0e09);
  auto const K = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0 = hpc::pressure<double>(1.0e+64);
  auto const n = hpc::adimensional<double>(4.0);
  auto const eps0 = hpc::strain<double>(1.0e-02);
  auto const Svis0 = hpc::pressure<double>(Y0);
  auto const m = hpc::adimensional<double>(2.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
  constexpr material_index body(0);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index z_max(4);
  in.materials = hpc::counting_range<material_index>(1);
  in.enable_variational_J2[body] = true;
  in.rho0[body] = rho;
  in.K0[body] = K;
  in.G0[body] = G;
  in.Y0[body] = Y0;
  in.n[body] = n;
  in.eps0[body] = eps0;
  in.Svis0[body] = Svis0;
  in.m[body] = m;
  in.eps_dot0[body] = eps_dot0;
  in.CFL = 1.0;
  in.use_constant_dt = true;
  in.constant_dt = hpc::time<double>(1.0e-06);
  auto const tol = hpc::length<double>(1.0e-04);
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  in.domains[body] = std::make_unique<clipped_domain<all_space>>(all_space{});
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, tol);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, tol);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, 0.0}, tol);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, 1.0}, tol);
  lgr::convert_tet_mesh_to_meshless(in, s);
  s.dt = in.constant_dt;
  s.dt_old = in.constant_dt;
  s.time = hpc::time<double>(0.0);
  s.max_stable_dt = in.constant_dt;
  otm_allocate_state(in, s);
  s.prescribed_v[body] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[body] = hpc::vector3<int>(0, 0, 0);
  s.prescribed_v[x_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[x_min] = hpc::vector3<int>(1, 0, 0);
  s.prescribed_v[y_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[y_min] = hpc::vector3<int>(0, 1, 0);
  s.prescribed_v[z_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[z_min] = hpc::vector3<int>(0, 0, 1);
  s.prescribed_v[z_max] = hpc::velocity<double>(0.0, 0.0, 10.0);
  s.prescribed_dof[z_max] = hpc::vector3<int>(0, 0, 1);
  auto const I = hpc::deformation_gradient<double>::identity();
  auto const ep0 = hpc::strain<double>(0.0);
  hpc::fill(hpc::device_policy(), s.rho, rho);
  hpc::fill(hpc::device_policy(), s.F_total, I);
  hpc::fill(hpc::device_policy(), s.Fp_total, I);
  hpc::fill(hpc::device_policy(), s.ep, ep0);
  otm_uniaxial_patch_test_ics(s);
  otm_initialize(in, s);
  otm_run(in, s);
}

} // namespace lgr
