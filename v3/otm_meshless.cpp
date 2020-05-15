#include <bitset>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_math.hpp>
#include <hpc_transform_reduce.hpp>
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
#include <otm_adapt.hpp>
#include <otm_distance_util.hpp>

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


void otm_initialize_point_volume_1(state& s) {
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

void otm_initialize_point_volume(state& s) {
  auto const num_points = s.points.size();
  s.V.resize(num_points);
  auto const points_per_element = s.points_in_element.size();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const points_in_element = hpc::make_counting_range(points_per_element);
  auto const elements_to_points = s.elements * points_in_element;
  auto func = [=] HPC_DEVICE (element_index const element)
  {
    auto const cur_elem_points = elements_to_points[element];
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::position<double>, 4> x;
    assert(element_nodes.size() == 4);
    for (auto i = 0; i < 4; ++i)
    {
      auto const node = element_nodes_to_nodes[element_nodes[i]];
      x[i] = nodes_to_x[node].load();
    }
    auto const volume = tetrahedron_volume(x);
    assert(volume > 0.0);
    for (auto element_point : points_in_element)
    {
      auto const point = cur_elem_points[element_point];
      point_to_V[point] = volume / points_per_element;
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, func);
}

void otm_update_shape_functions(state& s) {
  auto beta = s.otm_beta;
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xp = s.xp.cbegin();
  //otm_update_h(s);
  //auto const points_to_h = s.h_otm.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const eps = s.maxent_tolerance;
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = points_to_point_nodes[point];
    //auto const h = points_to_h[point];
    //auto const beta = gamma / (h * h);
    auto const xp = points_to_xp[point].load();
    // Newton's algorithm
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto iter = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
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
      converged = (error <= eps);
      J = dRdmu;
      ++iter;
    }
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

inline void otm_assemble_contact_force(state& s)
{
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_mass = s.mass.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const penalty_coeff = s.contact_penalty_coeff;
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const x = nodes_to_x[node].load();
    auto const m = nodes_to_mass[node];
    auto const z = x(2);
    if (z > 0.0) {
      node_f(2) = -penalty_coeff * m * z;
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
  if (s.use_penalty_contact == true) {
    otm_assemble_contact_force(s);
  }
}

void otm_update_nodal_mass(state& s) {
  auto const nodes_to_mass = s.mass.begin();
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
    nodes_to_mass[node] += node_m;
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
      F_incr += outer_product(u, old_grad_N);
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
        points_to_Fp[point] = Fp;
        points_to_ep[point] = ep;
      }
      points_to_sigma[point] = sigma;
      points_to_K[point] = Keff;
      points_to_G[point] = Geff;
      points_to_W[point] = W;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
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

inline void otm_enforce_boundary_conditions(state &s)
{
  auto const dt = s.dt;
  auto const boundary_indices = s.boundaries;
  auto const boundary_to_prescribed_v = s.prescribed_v.cbegin();
  auto const boundary_to_prescribed_dof = s.prescribed_dof.cbegin();
  auto const nodes_to_u = s.u.begin();
  for (auto boundary_index : boundary_indices)
  {
    auto const v = boundary_to_prescribed_v[boundary_index].load();
    auto const dof = boundary_to_prescribed_dof[boundary_index].load();
    auto functor = [=] HPC_DEVICE (node_index const node)
    {
      auto disp = nodes_to_u[node].load();
      if (dof(0) == 1)
      { disp(0) = v(0) * dt;}
      if (dof(1) == 1)
      { disp(1) = v(1) * dt;}
      if (dof(2) == 1)
      { disp(2) = v(2) * dt;}
      nodes_to_u[node] = disp;
    };
    hpc::for_each(hpc::device_policy(), s.node_sets[boundary_index], functor);
  }
}

inline void otm_enforce_contact_constraints(state &s)
{
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto functor = [=] HPC_DEVICE (node_index const node)
  {
    auto x = nodes_to_x[node].load();
    auto u = nodes_to_u[node].load();
    auto z = x(2);
    if (z >= 0.0) {
      u(2) = 0.0;
    }
    nodes_to_u[node] = u;
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
  auto disp_functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const lm = nodes_to_lm[node].load();
    auto const f = nodes_to_f[node].load();
    auto disp = (dt / m) * (lm + dt_avg * f);
    nodes_to_u[node] = disp;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, disp_functor);

  otm_enforce_boundary_conditions(s);
  if (s.use_displacement_contact == true) {
    otm_enforce_contact_constraints(s);
  }

  auto coord_vel_update_functor = [=] HPC_DEVICE (node_index const node) {
    auto disp = nodes_to_u[node].load();
    auto const velo = disp / dt;
    nodes_to_v[node] = velo;
    auto x_old = nodes_to_x[node].load();
    auto x_new = x_old + disp;
    nodes_to_x[node] = x_new;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, coord_vel_update_functor);
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
  s.u.resize(num_nodes);
  s.v.resize(num_nodes);
  s.V.resize(num_points);
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    return points_to_point_nodes[point].size();
  };
  auto const total_support_size = hpc::transform_reduce(hpc::device_policy(), s.points, 0, hpc::plus<int>(), functor);
  s.grad_N.resize(total_support_size);
  s.N.resize(total_support_size);
  s.F_total.resize(num_points);
  s.sigma_full.resize(num_points);
  s.symm_grad_v.resize(num_points);
  s.K.resize(num_points);
  s.G.resize(num_points);
  s.potential_density.resize(num_points);
  s.c.resize(num_points);
  s.Fp_total.resize(num_points);
  s.ep.resize(num_points);
  s.lm.resize(num_nodes);
  s.f.resize(num_nodes);
  s.rho.resize(num_points);
  s.b.resize(num_points);
  s.element_dt.resize(num_points);
  if (!in.use_constant_dt || in.enable_adapt)
  {
    s.nearest_point_neighbor_dist.resize(num_points);
    s.nearest_point_neighbor.resize(num_points);
  }
  if (in.enable_adapt)
  {
    s.nearest_node_neighbor_dist.resize(num_nodes);
    s.nearest_node_neighbor.resize(num_nodes);
  }
  s.mass.resize(num_nodes);
  s.a.resize(num_nodes);
  s.material.resize(num_elements);
  s.nodal_materials.resize(num_nodes);
  s.prescribed_v.resize(num_boundaries + num_materials);
  s.prescribed_dof.resize(num_boundaries + num_materials);
}

void otm_initialize(input& in, state& s)
{
  otm_mark_boundary_domains(in, s);
  collect_node_sets(in, s);
  otm_initialize_point_volume(s);
  otm_set_beta(in.otm_gamma, s);
  otm_update_shape_functions(s);
  for (auto material : in.materials) {
    otm_update_material_state(in, s, material);
  }
  otm_update_nodal_mass(s);
}

HPC_NOINLINE inline hpc::energy<double> compute_kinetic_energy(const state& s) {
  auto const nodes_to_lm = s.lm.cbegin();
  auto const nodes_to_mass = s.mass.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const lm = nodes_to_lm[node].load();
    return 0.5*hpc::norm_squared(lm)/m;
  };
  hpc::energy<double> init(0);
  auto const T = hpc::transform_reduce(hpc::device_policy(), s.nodes, init, hpc::plus<hpc::energy<double> >(), functor);
  return T;
}

HPC_NOINLINE inline hpc::energy<double> compute_free_energy(const state& s) {
  auto const points_to_potential_density = s.potential_density.cbegin();
  auto const points_to_volume = s.V.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const psi = points_to_potential_density[point];
    auto const dV = points_to_volume[point];
    return psi*dV;
  };
  hpc::energy<double> init(0);
  auto const F = hpc::transform_reduce(hpc::device_policy(), s.points, init, hpc::plus<hpc::energy<double> >(), functor);
  return F;
}

HPC_NOINLINE inline void update_point_dt(state& s) {
  auto const points_to_c = s.c.cbegin();
  auto const points_to_dt = s.element_dt.begin();
  auto const points_to_neighbor_dist = s.nearest_point_neighbor_dist.cbegin();
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

void
otm_update_neighbor_distances(input const& in, state& s)
{
  if (!in.use_constant_dt || in.enable_adapt)
  {
    otm_update_nearest_point_neighbor_distances(s);
  }
  if (in.enable_adapt)
  {
    otm_update_nearest_node_neighbor_distances(s);
    otm_update_min_nearest_neighbor_distances(s);
  }
}

void otm_update_time(input const& in, state& s)
{
  s.dt_old = s.dt;
  otm_update_neighbor_distances(in, s);
  if (in.use_constant_dt == true)
  {
    auto const fp_n = static_cast<double>(s.n);
    auto const fp_np1 = static_cast<double>(s.n + 1);
    auto const fp_N = static_cast<double>(s.num_time_steps);
    auto const old_time = fp_n / fp_N * in.end_time;
    auto const new_time = fp_np1 / fp_N * in.end_time;
    s.max_stable_dt = s.dt = new_time - old_time;
    s.time = new_time;
  } else
  {
    otm_update_time_step(s);
    s.dt = s.max_stable_dt * in.CFL;
    s.time += s.dt;
  }
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
  otm_update_neighbor_distances(in, s);
}

void otm_set_beta(double gamma, state& s)
{
  s.nearest_point_neighbor.resize(s.points.size());
  s.nearest_point_neighbor_dist.resize(s.points.size());
  otm_update_nearest_point_neighbor_distances(s);
  hpc::length<double> init{0};
  auto h = hpc::transform_reduce(hpc::device_policy(), s.nearest_point_neighbor_dist,
                                 init, hpc::maximum<hpc::length<double>>(),
                                 hpc::identity<hpc::length<double>>());
  h *= 4.0;
  s.otm_beta = gamma / (h * h);
}

void otm_run(input const& in, state& s)
{
  lgr::otm_file_writer output_file(in.name);
  std::cout << std::scientific << std::setprecision(17);
  auto const num_file_output_periods = in.num_file_output_periods;
  auto const file_output_period = num_file_output_periods != 0 ? in.end_time / double(num_file_output_periods) : hpc::time<double>(0.0);
  auto file_output_index = 0;
  if (in.use_constant_dt == true) {
    auto const num_time_steps_between_output = static_cast<int>(std::round(file_output_period / in.constant_dt));
    for (s.n = 0; s.n <= s.num_time_steps; ++s.n) {
      if (in.output_to_command_line == true) {
        auto const KE = compute_kinetic_energy(s);
        auto const SE = compute_free_energy(s);
        std::cout << "step " << s.n << " time " << double(s.time) << " dt " << double(s.dt);
        std::cout << " kinetic energy " << KE << " free energy " << SE << "\n";
      }
      auto const do_output = in.do_output == true && (s.n % num_time_steps_between_output == 0);
      if (do_output == true) {
        if (in.output_to_command_line == true) {
          std::cout << "outputting file " << file_output_index << " time " << double(s.time) << "\n";
        }
        output_file.capture(s);
        if (in.debug_output == true) {
          output_file.to_console();
        }
        output_file.write(file_output_index);
        ++file_output_index;
      }
      if (s.n >= s.num_time_steps) continue;
      otm_time_integrator_step(in, s);
      if (in.enable_adapt && (s.n % 10 == 0)) {
        for (int i=0; i<4; ++i)
        {
          otm_adapt(in, s);
          otm_update_min_nearest_neighbor_distances(s);
          otm_allocate_state(in, s);
        }
      }
    }
  } else {
    while (s.time <= in.end_time) {
      if (in.output_to_command_line == true) {
        auto const KE = compute_kinetic_energy(s);
        auto const SE = compute_free_energy(s);
        std::cout << "step " << s.n << " time " << double(s.time) << " dt " << double(s.dt);
        std::cout << " kinetic energy " << KE << " free energy " << SE << "\n";
      }
      auto const do_output = in.do_output == true && (s.time == hpc::time<double>(0.0) || (num_file_output_periods != 0 && s.time >= s.next_file_output_time));
      if (do_output == true) {
        if (in.output_to_command_line == true) {
          std::cout << "outputting file " << file_output_index << " time " << double(s.time) << "\n";
        }
        output_file.capture(s);
        if (in.debug_output == true) {
          output_file.to_console();
        }
        output_file.write(file_output_index);
        ++file_output_index;
        s.next_file_output_time = double(file_output_index) * file_output_period;
      }
      otm_time_integrator_step(in, s);
      if (in.enable_adapt && (s.n % 10 == 0)) {
      }
      ++s.n;
    }
  }
}
} // namespace lgr
