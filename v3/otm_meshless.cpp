#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_vector3.hpp>
#include <otm_exodus.hpp>
#include <otm_input.hpp>
#include <otm_materials.hpp>
#include <otm_meshless.hpp>
#include <otm_state.hpp>
#include <otm_tet2meshless.hpp>
#include <j2/hardening.hpp>

namespace lgr {

void otm_initialize_u(state& s)
{
  auto const x = std::acos(-1.0);
  auto const y = std::exp(1.0);
  auto const z = std::sqrt(2.0);
  auto const nodes_to_u = s.u.begin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    nodes_to_u[node] = hpc::position<double>(x, y, z);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_initialize_F(state& s)
{
  auto const points_to_F = s.F_total.begin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    points_to_F[point] = hpc::deformation_gradient<double>::identity();
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_initialize_grad_val_N(state& s) {
  hpc::dimensionless<double> gamma(1.5);
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
      point_nodes_to_grad_N[point_node] = NZ * Jinvr;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

inline void otm_assemble_internal_force(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
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
    auto node_to_f = nodes_to_f[node].load();
    node_to_f += node_f;
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
    auto node_to_f = nodes_to_f[node].load();
    node_to_f += node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_nodal_force(state& s) {
  auto const nodes_to_f = s.f.begin();
  auto node_f = hpc::force<double>::zero();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_to_f = nodes_to_f[node].load();
    node_to_f = node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
  otm_assemble_internal_force(s);
  otm_assemble_external_force(s);
}

void otm_lump_nodal_mass(state& s) {
  auto const node_to_mass = s.mass.begin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto zero_mass = [=] HPC_DEVICE (node_index const node) {
    node_to_mass[node] = 0.0;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, zero_mass);
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
  auto const points_to_sigma = s.sigma.begin();
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
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_nodal_linear_momentum(state& s) {
  auto const nodes_to_lm = s.lm.begin();
  auto const nodes_to_v = s.v.cbegin();
  auto const nodes_to_mass = s.mass.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const v = nodes_to_v[node].load();
    auto lm = nodes_to_lm[node].load();
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
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const m = nodes_to_mass[node];
    auto const lm = nodes_to_lm[node].load();
    auto const f = nodes_to_f[node].load();
    auto const disp = dt / m * (lm + dt_avg * f);
    auto x = nodes_to_x[node].load();
    auto u = nodes_to_u[node].load();
    x += disp;
    u = disp;
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
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_initialize_state(input const& in, state& s) {
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.V.resize(s.points.size());
  s.grad_N.resize(s.points.size() * s.points_to_point_nodes.size());
  s.N.resize(s.points.size() * s.points_to_point_nodes.size());
  s.F_total.resize(s.points.size());
  s.sigma.resize(s.points.size());
  s.symm_grad_v.resize(s.points.size());
  s.K.resize(s.points.size());
  s.G.resize(s.points.size());
  s.c.resize(s.points.size());
  s.element_f.resize(s.points.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.points.size());
  s.material_mass.resize(in.materials.size());
  for (auto& mm : s.material_mass) mm.resize(s.nodes.size());
  s.mass.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.points.size());
  s.element_dt.resize(s.points.size());
  s.p_h.resize(in.materials.size());
  s.p_h_dot.resize(in.materials.size());
  s.e_h.resize(in.materials.size());
  s.e_h_dot.resize(in.materials.size());
  s.rho_h.resize(in.materials.size());
  s.K_h.resize(in.materials.size());
  s.dp_de_h.resize(in.materials.size());
  s.temp.resize(in.materials.size());
  s.ep_h.resize(in.materials.size());
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.p_h_dot[material].resize(s.nodes.size());
      s.v_prime.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
    }
    if (in.enable_nodal_energy[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.e_h[material].resize(s.nodes.size());
      s.e_h_dot[material].resize(s.nodes.size());
      s.rho_h[material].resize(s.nodes.size());
      s.K_h[material].resize(s.nodes.size());
      s.q.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
      s.dp_de_h[material].resize(s.nodes.size());
      if (in.enable_p_prime[material]) {
        s.p_prime.resize(s.points.size());
      }
    }
    if (in.enable_hyper_ep[material]) {
      s.ep_h[material].resize(s.nodes.size());
    }
  }
  s.material.resize(s.elements.size());
  if (in.enable_adapt) {
    s.quality.resize(s.elements.size());
    s.h_adapt.resize(s.nodes.size());
  }
}

void otm_initialize(input& in, state& s, std::string const& filename)
{
  in.otm_material_points_to_add_per_element = 4;

  hpc::host_vector<hpc::position<double>, point_node_index> tet_gauss_pts(4);
  tet_gauss_pts[0] = { 0.1381966011250105, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[1] = { 0.5854101966249685, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[2] = { 0.1381966011250105, 0.5854101966249685, 0.1381966011250105 };
  tet_gauss_pts[3] = { 0.1381966011250105, 0.1381966011250105, 0.5854101966249685 };
  lgr::tet_gauss_points_to_material_points point_interpolator(tet_gauss_pts);
  in.xp_transform = std::ref(point_interpolator);

  auto const err_code = lgr::read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  lgr::convert_tet_mesh_to_meshless(s, in);
  in.name = "OTM";
  in.end_time = 0.001;
  in.num_file_outputs = 100;
  double const rho{7.8e+03};
  double const nu{0.25};
  double const E{200.0e09};
  double const K{E / (3.0 * (1.0 - 2.0 * nu))};
  double const G{E / (2.0 * (1.0 + nu))};
  double const Y0{1.0e+09};
  double const n{4.0};
  double const eps0{1e-2};
  double const Svis0{Y0};
  double const m{2.0};
  double const eps_dot0{1e-1};
  constexpr material_index body(0);
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
  auto constant_vy = [=] (
    hpc::counting_range<node_index> const nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const&,
    hpc::device_array_vector<hpc::velocity<double>, node_index>* v_vector) {
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=] HPC_DEVICE (node_index const node) {
      auto const v = hpc::velocity<double>(0.0, 10.0, 0.0);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = constant_vy;
  in.CFL = 0.1;
  otm_initialize_state(in, s);
}

void otm_run(std::string const& filename)
{
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state s;
  otm_initialize(in, s, filename);
  std::cout << std::scientific << std::setprecision(17);
  auto const num_file_outputs = in.num_file_outputs;
  auto const file_output_period = num_file_outputs ? in.end_time / double(num_file_outputs) : hpc::time<double>(0.0);
}

} // namespace lgr
