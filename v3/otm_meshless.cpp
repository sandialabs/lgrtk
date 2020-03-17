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
#include <otm_util.hpp>
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
      HPC_TRACE("ERROR : " << error << "  EPS : " << eps << '\n');
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
  auto const num_points = s.points.size();
  auto const num_nodes = s.nodes.size();
  auto const num_elements = s.elements.size();
  auto const num_materials = in.materials.size();
  HPC_TRACE("NUM NODES     : " << num_nodes << '\n');
  HPC_TRACE("NUM POINTS    : " << num_points << '\n');
  HPC_TRACE("NUM ELEMENTS  : " << num_elements << '\n');
  HPC_TRACE("NUM MATERIALS : " << num_materials << '\n');
  s.u.resize(num_nodes);
  s.v.resize(num_nodes);
  s.V.resize(num_points);
  s.grad_N.resize(num_points * s.points_to_point_nodes.size());
  s.N.resize(num_points * s.points_to_point_nodes.size());
  s.F_total.resize(num_points);
  s.sigma.resize(num_points);
  s.symm_grad_v.resize(num_points);
  s.K.resize(num_points);
  s.G.resize(num_points);
  s.potential_density.resize(num_points);
  s.c.resize(num_points);
  s.element_f.resize(num_points * s.nodes_in_element.size());
  s.lm.resize(num_nodes);
  s.f.resize(num_nodes);
  s.rho.resize(num_points);
  s.b.resize(num_points);
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
}

template <typename Quantity, typename Index, typename Range>
void otm_initialize_quantity(
    hpc::device_array_vector<Quantity, Index>& array, Quantity const& value, Range& range)
{
  array.resize(range.size());
  auto const node_or_point_to_array = array.begin();
  auto functor = [=] HPC_DEVICE (Index const node_or_point) {
    node_or_point_to_array[node_or_point] = value;
  };
  hpc::for_each(hpc::device_policy(), range, functor);
}

void otm_initialize(input& in, state& s, std::string const& filename)
{
  auto const points_per_element = point_index(4);
  in.otm_material_points_to_add_per_element = points_per_element;

  hpc::host_vector<hpc::position<double>, point_node_index> tet_gauss_pts(4);
  tet_gauss_pts[0] = { 0.1381966011250105, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[1] = { 0.5854101966249685, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[2] = { 0.1381966011250105, 0.5854101966249685, 0.1381966011250105 };
  tet_gauss_pts[3] = { 0.1381966011250105, 0.1381966011250105, 0.5854101966249685 };
  lgr::tet_nodes_to_points point_interpolator(points_per_element);
  in.xp_transform = std::ref(point_interpolator);

  auto const err_code = lgr::read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  in.name = "OTM";
  in.end_time = 0.001;
  in.num_file_outputs = 100;
  auto const rho = hpc::density<double>(7.8e+03);
  auto const nu = hpc::adimensional<double>(0.25);
  auto const E = hpc::pressure<double>(200.0e09);
  auto const K = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0 = hpc::pressure<double>(1.0e+09);
  auto const n = hpc::adimensional<double>(4.0);
  auto const eps0 = hpc::strain<double>(1.0e-02);
  auto const Svis0 = hpc::pressure<double>(Y0);
  auto const m = hpc::adimensional<double>(2.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
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
  in.CFL = 0.1;
  otm_initialize_state(in, s);
  lgr::convert_tet_mesh_to_meshless(in, s);
  auto const v0 = hpc::velocity<double>(0.0, 10.0, 0.0);
  otm_initialize_quantity(s.v, v0, s.nodes);
  auto const u0 = hpc::velocity<double>(0.0, 0.0, 0.0);
  otm_initialize_quantity(s.u, u0, s.nodes);
#if 1
  {
  auto const nodes_to_x = s.x.cbegin();
  auto print_x = [=] HPC_HOST (lgr::node_index const node) {
    auto const x = nodes_to_x[node].load();
    HPC_TRACE("node: " << node << ", x:\n" << x);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_x);

  auto const points_to_xp = s.xp.cbegin();
  auto print_xp = [=] HPC_HOST (lgr::point_index const point) {
    auto const xp = points_to_xp[point].load();
    HPC_TRACE("point: " << point << ", xp:\n" << xp);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_xp);

  auto const nodes_to_u = s.u.cbegin();
  auto print_u = [=] HPC_HOST (lgr::node_index const node) {
    auto const u = nodes_to_u[node].load();
    HPC_TRACE("node: " << node << ", u:\n" << u);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_u);
  }
#endif
  lgr::otm_initialize_grad_val_N(s);
  lgr::otm_lump_nodal_mass(s);
  lgr::otm_initialize_F(s);
  lgr::otm_update_reference(s);
  lgr::otm_update_material_state(in, s, 0);
}

void otm_run(std::string const& filename)
{
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state s;
  otm_initialize(in, s, filename);
  std::cout << std::scientific << std::setprecision(17);
  //auto const num_file_outputs = in.num_file_outputs;
  //auto const file_output_period = num_file_outputs ? in.end_time / double(num_file_outputs) : hpc::time<double>(0.0);
}

} // namespace lgr
