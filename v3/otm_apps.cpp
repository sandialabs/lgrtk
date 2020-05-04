#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>
#include <otm_apps.hpp>
#include <otm_meshless.hpp>
#include <otm_tet2meshless.hpp>
#include <otm_tetrahedron_util.hpp>
#include <otm_util.hpp>

namespace lgr {

hpc::stress<double> otm_compute_stress_average(state const& s)
{
  auto num_points = s.points.size();
  auto points_to_sigma_full = s.sigma_full.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const sigma = points_to_sigma_full[point].load();
    return sigma;
  };
  hpc::stress<double> init(0, 0, 0, 0, 0, 0, 0, 0, 0);
  auto const sigma_sum = hpc::transform_reduce(hpc::device_policy(), s.points, init, hpc::plus<hpc::stress<double> >(), functor);
  return sigma_sum / num_points;
}

hpc::deformation_gradient<double> otm_compute_deformation_gradient_average(state const& s)
{
  auto num_points = s.points.size();
  auto points_to_F = s.F_total.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const F = points_to_F[point].load();
    return F;
  };
  hpc::deformation_gradient<double> init(0, 0, 0, 0, 0, 0, 0, 0, 0);
  auto const F_sum = hpc::transform_reduce(hpc::device_policy(), s.points, init, hpc::plus<hpc::deformation_gradient<double> >(), functor);
  return F_sum / num_points;
}

void otm_nu_zero_patch_test_ics(state& s, hpc::speed<double> const top_velocity)
{
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const top_vel = top_velocity;
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const x = nodes_to_x[node].load();
    auto const vz = top_vel * x(2);
    nodes_to_u[node] = hpc::position<double>(0.0, 0.0, 0.0);
    nodes_to_v[node] = hpc::velocity<double>(0.0, 0.0, vz);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_uniaxial_patch_test_ics(state& s, hpc::speed<double> const top_velocity, hpc::adimensional<double> nu)
{
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const top_vel = top_velocity;
  auto const trans_vel = -nu * top_velocity;
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto const x = nodes_to_x[node].load();
    auto const vx = trans_vel * x(0);
    auto const vy = trans_vel * x(1);
    auto const vz = top_vel * x(2);
    nodes_to_u[node] = hpc::position<double>(0.0, 0.0, 0.0);
    nodes_to_v[node] = hpc::velocity<double>(vx, vy, vz);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_cylindrical_flyer_ics(state& s, hpc::speed<double> const init_velocity)
{
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const vz = init_velocity;
  auto functor = [=] HPC_DEVICE (node_index const node) {
    nodes_to_u[node] = hpc::position<double>(0.0, 0.0, 0.0);
    nodes_to_v[node] = hpc::velocity<double>(0.0, 0.0, vz);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

bool otm_j2_nu_zero_patch_test()
{
  material_index num_materials(1);
  material_index num_boundaries(4);
  input in(num_materials, num_boundaries);
  state s;
  std::string const filename{"cube.g"};
  auto const points_in_element = 1;
  s.points_in_element.resize(point_in_element_index(points_in_element));
  in.otm_material_points_to_add_per_element = points_in_element;
  tet_nodes_to_points point_interpolator(points_in_element);
  in.xp_transform = std::ref(point_interpolator);
  auto const err_code = read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  in.name = "nu-zero-patch-test";
  in.end_time = 1.0e-03;
  in.num_file_output_periods = 100;
  in.do_output = true;
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
  convert_tet_mesh_to_meshless(in, s);
  s.dt = in.constant_dt;
  s.dt_old = in.constant_dt;
  s.time = hpc::time<double>(0.0);
  s.max_stable_dt = in.constant_dt;
  s.num_time_steps = static_cast<int>(std::round(in.end_time / in.constant_dt));
  otm_allocate_state(in, s);
  auto const top_velocity = hpc::speed<double>(10.0);
  s.prescribed_v[body] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[body] = hpc::vector3<int>(0, 0, 0);
  s.prescribed_v[x_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[x_min] = hpc::vector3<int>(1, 0, 0);
  s.prescribed_v[y_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[y_min] = hpc::vector3<int>(0, 1, 0);
  s.prescribed_v[z_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[z_min] = hpc::vector3<int>(0, 0, 1);
  s.prescribed_v[z_max] = hpc::velocity<double>(0.0, 0.0, top_velocity);
  s.prescribed_dof[z_max] = hpc::vector3<int>(0, 0, 1);
  auto const I = hpc::deformation_gradient<double>::identity();
  auto const ep0 = hpc::strain<double>(0.0);
  hpc::fill(hpc::device_policy(), s.rho, rho);
  hpc::fill(hpc::device_policy(), s.F_total, I);
  hpc::fill(hpc::device_policy(), s.Fp_total, I);
  hpc::fill(hpc::device_policy(), s.ep, ep0);
  otm_nu_zero_patch_test_ics(s, top_velocity);
  otm_initialize(in, s);
  otm_run(in, s);
  auto const top_displacement = top_velocity * s.time;
  auto const F33 = hpc::strain<double>(1.0 + top_displacement / hpc::length<double>(1.0));
  auto const F = hpc::deformation_gradient<double>(1, 0, 0, 0, 1, 0, 0, 0, F33);
  auto const J = hpc::determinant(F);
  auto const C = hpc::transpose(F) * F;
  auto const e = 0.5 * hpc::log(C);
  // Not true in general, but this is nu_zero deformation
  auto sigma_gold = (E / J) * e;
  auto const sigma_avg = otm_compute_stress_average(s);
  auto const error_sigma = hpc::norm(sigma_avg - sigma_gold) / hpc::norm(sigma_gold);
  auto const tol_sigma = 1.0e-12;
  std::cout << "STRESS GOLD:\n" << sigma_gold << "STRESS AVERAGE:\n" << sigma_avg << '\n';
  std::cout << "STRESS ERROR: " << error_sigma << ", STRESS TOLERANCE: " << tol_sigma << '\n';
  return error_sigma <= tol_sigma;
}

bool otm_j2_uniaxial_patch_test()
{
  material_index num_materials(1);
  material_index num_boundaries(4);
  input in(num_materials, num_boundaries);
  state s;
  std::string const filename{"cube.g"};
  auto const points_in_element = 1;
  s.points_in_element.resize(point_in_element_index(points_in_element));
  in.otm_material_points_to_add_per_element = points_in_element;
  tet_nodes_to_points point_interpolator(points_in_element);
  in.xp_transform = std::ref(point_interpolator);
  auto const err_code = read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  in.name = "uniaxial-patch-test";
  in.end_time = 1.0e-03;
  in.num_file_output_periods = 100;
  in.do_output = false;
  in.debug_output = true;
  auto const rho = hpc::density<double>(7.8e+03);
  auto const nu = hpc::adimensional<double>(0.25);
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
  convert_tet_mesh_to_meshless(in, s);
  s.dt = in.constant_dt;
  s.dt_old = in.constant_dt;
  s.time = hpc::time<double>(0.0);
  s.max_stable_dt = in.constant_dt;
  s.num_time_steps = static_cast<int>(std::round(in.end_time / in.constant_dt));
  otm_allocate_state(in, s);
  auto const top_velocity = hpc::speed<double>(10.0);
  auto const trans_velocity = -nu * top_velocity;
  s.prescribed_v[body] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[body] = hpc::vector3<int>(0, 0, 0);
  s.prescribed_v[x_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[x_min] = hpc::vector3<int>(1, 0, 0);
  s.prescribed_v[y_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[y_min] = hpc::vector3<int>(0, 1, 0);
  s.prescribed_v[z_min] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[z_min] = hpc::vector3<int>(0, 0, 1);
  s.prescribed_v[z_max] = hpc::velocity<double>(0.0, 0.0, top_velocity);
  s.prescribed_dof[z_max] = hpc::vector3<int>(0, 0, 1);
  auto const I = hpc::deformation_gradient<double>::identity();
  auto const ep0 = hpc::strain<double>(0.0);
  hpc::fill(hpc::device_policy(), s.rho, rho);
  hpc::fill(hpc::device_policy(), s.F_total, I);
  hpc::fill(hpc::device_policy(), s.Fp_total, I);
  hpc::fill(hpc::device_policy(), s.ep, ep0);
  otm_uniaxial_patch_test_ics(s, top_velocity, nu);
  otm_initialize(in, s);
  otm_run(in, s);
  auto const top_displacement = top_velocity * s.time;
  auto const trans_displacement = trans_velocity * s.time;
  auto const F11 = hpc::strain<double>(1.0 + trans_displacement / hpc::length<double>(1.0));
  auto const F22 = hpc::strain<double>(1.0 + trans_displacement / hpc::length<double>(1.0));
  auto const F33 = hpc::strain<double>(1.0 + top_displacement / hpc::length<double>(1.0));
  auto const F = hpc::deformation_gradient<double>(F11, 0, 0, 0, F22, 0, 0, 0, F33);
  auto const J = hpc::determinant(F);
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Cdev = Jm23 * hpc::transpose(F) * F;
  auto const edev = 0.5 * hpc::log(Cdev);
  auto const p = K * std::log(J) / J;
  auto const Mdev = 2.0 * G * edev;
  auto sigma_vol = p * hpc::matrix3x3<double>::identity();
  auto sigma_dev = hpc::transpose(hpc::inverse(F)) * Mdev * hpc::transpose(F) / J;
  auto sigma_gold = sigma_vol + sigma_dev;
  auto const sigma_avg = otm_compute_stress_average(s);
  auto const F_avg = otm_compute_deformation_gradient_average(s);
  auto const error_sigma = hpc::norm(sigma_avg - sigma_gold) / hpc::norm(sigma_gold);
  auto const tol_sigma = 3.62e-03;
  std::cout << "DEF GRAD GOLD:\n" << F << "DEF GRAD AVERAGE:\n" << F_avg << '\n';
  std::cout << "STRESS GOLD:\n" << sigma_gold << "STRESS AVERAGE:\n" << sigma_avg << '\n';
  std::cout << "STRESS ERROR: " << error_sigma << ", STRESS TOLERANCE: " << tol_sigma << '\n';
  return error_sigma <= tol_sigma;
}

bool otm_cylindrical_flyer()
{
  material_index num_materials(1);
  material_index num_boundaries(0);
  input in(num_materials, num_boundaries);
  state s;
  std::string const filename{"cylinder.g"};
  auto const points_in_element = 1;
  s.points_in_element.resize(point_in_element_index(points_in_element));
  s.use_custom_initial_support_size = true;
  s.initial_support_size = node_in_element_index(4);
  in.otm_material_points_to_add_per_element = points_in_element;
  tet_nodes_to_points point_interpolator(points_in_element);
  in.xp_transform = std::ref(point_interpolator);
  auto const err_code = read_exodus_file(filename, in, s);
  if (err_code != 0) {
    HPC_ERROR_EXIT("Reading Exodus file : " << filename);
  }
  in.name = "cylindrical-flyer";
  in.end_time = 1.0e-04;
  in.num_file_output_periods = 100;
  in.do_output = true;
  in.debug_output = false;
  auto const rho = hpc::density<double>(8.96e+03);
  auto const nu = hpc::adimensional<double>(0.343);
  auto const E = hpc::pressure<double>(110.0e09);
  auto const K = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0 = hpc::pressure<double>(400.0e+06);
  auto const n = hpc::adimensional<double>(1.0);
  auto const H0 = hpc::pressure<double>(100.0e6);
  auto const eps0 = hpc::strain<double>(Y0 / H0);
  auto const Svis0 = hpc::pressure<double>(0.0);
  auto const m = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
  constexpr material_index body(0);
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
  in.use_constant_dt = false;
  in.constant_dt = hpc::time<double>(1.0e-07);
  in.domains[body] = std::make_unique<clipped_domain<all_space>>(all_space{});
  convert_tet_mesh_to_meshless(in, s);
  s.dt = in.constant_dt;
  s.dt_old = in.constant_dt;
  s.time = hpc::time<double>(0.0);
  s.max_stable_dt = in.constant_dt;
  s.num_time_steps = static_cast<int>(std::round(in.end_time / in.constant_dt));
  s.use_penalty_contact = true;
  s.contact_penalty_coeff = hpc::strain_rate_rate<double>(1.0e14);
  otm_allocate_state(in, s);
  s.prescribed_v[body] = hpc::velocity<double>(0.0, 0.0, 0.0);
  s.prescribed_dof[body] = hpc::vector3<int>(0, 0, 0);
  auto const I = hpc::deformation_gradient<double>::identity();
  auto const ep0 = hpc::strain<double>(0.0);
  hpc::fill(hpc::device_policy(), s.rho, rho);
  hpc::fill(hpc::device_policy(), s.F_total, I);
  hpc::fill(hpc::device_policy(), s.Fp_total, I);
  hpc::fill(hpc::device_policy(), s.ep, ep0);
  auto const init_velocity = hpc::speed<double>(227.0);
  otm_cylindrical_flyer_ics(s, init_velocity);
  otm_initialize(in, s);
  otm_run(in, s);
  return true;
}

} // namespace lgr
