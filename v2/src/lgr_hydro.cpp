#include <Omega_h_align.hpp>
#include <lgr_element_functions.hpp>
#include <lgr_for.hpp>
#include <lgr_hydro.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
void initialize_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_gradients = sim.set(sim.gradient);
  auto const points_to_weights = sim.set(sim.weight);
  auto const nodes_to_X = sim.get(sim.ref_coords);
  auto const points_to_F = sim.set(sim.def_grad);
  auto const points_to_J = sim.set(sim.det_def_grad);
  auto const elems_to_nodes = sim.elems_to_nodes();
  auto const elems_to_time_len = sim.set(sim.time_step_length);
  auto const elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto const X = getvecs<Elem>(nodes_to_X, elem_nodes);
    auto const shape = Elem::shape(X);
    elems_to_time_len[elem] = shape.lengths.time_step_length;
    elems_to_visc_len[elem] = shape.lengths.viscosity_length;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const pt = elem * Elem::points + elem_pt;
      setgrads<Elem>(points_to_gradients, pt, shape.basis_gradients[elem_pt]);
      points_to_weights[pt] = shape.weights[elem_pt];
      setfull<Elem>(points_to_F, pt, identity_matrix<Elem::dim, Elem::dim>());
      points_to_J[pt] = 1.0;
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void reset_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const field_x = sim.get(sim.position);
  auto const field_X = sim.set(sim.ref_coords);
  auto const field_rho = sim.get(sim.density);
  auto const field_rho0 = sim.set(sim.ref_density);
  // copy from -> into
  Omega_h::copy_into(field_x, field_X);
  Omega_h::copy_into(field_rho, field_rho0);
  auto const points_to_F = sim.set(sim.def_grad);
  auto const points_to_F_init = sim.set(sim.def_grad_init);
  auto  functor = OMEGA_H_LAMBDA(int const point) {
    auto F = getfull<Elem>(points_to_F, point);
    auto F_init = getfull<Elem>(points_to_F_init, point);
    auto F_init_new = F * F_init;  // right order?
    auto const F_new = identity_matrix<Elem::dim, Elem::dim>();
    setfull<Elem>(points_to_F, point, F_new);
    setfull<Elem>(points_to_F_init, point, F_init_new);
  };
  parallel_for(sim.points(), std::move(functor));
}

template <class Elem>
void lump_normal_masses(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_rho0 = sim.get(sim.ref_density);
  auto const points_to_w = sim.get(sim.weight);
  auto const nodes_to_elems = sim.nodes_to_elems();
  auto const nodes_to_mass = sim.set(sim.nodal_mass);
  auto functor = OMEGA_H_LAMBDA(int node) {
    double node_mass = 0.0;
    auto const begin = nodes_to_elems.a2ab[node];
    auto const end = nodes_to_elems.a2ab[node + 1];
    for (auto node_elem = begin; node_elem < end; ++node_elem) {
      auto const elem = nodes_to_elems.ab2b[node_elem];
      auto const code = nodes_to_elems.codes[node_elem];
      auto const elem_node = Omega_h::code_which_down(code);
      double elem_mass = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const rho0 = points_to_rho0[point];
        auto const w = points_to_w[point];
        elem_mass += rho0 * w;
      }
      node_mass += elem_mass * Elem::lumping_factor(elem_node);
    }
    nodes_to_mass[node] = node_mass;
  };
  parallel_for(sim.nodes(), std::move(functor));
}

static void lump_comptet_masses(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_x = sim.get(sim.position);
  auto const points_to_rho0 = sim.get(sim.ref_density);
  auto const elems_to_nodes = sim.elems_to_nodes();
  auto const nodes_to_elems = sim.nodes_to_elems();
  auto const nodes_to_mass = sim.set(sim.nodal_mass);
  auto functor = OMEGA_H_LAMBDA(int node) {
    double node_mass = 0.0;
    Omega_h::Vector<CompTet::points> density;
    auto const begin = nodes_to_elems.a2ab[node];
    auto const end = nodes_to_elems.a2ab[node + 1];
    for (auto node_elem = begin; node_elem < end; ++node_elem) {
      auto const elem = nodes_to_elems.ab2b[node_elem];
      auto const code = nodes_to_elems.codes[node_elem];
      auto const elem_node = Omega_h::code_which_down(code);
      auto const elem_nodes = getnodes<CompTet>(elems_to_nodes, elem);
      auto const x = getvecs<CompTet>(nodes_to_x, elem_nodes);
      for (int elem_pt = 0; elem_pt < CompTet::points; ++elem_pt) {
        auto const point = elem * CompTet::points + elem_pt;
        density[elem_pt] = points_to_rho0[point];
      }
      auto const lumped = CompTet::lump_mass(x, density);
      node_mass += lumped[elem_node];
    }
    nodes_to_mass[node] = node_mass;
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void lump_masses(Simulation& sim) {
  if (sim.elem_name == "CompTet") lump_comptet_masses(sim);
  else lump_normal_masses<Elem>(sim);
}

template <class Elem>
void update_position(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_v = sim.getset(sim.velocity);
  auto const nodes_to_x = sim.getset(sim.position);
  auto const nodes_to_a = sim.get(sim.acceleration);
  auto const dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const v_n = getvec<Elem>(nodes_to_v, node);
    auto const x_n = getvec<Elem>(nodes_to_x, node);
    auto const a_n = getvec<Elem>(nodes_to_a, node);
    auto const v_np12 = v_n + (dt / 2.0) * a_n;
    auto const x_np1 = x_n + dt * v_np12;
    setvec<Elem>(nodes_to_v, node, v_np12);
    setvec<Elem>(nodes_to_x, node, x_np1);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void update_lengths(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const elems_to_nodes = sim.disc.ents_to_nodes(ELEMS);
  auto const nodes_to_x = sim.get(sim.position);
  auto const elems_to_time_len = sim.set(sim.time_step_length);
  auto const elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto const x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto const len = Elem::lengths(x);
    elems_to_time_len[elem] = len.time_step_length;
    elems_to_visc_len[elem] = len.viscosity_length;
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void update_deformation_gradients(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_x = sim.get(sim.position);
  auto const nodes_to_X = sim.get(sim.ref_coords);
  auto const points_to_grad = sim.get(sim.gradient);
  auto const points_to_rho0 = sim.get(sim.ref_density);
  auto const points_to_F = sim.set(sim.def_grad);
  auto const points_to_J = sim.set(sim.det_def_grad);
  auto const points_to_rho = sim.set(sim.density);
  auto const elems_to_nodes = sim.elems_to_nodes();
  auto functor = OMEGA_H_LAMBDA(int const point) {
    auto const elem = point / Elem::points;
    auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto const x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto const X = getvecs<Elem>(nodes_to_X, elem_nodes);
    auto const BT = getgrads<Elem>(points_to_grad, point);
    auto const B = Omega_h::transpose(BT);
    auto const I = identity_matrix<Elem::dim, Elem::dim>();
    //auto const F = x * B;
    //so... the above expression causes an instability (explosion!) in the elastic wave tests
    //with the velocity Verlet (Newmark-Beta) time integrator
    //using 1000 elements along the domain and running out to t = 32e-3
    //the below expression seems to be stable for CFL=0.8 for Bar2,
    //CFL=0.5 for Tri3 and Tet4.
    //the lower CFL may be due to using the height as h_min.
    //Jake Ostien commented earlier that using (I + u * B) for F in Sierra helped fix a bug
    //there as well, not sure if they're related.
    auto const F = I + ((x - X) * B);
    auto const J = determinant(F);
    auto const rho0 = points_to_rho0[point];
    OMEGA_H_CHECK(J > 0.0);
    setfull<Elem>(points_to_F, point, F);
    points_to_J[point] = J;
    points_to_rho[point] = rho0 / J;
  };
  parallel_for(sim.points(), std::move(functor));
}

template <class Elem>
void vol_avg_deformation_gradients(Simulation& sim) {
  if (sim.elem_name != "CompTet") return;
  LGR_SCOPE(sim);
  auto const points_to_J = sim.get(sim.det_def_grad);
  auto const points_to_wts = sim.get(sim.weight);
  auto const points_to_F = sim.set(sim.def_grad);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    double vol = 0.0;
    double J_bar = 0.0;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const point = elem * Elem::points + elem_pt;
      auto const wt = points_to_wts[point];
      auto const J = points_to_J[point];
      vol += wt;
      J_bar += wt * J;
    }
    J_bar /= vol;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const point = elem * Elem::points + elem_pt;
      auto const J = points_to_J[point];
      double fac = Omega_h::root<Elem::dim>(J_bar / J);
      auto F = getfull<Elem>(points_to_F, point);
      F *= fac;
      setfull<Elem>(points_to_F, point, F);
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void update_configuration(Simulation& sim) {
  update_lengths<Elem>(sim);
  update_deformation_gradients<Elem>(sim);
  vol_avg_deformation_gradients<Elem>(sim); // BNG
}

template <class Elem>
void correct_velocity(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_v = sim.getset(sim.velocity);
  auto const nodes_to_a = sim.get(sim.acceleration);
  auto const dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const v_np12 = getvec<Elem>(nodes_to_v, node);
    auto const a_np1 = getvec<Elem>(nodes_to_a, node);
    auto const v_np1 = v_np12 + (dt / 2.0) * a_np1;
    setvec<Elem>(nodes_to_v, node, v_np1);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void pull_back_stress(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_F = sim.get(sim.def_grad);
  auto const points_to_sigma = sim.get(sim.stress);
  auto const points_to_first_pk = sim.set(sim.first_pk);
  auto functor = OMEGA_H_LAMBDA(int const point) {
    auto const F = getfull<Elem>(points_to_F, point);
    auto const sigma = resize<Elem::dim>(getstress(points_to_sigma, point));
    auto const J = determinant(F);
    auto const FinvT = transpose(invert(F));
    auto const P = J * sigma * FinvT;
    setfull<Elem>(points_to_first_pk, point, P);
  };
  parallel_for(sim.points(), std::move(functor));
}

template <class Elem>
void pressure_avg_first_pk(Simulation& sim) {
  if (sim.elem_name != "CompTet") return;
  LGR_SCOPE(sim);
  auto const points_to_F = sim.get(sim.def_grad);
  auto const points_to_J = sim.get(sim.det_def_grad);
  auto const points_to_wts = sim.get(sim.weight);
  auto const points_to_first_pk = sim.set(sim.first_pk);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    double vol = 0.0;
    double p_bar = 0.0;
    double J_bar = 0.0;
    Omega_h::Vector<Elem::points> inner;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const point = elem * Elem::points + elem_pt;
      auto wt = points_to_wts[point];
      auto F = getfull<Elem>(points_to_F, point);
      auto first_pk = getfull<Elem>(points_to_first_pk, point);
      inner[elem_pt] = inner_product(F, first_pk);
      // assumption that J_bar is equal for all Fs in the elem
      J_bar = determinant(F);
      vol += wt;
      p_bar += wt * inner[elem_pt] / (Elem::dim * J_bar);
    }
    p_bar /= vol;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const point = elem * Elem::points + elem_pt;
      auto J_not_vol_avg = points_to_J[point];
      auto fac = Omega_h::root<Elem::dim>(J_bar / J_not_vol_avg);
      auto F = getfull<Elem>(points_to_F, point);
      auto first_pk = getfull<Elem>(points_to_first_pk, point);
      auto pk_adjust = invert(transpose(F));
      pk_adjust *= ((p_bar * J_not_vol_avg) - (inner[elem_pt] / 3.0));
      first_pk += pk_adjust;
      first_pk *= fac;
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

OMEGA_H_DEVICE
void do_proj_first_pk_stress(
    Omega_h::Vector<4> weights,
    Omega_h::Matrix<3, 10> node_coords,
    Omega_h::Few<Omega_h::Matrix<3, 3>, 4>& first_pk) {
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> stress_integral;
  for (int i = 0; i < lgr::CompTet::nbarycentric_coords; ++i) {
    stress_integral[i] = Omega_h::zero_matrix<3, 3>();
  }
  auto ref_points = lgr::CompTet::get_ref_points();
  for (int pt = 0;  pt < lgr::CompTet::points; ++pt) {
    auto lambda = lgr::CompTet::get_barycentric_coord(ref_points[pt]);
    for (int l1 = 0; l1 < lgr::CompTet::nbarycentric_coords; ++l1) {
      stress_integral[l1] += lambda[l1] * weights[pt] * first_pk[pt];
    }
  }
  auto M_inv = lgr::CompTet::compute_M_inv(node_coords);
  for (int pt = 0;  pt < lgr::CompTet::points; ++pt) {
    auto lambda = lgr::CompTet::get_barycentric_coord(ref_points[pt]);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        first_pk[pt](i, j) = 0.0;
        for (int l1 = 0;  l1 < lgr::CompTet::nbarycentric_coords; ++l1) {
          for (int l2 = 0;  l2 < lgr::CompTet::nbarycentric_coords; ++l2) {
            first_pk[pt](i, j) += lambda[l1] * M_inv(l1, l2) * stress_integral[l2](i, j);
          }
        }
      }
    }
  }
}

static void linear_proj_first_pk(Simulation& sim) {
  if (sim.elem_name != "CompTet") return;
  LGR_SCOPE(sim);
  auto const points_to_wts = sim.get(sim.weight);
  auto const points_to_first_pk = sim.set(sim.first_pk);
  auto const elems_to_nodes = sim.disc.ents_to_nodes(ELEMS);
  auto const nodes_to_X = sim.get(sim.ref_coords);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    Omega_h::Vector<4> weights;
    Omega_h::Few<Omega_h::Matrix<3, 3>, 4> first_pk;
    auto const elem_nodes = getnodes<CompTet>(elems_to_nodes, elem);
    auto const X = getvecs<CompTet>(nodes_to_X, elem_nodes);
    for (int elem_pt = 0; elem_pt < CompTet::points; ++elem_pt) {
      auto const point = elem * CompTet::points + elem_pt;
      weights[elem_pt] = points_to_wts[point];
      first_pk[elem_pt] = getfull<CompTet>(points_to_first_pk, point);
    }
    do_proj_first_pk_stress(weights, X, first_pk);
    for (int elem_pt = 0; elem_pt < CompTet::points; ++elem_pt) {
      auto const point = elem * CompTet::points + elem_pt;
      setfull<CompTet>(points_to_first_pk, point, first_pk[elem_pt]);
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void compute_nodal_force(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_first_pk = sim.get(sim.first_pk);
  auto const points_to_grads = sim.get(sim.gradient);
  auto const points_to_weights = sim.get(sim.weight);
  auto const nodes_to_f = sim.set(sim.force);
  auto const nodes_to_elems = sim.nodes_to_elems();
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto node_f = zero_vector<Elem::dim>();
    auto const begin = nodes_to_elems.a2ab[node];
    auto const end = nodes_to_elems.a2ab[node + 1];
    for (auto node_elem = begin; node_elem < end; ++node_elem) {
      auto const elem = nodes_to_elems.ab2b[node_elem];
      auto const code = nodes_to_elems.codes[node_elem];
      auto const elem_node = Omega_h::code_which_down(code);
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const grad =
            getvec<Elem>(points_to_grads, point * Elem::nodes + elem_node);
        auto const first_pk = getfull<Elem>(points_to_first_pk, point);
        auto const weight = points_to_weights[point];
        auto const cell_f = -(first_pk * grad) * weight;
        node_f += cell_f;
      }
    }
    setvec<Elem>(nodes_to_f, node, node_f);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_stress_divergence(Simulation& sim) {
  pull_back_stress<Elem>(sim);
  pressure_avg_first_pk<Elem>(sim); // BNG
  linear_proj_first_pk(sim);        // BNG
  compute_nodal_force<Elem>(sim);
}

template <class Elem>
void compute_nodal_acceleration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_f = sim.get(sim.force);
  auto const nodes_to_m = sim.get(sim.nodal_mass);
  auto const nodes_to_a = sim.set(sim.acceleration);
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const f = getvec<Elem>(nodes_to_f, node);
    auto const m = nodes_to_m[node];
    auto const a = f / m;
    setvec<Elem>(nodes_to_a, node, a);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_point_time_steps(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_c = sim.get(sim.wave_speed);
  auto const elems_to_h = sim.get(sim.time_step_length);
  auto const points_to_dt = sim.set(sim.point_time_step);
  double const max = std::numeric_limits<double>::max();
  auto functor = OMEGA_H_LAMBDA(int const point) {
    auto const elem = point / Elem::points;
    auto const h = elems_to_h[elem];
    OMEGA_H_CHECK(h > 0.0);
    auto const c = points_to_c[point];
    OMEGA_H_CHECK(c >= 0.0);
    auto const dt = (c == 0.0) ? max : (h / c);
    OMEGA_H_CHECK(dt > 0.0);
    points_to_dt[point] = dt;
  };
  parallel_for(sim.points(), std::move(functor));
}

#define LGR_EXPL_INST(Elem)                                                    \
  template void initialize_configuration<Elem>(Simulation & sim);              \
  template void reset_configuration<Elem>(Simulation & sim);                   \
  template void lump_masses<Elem>(Simulation & sim);                           \
  template void update_position<Elem>(Simulation & sim);                       \
  template void update_configuration<Elem>(Simulation & sim);                  \
  template void correct_velocity<Elem>(Simulation & sim);                      \
  template void compute_stress_divergence<Elem>(Simulation & sim);             \
  template void compute_nodal_acceleration<Elem>(Simulation & sim);            \
  template void compute_point_time_steps<Elem>(Simulation & sim);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
