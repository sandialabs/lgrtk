#include <lgr_circular_symmetry.hpp>
#include <lgr_element_functions.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
struct CircularSymmetry : public Model<Elem> {
  int radius_coordinate;
  double angle;
  CircularSymmetry(Simulation& sim_in, Omega_h::InputMap& pl):Model<Elem>(sim_in, pl) {
    this->radius_coordinate = sim_in.get_int(pl, "radius coordinate", "0");
    this->angle = sim_in.get_double(pl, "angle", "2.0 * pi");
  }
  std::uint64_t exec_stages() override final { return AFTER_CONFIGURATION; }
  char const* name() override final { return "circular symmetry"; }
  void after_configuration() override final {
    auto const points_to_w = this->points_getset(this->sim.weight);
    auto const points_to_rho = this->points_getset(this->sim.density);
    auto const nodes_to_x = this->sim.get(this->sim.position);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const rc = this->radius_coordinate;
    auto const theta = this->angle;
    auto const t = this->sim.time;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_point = point % Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const x = getvecs<Elem>(nodes_to_x, elem_nodes);
      auto const N = Elem::basis_values()[elem_point];
      auto const point_x = x * N;
      auto const r = point_x[rc];
      auto const w_old = points_to_w[point];
      auto const rho_old = points_to_rho[point];
      auto const m = w_old * rho_old;
      auto const w_new = w_old * r * theta;
      auto const rho_new = m / w_new;
      points_to_w[point] = w_new;
      points_to_rho[point] = (t == 0.0) ? rho_old : rho_new;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* circular_symmetry_factory(
    Simulation& sim, std::string const&,
    Omega_h::InputMap& pl) {
  return new CircularSymmetry<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* \
circular_symmetry_factory<Elem>( \
    Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
