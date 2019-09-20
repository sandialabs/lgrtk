#include <lgr_l2_error.hpp>
#include <lgr_node_scalar.hpp>
#include <lgr_scalars.hpp>
#include <lgr_field.hpp>
#include <lgr_simulation.hpp>
#include <algorithm>
#include <string>

namespace lgr {

std::string const& NameOfScalarPtr::operator()(Scalar* ptr) {
  return ptr->name;
}

Scalars::Scalars(Simulation& sim_in) : sim(sim_in) {}

double Scalars::ask_value(std::string const& name) {
  if (name == "CPU time") return sim.cpu_time;
  if (name == "time") return sim.time;
  if (name == "dt") return sim.dt;
  if (name == "step") return double(sim.step);
  if (name == "elements") return double(sim.disc.mesh.nelems());
  if (sim.circuit.usingMesh) {
     auto const vdrop = sim.circuit.GetMeshVoltageDrop();
     auto const conductance = sim.circuit.GetMeshConductance();
     auto const current = sim.circuit.GetMeshCurrent();
     if (name == "mesh voltage")     return vdrop;
     if (name == "mesh current")     return current;
     if (name == "mesh conductance") return conductance;
  }
  for (auto it = sim.globals.data.begin(); it != sim.globals.data.end(); ++it) {
      if (name == (*it).name) return sim.globals.get(name);
  }
  auto it = by_name.find(name);
  if (it == by_name.end())
    Omega_h_fail("Request for undefined scalar \"%s\"\n", name.c_str());
  return (*it)->ask_value();
}
void Scalars::setup(Omega_h::InputMap& pl) {
  ::lgr::setup(sim.factories.scalar_factories, sim, pl, storage, "scalar");
  for (auto& ptr : storage) {
    by_name.insert(ptr.get());
  }
}

ScalarFactories get_builtin_scalar_factories() {
  ScalarFactories out;
  out["node"] = node_scalar_factory;
  out["L2 error"] = l2_error_factory;
  return out;
}

}  // namespace lgr
