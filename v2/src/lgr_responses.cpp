#include <Omega_h_profile.hpp>
#include <lgr_osh_output.hpp>
#include <lgr_responses.hpp>
#include <lgr_simulation.hpp>
#include <lgr_vtk_output.hpp>

namespace lgr {

Responses::Responses(Simulation& sim_in) : sim(sim_in) {}

void Responses::setup(Omega_h::InputList& pl) {
  ::lgr::setup(sim.factories.response_factories, sim, pl, storage, "response");
}

void Responses::add(Response* new_response) {
  std::unique_ptr<Response> uptr(new_response);
  storage.push_back(std::move(uptr));
}

void Responses::evaluate() {
  Omega_h::ScopedTimer timer("Responses::evaluate");
  for (auto& response : storage) {
    if (!response->when->active(sim.prev_time, sim.time)) continue;
    response->respond();
  }
}

double Responses::next_event(double time) {
  double out = std::numeric_limits<double>::max();
  for (auto& response : storage) {
    out = Omega_h::min2(out, response->when->next_event(time));
  }
  return out;
}

ResponseFactories get_builtin_response_factories() {
  ResponseFactories out;
  out["VTK output"] = vtk_output_factory;
  out["osh output"] = osh_output_factory;
  out["checkpoint"] = osh_output_factory;
  return out;
}

}  // namespace lgr
