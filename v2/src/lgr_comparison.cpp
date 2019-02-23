#include <lgr_comparison.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct Comparison : public Response {
  std::string scalar;
  Omega_h::ExprEnv env;
  std::shared_ptr<Omega_h::ExprOp> op;
  double tolerance;
  double floor;
  Comparison(
      Simulation& sim_in, Omega_h::InputMap& pl)
      : Response(sim_in, pl), env(1, 1) {
    scalar = pl.get<std::string>("scalar");
    tolerance = pl.get<double>("tolerance", "1e-10");
    floor = pl.get<double>("floor", "1e-10");
    auto str = pl.get<std::string>("expected value");
    Omega_h::ExprOpsReader reader;
    op = reader.read_ops(str);
  }
  void respond() override final {
    auto value = sim.scalars.ask_value(scalar);
    env.register_variable("t", Omega_h::any(sim.time));
    auto expected_value = Omega_h::any_cast<double>(op->eval(env));
    if (!Omega_h::are_close(value, expected_value, tolerance, floor)) {
      Omega_h_fail(
          "Comparison of %s value %.17e to %.17e with tolerance %.1e and "
          "floor %.1e failed!\n",
          scalar.c_str(), value, expected_value, tolerance,
          floor);
    }
  }
  void out_of_line_virtual_method() override;
};

void Comparison::out_of_line_virtual_method() {}

void setup_comparison(Simulation& sim, Omega_h::InputMap& pl) {
  auto& responses_pl = pl.get_list("responses");
  for (int i = 0; i < responses_pl.size(); ++i) {
    auto& response_pl = responses_pl.get_map(i);
    if (response_pl.get<std::string>("type") == "comparison") {
      sim.responses.add(new Comparison(sim, response_pl));
    }
  }
}

}  // namespace lgr
