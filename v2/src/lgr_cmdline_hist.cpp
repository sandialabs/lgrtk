#include <lgr_cmdline_hist.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct CmdLineHist : public Response {
  std::vector<std::string> scalars;
  std::size_t column_width;
  CmdLineHist(Simulation& sim_in, Omega_h::InputMap& pl)
      : Response(sim_in, pl) {
    auto& scalars_in = pl.get_list("scalars");
    for (int i = 0; i < scalars_in.size(); ++i) {
      auto scalar = scalars_in.get<std::string>(i);
      scalars.push_back(scalar);
    }
    column_width =
        decltype(column_width)(pl.get<int>("minimum column width", "8"));
    for (auto& name : scalars) {
      column_width = Omega_h::max2(column_width, name.length());
    }
    column_width += 2;
    if (sim.no_output) return;
    for (auto& name : scalars) {
      std::printf(" %*s ", int(column_width - 2), name.c_str());
    }
    std::printf("\n");
  }
  void respond() override final {
    if (sim.no_output) return;
    for (auto& name : scalars) {
      auto val = sim.scalars.ask_value(name);
      if (val < 0.0)
        std::printf(
            " %*.*e ", int(column_width - 2), int(column_width - 9), val);
      else
        std::printf(
            " +%*.*e ", int(column_width - 3), int(column_width - 9), val);
    }
    std::printf("\n");
  }
  void out_of_line_virtual_method() override;
};

void CmdLineHist::out_of_line_virtual_method() {}

void setup_cmdline_hist(Simulation& sim, Omega_h::InputMap& pl) {
  auto& responses_pl = pl.get_list("responses");
  for (int i = 0; i < responses_pl.size(); ++i) {
    auto& response_pl = responses_pl.get_map(i);
    if (response_pl.get<std::string>("type") == "command line history") {
      sim.responses.add(new CmdLineHist(sim, response_pl));
    }
  }
}

}  // namespace lgr
