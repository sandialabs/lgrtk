#include <lgr_cmdline_hist.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct CmdLineHist : public Response {
  std::vector<std::string> scalars;
  std::size_t column_width;
  CmdLineHist(Simulation& sim_in, Teuchos::ParameterList& pl)
    :Response(sim_in, pl)
  {
    auto scalars_teuchos = pl.get<Teuchos::Array<std::string>>("scalars");
    scalars.assign(scalars_teuchos.begin(), scalars_teuchos.end());
    column_width = 8;
    for (auto& name : scalars) {
      column_width = Omega_h::max2(column_width, name.length());
    }
    column_width += 2;
    for (auto& name : scalars) {
      std::printf(" %*s ", int(column_width - 2), name.c_str());
    }
    std::printf("\n");
  }
  void respond() override final {
    for (auto& name : scalars) {
      auto val = sim.scalars.ask_value(name);
      if (val < 0.0) std::printf(" %*.*e ", int(column_width - 2), int(column_width - 9), val);
      else std::printf(" +%*.*e ", int(column_width - 3), int(column_width - 9), val);
    }
    std::printf("\n");
  }
  void out_of_line_virtual_method() override;
};

void CmdLineHist::out_of_line_virtual_method() {}

Response* cmdline_hist_factory(Simulation& sim, std::string const&,
    Teuchos::ParameterList& pl)
{
  return new CmdLineHist(sim, pl);
}

}
