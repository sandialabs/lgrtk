#include <lgr_csv_hist.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

#include <fstream>
#include <sstream>
#include <iomanip>

namespace lgr {

struct CsvHist : public Response {
  std::vector<std::string> scalars;
  std::ofstream stream;
  CsvHist(Simulation& sim_in, Omega_h::InputMap& pl)
    :Response(sim_in, pl)
  {
    auto& scalars_in = pl.get_list("scalars");
    for (int i = 0; i < scalars_in.size(); ++i) {
      scalars.push_back(scalars_in.get<std::string>(i));
    }
    auto path = pl.get<std::string>("path", "lgr_out.csv");
    stream.open(path.c_str());
    OMEGA_H_CHECK(stream.is_open());
    std::stringstream header_stream;
    for (std::size_t i = 0; i < scalars.size(); ++i) {
      if (i) header_stream << ", ";
      header_stream << scalars[i];
    }
    header_stream << '\n';
    auto header_string = header_stream.str();
    stream.write(header_string.data(), std::streamsize(header_string.length()));
  }
  void respond() override final {
    std::stringstream step_stream;
    step_stream << std::scientific << std::setprecision(17);
    for (std::size_t i = 0; i < scalars.size(); ++i) {
      if (i) step_stream << ", ";
      auto val = sim.scalars.ask_value(scalars[i]);
      step_stream << val;
    }
    step_stream << '\n';
    auto step_string = step_stream.str();
    stream.write(step_string.data(), std::streamsize(step_string.length()));
  }
  void out_of_line_virtual_method() override;
};

void CsvHist::out_of_line_virtual_method() {}

Response* csv_hist_factory(Simulation& sim, std::string const&,
    Omega_h::InputMap& pl)
{
  return new CsvHist(sim, pl);
}

}

