#include <lgr_csv_hist.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>

namespace lgr {

struct CsvHist : public Response {
  std::vector<std::string> scalars;
  std::ofstream stream;
  std::string path;
  CsvHist(Simulation& sim_in, Omega_h::InputMap& pl) : Response(sim_in, pl) {
    auto& scalars_in = pl.get_list("scalars");
    for (int i = 0; i < scalars_in.size(); ++i) {
      scalars.push_back(scalars_in.get<std::string>(i));
    }
    path = pl.get<std::string>("path", "lgr_out.csv");
    auto write_units = pl.get<bool>("units","false");
    stream.open(path.c_str(),std::ios_base::out);
    OMEGA_H_CHECK(stream.is_open());
    // Headers
    std::stringstream header_stream;
    for (std::size_t i = 0; i < scalars.size(); ++i) {
      if (i) header_stream << ",";
      header_stream << scalars[i];
    }
    header_stream << '\n';
    auto header_string = header_stream.str();
    stream.write(header_string.data(), std::streamsize(header_string.length()));
    // Units placeholder (for now show "ND" for not defined)
    if (write_units) {
      std::stringstream unit_stream;
      for (std::size_t i = 0; i < scalars.size(); ++i) {
        if (i) unit_stream << ",";
        unit_stream << "ND";
      }
      unit_stream << '\n';
      auto unit_string = unit_stream.str();
      stream.write(unit_string.data(), std::streamsize(unit_string.length()));
    }
    // Close
    stream.close();
  }
  void respond() override final {
    if (sim.no_output) return;
    std::stringstream step_stream;
    step_stream << std::scientific << std::setprecision(17);
    for (std::size_t i = 0; i < scalars.size(); ++i) {
      if (i) step_stream << ",";
      auto val = sim.scalars.ask_value(scalars[i]);
      step_stream << val;
    }
    step_stream << '\n';
    auto step_string = step_stream.str();
    stream.open(path.c_str(), std::ios_base::out | std::ios_base::app);
    stream.write(step_string.data(), std::streamsize(step_string.length()));
    stream.close();
  }
  void out_of_line_virtual_method() override;
};

void CsvHist::out_of_line_virtual_method() {}

Response* csv_hist_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new CsvHist(sim, pl);
}

}  // namespace lgr
