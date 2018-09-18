#include <lgr_scalar.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

Scalar::Scalar(Simulation& sim_in, std::string const& name_in)
  :name(name_in)
  ,sim(sim_in)
  ,value_(std::numeric_limits<double>::quiet_NaN())
  ,cached_time_(std::numeric_limits<double>::quiet_NaN())
{}

void Scalar::out_of_line_virtual_method() {}

double Scalar::ask_value() {
  if (sim.time != cached_time_) {
    value_ = this->compute_value();
  }
  return value_;
}

}
