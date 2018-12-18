#ifndef LGR_SCALAR_HPP
#define LGR_SCALAR_HPP

#include <string>

namespace lgr {

struct Simulation;

struct Scalar {
  std::string name;
  Scalar(Simulation& sim_in, std::string const& name_in);
  virtual ~Scalar() = default;
  virtual void out_of_line_virtual_method();
  double ask_value();

 protected:
  Simulation& sim;
  virtual double compute_value() = 0;

 private:
  double value_;
  double cached_time_;
};

}  // namespace lgr

#endif
