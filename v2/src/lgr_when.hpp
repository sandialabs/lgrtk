#ifndef LGR_WHEN_HPP
#define LGR_WHEN_HPP

#include <Omega_h_teuchos.hpp>

namespace lgr {

struct When {
  virtual ~When() = default;
  virtual void out_of_line_virtual_method();
  virtual double next_event(double time) = 0;
  virtual bool active(double prev_time, double time) = 0;
};

When* time_periodic(double period);
When* time_range(double start, double end);
When* at_time(double time);
When* always();
When* never();

When* setup_when(Teuchos::ParameterList& pl);

}

#endif
