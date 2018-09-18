#ifndef LGR_SCALARS_HPP
#define LGR_SCALARS_HPP

#include <lgr_scalar.hpp>
#include <lgr_factories.hpp>
#include <vector>
#include <memory>
#include <Omega_h_rbtree.hpp>

namespace lgr {

struct NameOfScalarPtr {
  std::string const& operator()(Scalar* ptr);
};

struct Scalars {
  Simulation& sim;
  std::vector<std::unique_ptr<Scalar>> storage;
  Omega_h::rb_tree<std::string, Scalar*, NameOfScalarPtr> by_name;
  Scalars(Simulation& sim_in);
  void setup(Teuchos::ParameterList& pl);
  double ask_value(std::string const& name);
};

ScalarFactories get_builtin_scalar_factories();

}

#endif
