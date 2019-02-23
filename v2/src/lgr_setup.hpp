#pragma once

#include <functional>
#include <vector>

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

struct Setups {
  using Vector = std::vector<std::function<void(Simulation&, Omega_h::InputMap&)>>;
  Vector material_models;
  Vector modifiers;
  Vector responses;
};

void add_builtin_setups(Setups&);

}
