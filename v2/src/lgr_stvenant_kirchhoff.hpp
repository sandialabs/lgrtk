#ifndef LGR_STVENANT_KIRCHHOFF_HPP
#define LGR_STVENANT_KIRCHHOFF_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

OMEGA_H_INLINE void stvenant_kirchhoff_update(
    double bulk_modulus,
    double shear_modulus,
    double density,
    Matrix<3, 3> F,
    Matrix<3, 3>& stress,
    double& wave_speed) {
  OMEGA_H_CHECK(density > 0.0);
  auto const J = Omega_h::determinant(F);
  OMEGA_H_CHECK(J > 0.0);
  auto const Jinv = 1.0 / J;
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const C = transpose(F) * F;
  auto const E = 0.5 * (C - I);
  auto const mu = shear_modulus;
  auto const lambda = bulk_modulus - 2.0 * mu / 3.0;
  auto const S = lambda * trace(E) * I + 2.0 * mu * E;
  stress = Jinv * F * S * transpose(F);
  wave_speed = std::sqrt(bulk_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

template <class Elem>
ModelBase* stvenant_kirchhoff_factory(Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* stvenant_kirchhoff_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
