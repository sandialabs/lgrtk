#ifndef LGR_MIE_GRUNEISEN_HPP
#define LGR_MIE_GRUNEISEN_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

//
// This implementation of Mie Gruneisen is based on the Hugoniot
// relations. In the code, you'll see references to 'ph' and 'eh'
// which are the  pressure and energy on the Hugoniot, respectively.
//
// The value of 'ph' the Hugoniot pressure is found by assuming a
// linear us-up relation:
//
//                     u_s = c_0 + s * u_p
//
// Then, by using the Hugoniot equations for the conseration of mass
// and momentum we get to:
//
//                             C_0^2
//   p_h = rho_0 * ------------------------------ * (1 - rho_0 / rho)
//                  (1 - s * (1 - rho_0 / rho))^2
//
// This assumes that the reference pressure is zero and the reference
// density is rho_0.
//
//
//
//
//            HUGONIOT CONSERVATION LAWS
//            --------------------------
//
// Conservation of mass:
// (1)  rho1 * us = rho2 * (us - u2)
// (2)         u2 = us * (1- rho1 / rho2)
//
// Conservation of momentum:
// (3)  p2 - p1 = rho1 * us * u2
//
// Conservation of energy:
// (4)  e2 - e1 = (p2 + p1) * (1 / rho1 - 1 / rho2) / 2
//
// Merging conservation of mass and momentum:
// (5)  p2 - p1 = rho1 * us^2 * (1 - rho1 / rho2)
//
//                    ASSUMPTIONS
//                    -----------
//
// Linear us-up relation - simplified using Equation (2):
// (6)  us = c0 + s * up = c0 + s * u2
// (7)  us = c0 / (1 - s * (1 - rho1 / rho2))
//
// The following substitutions are made to get to the final form:
//   * p2 = ph
//   * p1 = 0
//   * e2 = eh
//   * e1 = 0
//   * rho2 = rho
//   * rho1 = rho0
//
//                   FINAL FORMS
//                   -----------
//
// Reference Hugoniot Pressure - Equation (3) plus Equations (7) and (2)
// (8)  ph = rho0 * us^2 * (1 - rho0 / rho)
// (9)  ph = rho0 * c0^2 / (1 - s * (1 - rho0 / rho))^2 * (1 - rho0 / rho)
//
// Reference Hugoniot Energy - Equation (4)
// (10)  eh = ph * (1 / rho0 - 1 / rho ) / 2
//
// Wrapping it all together, we integrate the Gruneisen model to get and
// explicitly note the dependence of 'ph' and 'eh' on 'rho':
// (11)  p - ph(rho) = Gamma * rho * (e - eh(rho))
// (12)  p = ph(rho) + Gamma * rho * (e - eh(rho))
//

OMEGA_H_INLINE void mie_gruneisen_update(double const rho0, double const gamma0,
    double const c0, double const s1, double const rho,
    double const internal_energy, double& pressure, double& wave_speed) {
  auto const mu = 1.0 - (rho0 / rho);
  auto const dmu = rho0 / rho / rho; /* = \frac{\partial \mu}{\partial \rho} */
  // For expansion (mu <= 0):
  auto const compression = (mu > 0.);
  //  * limit 'us' to not drop below 'c0'
  auto const us = c0 / (1.0 - s1 * Omega_h::max2(0.0, mu));
  auto const ph = rho0 * us * us * mu;
  auto const eh = 0.5 * ph * mu / rho0;
  // derivative of pressure with respect to density
  auto const dus = compression ? (us * s1 / (1.0 - s1 * mu)) * dmu : 0.0;
  auto const dph = 2.0 * rho0 * us * dus * mu + rho0 * us * us * dmu;
  auto const deh = 0.5 * (mu * dph + ph * dmu) / rho0;
  auto const dpdrho = dph - gamma0 * rho0 * deh;
  // derivative of pressure with repect to energy
  auto const dpde = gamma0 * rho0;
  // Pressure
  pressure = ph + gamma0 * rho0 * (internal_energy - eh);
  // Wave speed
  auto const bulk_modulus = rho * dpdrho + (pressure / rho) * dpde;
  wave_speed = std::sqrt(bulk_modulus / rho);
}

template <class Elem>
ModelBase* mie_gruneisen_factory(
    Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem)                                                    \
  extern template ModelBase* mie_gruneisen_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
