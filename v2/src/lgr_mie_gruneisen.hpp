#ifndef LGR_MIE_GRUNEISEN_HPP
#define LGR_MIE_GRUNEISEN_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>
#ifndef OMEGA_H_THROW
#include <exception>
#endif

namespace lgr {

namespace mie_gruneisen_details {

OMEGA_H_INLINE
void
read_and_validate_params(Teuchos::ParameterList& pl,
    double& rho0, double& gamma0, double& c0, double& s1, double& e0)
{
  auto errors = 0;
  auto os = std::ostringstream();

  // Read and validate inputs
  if (!pl.isParameter("rho0")) {
    errors++;
    os << "Mie Gruneisen model requires 'initial density' parameter\n";
  }
  else {
    rho0 = pl.get<double>("rho0");
  }

  if (!pl.isParameter("gamma0")) {
    errors++;
    os << "Mie Gruneisen model requires 'gamma0' parameter\n";
  }
  else {
    gamma0 = pl.get<double>("gamma0");
  }

  if (!pl.isParameter("c0")) {
    errors++;
    os << "Mie Gruneisen model requires 'c0' parameter\n";
  }
  else {
    c0 = pl.get<double>("c0");
  }

  if (!pl.isParameter("s1")) {
    errors++;
    os << "Mie Gruneisen model requires 's1' parameter\n";
  }
  else {
    s1 = pl.get<double>("s1");
  }

  if (!pl.isParameter("e0")) {
    errors++;
    os << "Mie Gruneisen model requires 'e0' parameter\n";
  }
  else {
    e0 = pl.get<double>("e0");
  }

  if (errors != 0) {
#ifdef OMEGA_H_THROW
    throw Omega_h::exception(os.str());
#else
    throw std::invalid_argument(os.str());
#endif
  }
}
} // mie_gruneisen_details

//
// This implementation of Mie Gruneisen is based on the hugoniot
// relations. In the code, you'll see references to 'ph' and 'eh'
// which are the  pressure and energy on the hugoniot, respectively.
//
// The value of 'ph' the hugoniot pressure is found by assuming a
// linear us-up relation:
//
//                     u_s = c_0 + s * u_p
//
// Then, by using the hugoniot equations for the conseration of mass
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


OMEGA_H_INLINE void mie_gruneisen_update(
    double rho0, double gamma0, double c0, double s1,
    double rho, double internal_energy,
    double& pressure, double& wave_speed)
{

  const double mu = 1.0 - (rho0 / rho);
  const double dmu = rho0 / rho / rho; /* = \frac{\partial \mu}{\partial \rho} */

  // For expansion (mu <= 0):
  const bool compression = (mu > 0.);

  //  * limit 'us' to not drop below 'c0'
  const double us = c0 / (1.0 - s1 * Omega_h::max2(0.0, mu));
  const double ph = rho0 * us * us * mu;
  const double eh = 0.5 * ph * mu / rho0;

  // derivative of pressure with respect to density
  const double dus = compression ? (us * s1 / (1.0 - s1 * mu)) * dmu : 0.0;
  const double dph = 2.0 * rho0 * us * dus * mu + rho0 * us * us * dmu;
  const double deh = 0.5 * (mu * dph + ph * dmu) / rho0;
  const double dpdrho = dph - gamma0 * rho0 * deh;

  //derivative of pressure with repect to energy
  const double dpde = gamma0 * rho0;

  // Pressure
  pressure = ph + gamma0 * rho0 * (internal_energy - eh);

  // Wave speed
  const double bulk_modulus = rho * dpdrho + (pressure / rho) * dpde;
  wave_speed = std::sqrt(bulk_modulus / rho);
}

template <class Elem>
ModelBase* mie_gruneisen_factory(Simulation& sim, std::string const& name, Teuchos::ParameterList& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* mie_gruneisen_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
