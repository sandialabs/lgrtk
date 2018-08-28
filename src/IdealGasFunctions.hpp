#include "ErrorHandling.hpp"  // get function LGR_THROW_IF()
#include "LGR_Types.hpp"    // get data type lgr::Scalar
#include "Omega_h_matrix.hpp" // get function Omega_h::max2()

using namespace lgr;

namespace IdealGasFunctions {


  KOKKOS_INLINE_FUNCTION
  Scalar hyperelasticCauchyStress(
                    // State
                    Scalar internal_energy,
                    Scalar mass_density,
                    // User Inputs
                    Scalar gamma
                   ) {

    const Scalar p = (gamma - 1.0) * mass_density * internal_energy;

/*
    LGR_THROW_IF(p < 0.0,
      "Ideal Gas returned a negative pressure. Diagnostics:"
      << "\ninternal_energy: " << internal_energy
      << "\nmass_density: " << mass_density
      << "\ngamma: " << gamma
      << "\npressure: " << p
      << "\n"
    );
*/

    return p;
  }


  KOKKOS_INLINE_FUNCTION
  Scalar waveModuli(// State
                    Scalar internal_energy,
                    Scalar mass_density,
                    // User Inputs
                    Scalar gamma
                   ) {

    /*
      K = rho*c^2 = rho * (gamma*p/rho) = gamma*p
    */

//    const Scalar p = IdealGasFunctions::hyperelasticCauchyStress(internal_energy,
//                                              mass_density,
//                                              gamma);
    const Scalar p = (gamma - 1.0) * mass_density * Omega_h::max2(internal_energy, 1.0e-14);
    const Scalar K = gamma * p;

    // The test for negative pressure in 'hyperelasticCauchyStress()' will
    // catch negative bulk modulii as well.

    return K;
  }

}
