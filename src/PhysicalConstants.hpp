#ifndef LGR_PHYSICAL_CONSTANTS_HPP
#define LGR_PHYSICAL_CONSTANTS_HPP

#include "LGR_Types.hpp"

//SI units only!
namespace PhysicalConstants {
  using Scalar = lgr::Scalar;
  constexpr Scalar VacuumSOL = 299792458.0;
  constexpr Scalar VacuumMu  = 4.0 * 3.14159265358979323846 * 1.e-7;
  constexpr Scalar VacuumEps = 1.0 / ( (VacuumMu*VacuumSOL) * VacuumSOL );
}

#endif
