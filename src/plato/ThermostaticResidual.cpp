#include "plato/ThermostaticResidual.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::ThermostaticResidual, SimplexThermal, 1)
#endif
#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::ThermostaticResidual, SimplexThermal, 2)
#endif
#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::ThermostaticResidual, SimplexThermal, 3)
#endif
