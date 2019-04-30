#include "InternalThermalEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, SimplexThermal, 1)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, SimplexThermal, 2)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, SimplexThermal, 3)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, SimplexThermal, 3)
#endif
