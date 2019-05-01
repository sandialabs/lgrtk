#include "InternalThermalEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, Plato::SimplexThermal, 1)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, Plato::SimplexThermal, 2)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::InternalThermalEnergy, Plato::SimplexThermal, 3)
PLATO_EXPL_DEF(Plato::InternalThermalEnergyInc, Plato::SimplexThermal, 3)
#endif
