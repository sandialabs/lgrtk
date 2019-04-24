#include "plato/InternalThermoelasticEnergy.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 1)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 2)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 3)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 3)
#endif
