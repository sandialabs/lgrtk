#include "plato/TemperatureAverage.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::TemperatureAverageInc, SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::TemperatureAverageInc, SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::TemperatureAverageInc, SimplexThermal, 3)
#endif
