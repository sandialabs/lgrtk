#include "plato/EffectiveEnergy.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::EffectiveEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::EffectiveEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::EffectiveEnergy, Plato::SimplexMechanics, 3)
#endif
