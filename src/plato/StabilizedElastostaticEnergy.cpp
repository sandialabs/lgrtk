#include "StabilizedElastostaticEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 3)
#endif
