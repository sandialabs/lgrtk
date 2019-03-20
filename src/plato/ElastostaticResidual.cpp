#include "plato/ElastostaticResidual.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::ElastostaticResidual, Plato::SimplexMechanics, 1)
#endif
#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::ElastostaticResidual, Plato::SimplexMechanics, 2)
#endif
#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::ElastostaticResidual, Plato::SimplexMechanics, 3)
#endif
