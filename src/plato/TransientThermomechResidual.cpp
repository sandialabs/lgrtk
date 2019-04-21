#include "plato/TransientThermomechResidual.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 1)
#endif
#ifdef PLATO_2D
PLATO_EXPL_DEF_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 2)
#endif
#ifdef PLATO_3D
PLATO_EXPL_DEF_INC(TransientThermomechResidual, Plato::SimplexThermomechanics, 3)
#endif
