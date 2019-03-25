#include "plato/EMStressPNorm.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(EMStressPNorm, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(EMStressPNorm, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(EMStressPNorm, Plato::SimplexElectromechanics, 3)
#endif
