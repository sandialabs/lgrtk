#include "plato/InternalElectroelasticEnergy.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(InternalElectroelasticEnergy, Plato::SimplexElectromechanics, 3)
#endif
