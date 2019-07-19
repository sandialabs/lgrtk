#include "plato/J2PlasticityLocalResidual.hpp"

#ifdef PLATO_2D
PLATO_EXPL_DEF_INC_LOCAL(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 2)
//PLATO_EXPL_DEF_INC_LOCAL(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 2)
#endif
#ifdef PLATO_3D
//PLATO_EXPL_DEF_INC_LOCAL(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 3)
//PLATO_EXPL_DEF_INC_LOCAL(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 3)
#endif