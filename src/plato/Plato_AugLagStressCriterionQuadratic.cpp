/*
 * Plato_AugLagStressCriterionQuadratic.cpp
 *
 */

#include "Plato_AugLagStressCriterionQuadratic.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 1)
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 2)
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 3)
PLATO_EXPL_DEF2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 3)
#endif
