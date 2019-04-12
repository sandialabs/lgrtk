/*
 * Plato_AugLagStressCriterionGeneral.cpp
 *
 *  Created on: Apr 2, 2019
 */

#include "Plato_AugLagStressCriterionGeneral.hpp"

#ifdef PLATO_1D
template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::AugLagStressCriterionGeneral<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterionGeneral<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
