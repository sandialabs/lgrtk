/*
 * Plato_AugLagStressCriterion.cpp
 *
 *  Created on: Apr 2, 2019
 */

#include "Plato_AugLagStressCriterion.hpp"

#ifdef PLATO_1D
template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
