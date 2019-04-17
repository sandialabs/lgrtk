/*
 * Plato_CenterGravityCriterion.cpp
 *
 *  Created on: Apr 17, 2019
 */

#include "plato/Plato_CenterGravityCriterion.hpp"

#ifdef PLATO_1D
template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::CenterGravityCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::CenterGravityCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::CenterGravityCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::CenterGravityCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
