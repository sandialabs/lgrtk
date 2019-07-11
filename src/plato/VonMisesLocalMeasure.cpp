/*
 * VonMisesLocalMeasure.cpp
 *
 */

#include "plato/VonMisesLocalMeasure.hpp"

#ifdef PLATO_1D
template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
