/*
 * TensileEnergyDensityLocalMeasure.cpp
 *
 */

#include "plato/TensileEnergyDensityLocalMeasure.hpp"

#ifdef PLATO_1D
template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
