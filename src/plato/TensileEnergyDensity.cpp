/*
 * TensileEnergyDensity.cpp
 *
 */

#include "plato/TensileEnergyDensity.hpp"

#ifdef PLATO_1D
template class Plato::TensileEnergyDensity<1>;
#endif

#ifdef PLATO_2D
template class Plato::TensileEnergyDensity<2>;
#endif

#ifdef PLATO_3D
template class Plato::TensileEnergyDensity<3>;
#endif