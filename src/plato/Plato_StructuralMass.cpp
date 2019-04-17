/*
 * Plato_StructuralMass.cpp
 *
 *  Created on: Apr 17, 2019
 */

#include "plato/Plato_StructuralMass.hpp"

#ifdef PLATO_1D
template class Plato::StructuralMass<1>;
#endif

#ifdef PLATO_2D
template class Plato::StructuralMass<2>;
#endif

#ifdef PLATO_3D
template class Plato::StructuralMass<3>;
#endif
