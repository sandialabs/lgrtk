/*
 * ThermalVonMisesLocalMeasure.cpp
 *
 */

#include "plato/ThermalVonMisesLocalMeasure.hpp"


#ifdef PLATO_1D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 3)
#endif