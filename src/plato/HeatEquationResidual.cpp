#include "plato/HeatEquationResidual.hpp"
#include "plato/ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif
#ifdef PLATO_2D
PLATO_EXPL_DEF_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif
#ifdef PLATO_3D
PLATO_EXPL_DEF_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif
