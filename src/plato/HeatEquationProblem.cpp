#include "HeatEquationProblem.hpp"

#ifdef PLATO_1D
template class HeatEquationProblem<::Plato::Thermal<1>>;
#endif
#ifdef PLATO_2D
template class HeatEquationProblem<::Plato::Thermal<2>>;
#endif
#ifdef PLATO_3D
template class HeatEquationProblem<::Plato::Thermal<3>>;
#endif
