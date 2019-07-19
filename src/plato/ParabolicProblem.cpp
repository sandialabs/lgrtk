#include "ParabolicProblem.hpp"

#ifdef PLATO_1D
template class ParabolicProblem<::Plato::Thermal<1>>;
#endif
#ifdef PLATO_2D
template class ParabolicProblem<::Plato::Thermal<2>>;
#endif
#ifdef PLATO_3D
template class ParabolicProblem<::Plato::Thermal<3>>;
#endif
