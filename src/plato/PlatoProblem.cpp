#include "PlatoProblem.hpp"

#ifdef PLATO_1D
template class Problem<::Plato::Thermal<1>>;
template class Problem<::Plato::Mechanics<1>>;
template class Problem<::Plato::Electromechanics<1>>;
template class Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Problem<::Plato::Thermal<2>>;
template class Problem<::Plato::Mechanics<2>>;
template class Problem<::Plato::Electromechanics<2>>;
template class Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Problem<::Plato::Thermal<3>>;
template class Problem<::Plato::Mechanics<3>>;
template class Problem<::Plato::Electromechanics<3>>;
template class Problem<::Plato::Thermomechanics<3>>;
#endif
