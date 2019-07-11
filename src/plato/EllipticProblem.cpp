#include "EllipticProblem.hpp"

#ifdef PLATO_1D
template class Plato::EllipticProblem<::Plato::Thermal<1>>;
template class Plato::EllipticProblem<::Plato::Mechanics<1>>;
template class Plato::EllipticProblem<::Plato::Electromechanics<1>>;
template class Plato::EllipticProblem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::EllipticProblem<::Plato::Thermal<2>>;
template class Plato::EllipticProblem<::Plato::Mechanics<2>>;
template class Plato::EllipticProblem<::Plato::Electromechanics<2>>;
template class Plato::EllipticProblem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::EllipticProblem<::Plato::Thermal<3>>;
template class Plato::EllipticProblem<::Plato::Mechanics<3>>;
template class Plato::EllipticProblem<::Plato::Electromechanics<3>>;
template class Plato::EllipticProblem<::Plato::Thermomechanics<3>>;
#endif
