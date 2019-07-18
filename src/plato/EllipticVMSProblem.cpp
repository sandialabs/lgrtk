#include "EllipticVMSProblem.hpp"

#ifdef PLATO_1D
template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<1>>;
//template class Plato::EllipticVMSProblem<::Plato::Electromechanics<1>>;
template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<2>>;
//template class Plato::EllipticVMSProblem<::Plato::Electromechanics<2>>;
template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<3>>;
//template class Plato::EllipticVMSProblem<::Plato::Electromechanics<3>>;
template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<3>>;
#endif
