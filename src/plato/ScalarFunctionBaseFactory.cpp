
#include "ScalarFunctionBaseFactory.hpp"
#include "ScalarFunctionBaseFactory_def.hpp"


#ifdef PLATO_1D
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<1>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<1>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif

#ifdef PLATO_2D
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<2>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<2>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif

#ifdef PLATO_3D
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Mechanics<3>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Electromechanics<3>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
template class Plato::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
