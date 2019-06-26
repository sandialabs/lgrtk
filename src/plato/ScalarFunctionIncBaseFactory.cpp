
#include "ScalarFunctionIncBaseFactory.hpp"
#include "ScalarFunctionIncBaseFactory_def.hpp"


#ifdef PLATO_1D
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<1>>;
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<2>>;
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<3>>;
template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<3>>;
#endif