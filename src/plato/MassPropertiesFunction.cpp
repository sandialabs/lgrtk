#include "plato/MassPropertiesFunction.hpp"

#ifdef PLATO_1D
template class Plato::MassPropertiesFunction<::Plato::Thermal<1>>;
template class Plato::MassPropertiesFunction<::Plato::Mechanics<1>>;
template class Plato::MassPropertiesFunction<::Plato::Electromechanics<1>>;
template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::MassPropertiesFunction<::Plato::Thermal<2>>;
template class Plato::MassPropertiesFunction<::Plato::Mechanics<2>>;
template class Plato::MassPropertiesFunction<::Plato::Electromechanics<2>>;
template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::MassPropertiesFunction<::Plato::Thermal<3>>;
template class Plato::MassPropertiesFunction<::Plato::Mechanics<3>>;
template class Plato::MassPropertiesFunction<::Plato::Electromechanics<3>>;
template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<3>>;
#endif
