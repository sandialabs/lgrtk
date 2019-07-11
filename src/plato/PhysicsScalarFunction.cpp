#include "plato/PhysicsScalarFunction.hpp"

#ifdef PLATO_1D
template class Plato::PhysicsScalarFunction<::Plato::Thermal<1>>;
template class Plato::PhysicsScalarFunction<::Plato::Mechanics<1>>;
template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<1>>;
template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::PhysicsScalarFunction<::Plato::Thermal<2>>;
template class Plato::PhysicsScalarFunction<::Plato::Mechanics<2>>;
template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<2>>;
template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::PhysicsScalarFunction<::Plato::Thermal<3>>;
template class Plato::PhysicsScalarFunction<::Plato::Mechanics<3>>;
template class Plato::PhysicsScalarFunction<::Plato::Electromechanics<3>>;
template class Plato::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif
