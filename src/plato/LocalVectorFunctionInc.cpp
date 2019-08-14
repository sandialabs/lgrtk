#include "plato/LocalVectorFunctionInc.hpp"

#ifdef PLATO_2D
template class Plato::LocalVectorFunctionInc<Plato::Plasticity<2>>;
template class Plato::LocalVectorFunctionInc<Plato::ThermoPlasticity<2>>;
#endif
#ifdef PLATO_3D
template class Plato::LocalVectorFunctionInc<Plato::Plasticity<3>>;
template class Plato::LocalVectorFunctionInc<Plato::ThermoPlasticity<3>>;
#endif