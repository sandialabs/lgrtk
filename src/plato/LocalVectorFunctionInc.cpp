#include "plato/LocalVectorFunctionInc.hpp"

#ifdef PLATO_2D
template class Plato::LocalVectorFunctionInc<Plato::SimplexPlasticity<2>>;
template class Plato::LocalVectorFunctionInc<Plato::SimplexThermoPlasticity<2>>;
#endif
#ifdef PLATO_3D
template class Plato::LocalVectorFunctionInc<Plato::SimplexPlasticity<3>>;
template class Plato::LocalVectorFunctionInc<Plato::SimplexThermoPlasticity<3>>;
#endif