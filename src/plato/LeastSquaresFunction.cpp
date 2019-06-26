#include "plato/LeastSquaresFunction.hpp"

#ifdef PLATO_1D
template class Plato::LeastSquaresFunction<::Plato::Thermal<1>>;
template class Plato::LeastSquaresFunction<::Plato::Mechanics<1>>;
template class Plato::LeastSquaresFunction<::Plato::Electromechanics<1>>;
template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::LeastSquaresFunction<::Plato::Thermal<2>>;
template class Plato::LeastSquaresFunction<::Plato::Mechanics<2>>;
template class Plato::LeastSquaresFunction<::Plato::Electromechanics<2>>;
template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::LeastSquaresFunction<::Plato::Thermal<3>>;
template class Plato::LeastSquaresFunction<::Plato::Mechanics<3>>;
template class Plato::LeastSquaresFunction<::Plato::Electromechanics<3>>;
template class Plato::LeastSquaresFunction<::Plato::Thermomechanics<3>>;
#endif
