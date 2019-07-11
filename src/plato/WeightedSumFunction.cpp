#include "plato/WeightedSumFunction.hpp"

#ifdef PLATO_1D
template class Plato::WeightedSumFunction<::Plato::Thermal<1>>;
template class Plato::WeightedSumFunction<::Plato::Mechanics<1>>;
template class Plato::WeightedSumFunction<::Plato::Electromechanics<1>>;
template class Plato::WeightedSumFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATO_2D
template class Plato::WeightedSumFunction<::Plato::Thermal<2>>;
template class Plato::WeightedSumFunction<::Plato::Mechanics<2>>;
template class Plato::WeightedSumFunction<::Plato::Electromechanics<2>>;
template class Plato::WeightedSumFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATO_3D
template class Plato::WeightedSumFunction<::Plato::Thermal<3>>;
template class Plato::WeightedSumFunction<::Plato::Mechanics<3>>;
template class Plato::WeightedSumFunction<::Plato::Electromechanics<3>>;
template class Plato::WeightedSumFunction<::Plato::Thermomechanics<3>>;
#endif
