
#include "MassMoment.hpp"

#ifdef PLATO_1D
template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif