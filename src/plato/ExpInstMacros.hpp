#ifndef EXP_INST_MACROS_HPP
#define EXP_INST_MACROS_HPP

#define PLATO_EXPL_DEF_INC(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, SIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, SIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Heaviside >; \
template class C<Plato::JacobianPTypes<T<D>>, SIMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, SIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, SIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Heaviside >;

#define PLATO_EXPL_DEC_INC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, SIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, SIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Heaviside >; \
extern template class C<Plato::JacobianPTypes<T<D>>, SIMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, SIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, SIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Heaviside >;

#define PLATO_EXPL_DEF(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, SIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, SIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, SIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, SIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Heaviside >;

#define PLATO_EXPL_DEC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, SIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, SIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, SIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, SIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Heaviside >;

#endif
