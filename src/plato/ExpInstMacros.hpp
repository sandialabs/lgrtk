#ifndef EXP_INST_MACROS_HPP
#define EXP_INST_MACROS_HPP

#define PLATO_EXPL_DEF_INC(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEC_INC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEF(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;


#define PLATO_EXPL_DEC_INC_LOCAL(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEF_INC_LOCAL(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEC2(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D> >; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D> >; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D> >; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D> >;

#define PLATO_EXPL_DEF2(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, T<D> >; \
template class C<Plato::JacobianTypes<T<D>>, T<D> >; \
template class C<Plato::GradientXTypes<T<D>>, T<D> >; \
template class C<Plato::GradientZTypes<T<D>>, T<D> >;

#endif
