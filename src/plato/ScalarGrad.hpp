#ifndef SCALAR_GRAD_HPP
#define SCALAR_GRAD_HPP

#include "plato/PlatoStaticsTypes.hpp"
#include <Omega_h_vector.hpp>

namespace Plato
{

/******************************************************************************/
/*! Scalar gradient functor.
  
    given a gradient matrix and scalar field, compute the scalar gradient.
*/
/******************************************************************************/
template<int SpaceDim>
class ScalarGrad
{
  private:
    static constexpr auto mNumNodesPerCell = SpaceDim+1;
    static constexpr auto mNumDofsPerCell  = mNumNodesPerCell;


  public:

    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace>  sgrad,
                Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace>  s,
                Omega_h::Vector<SpaceDim>* gradient) const {

      // compute scalar gradient
      //
      for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        sgrad(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType iDof=0; iDof<mNumDofsPerCell; iDof++){
          sgrad(cellOrdinal,iDim) += s(cellOrdinal,iDof)*gradient[iDof][iDim];
        }
      }
    }

    template<typename ScalarGradType, typename ScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< ScalarGradType     > sgrad,
                Plato::ScalarMultiVectorT< ScalarType         > s,
                Plato::ScalarArray3DT<     GradientScalarType > gradient) const {

      for(Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        sgrad(cellOrdinal,iDim)=0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          sgrad(cellOrdinal,iDim) += s(cellOrdinal,iNode)*gradient(cellOrdinal,iNode,iDim);
        }
      }
    }
};
// class ScalarGrad

} // namespace Plato

#endif
