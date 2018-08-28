#ifndef SCALAR_GRAD_HPP
#define SCALAR_GRAD_HPP

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Scalar gradient functor.
  
    given a gradient matrix and scalar field, compute the scalar gradient.
*/
/******************************************************************************/
template<int SpaceDim>
class ScalarGrad
{
  private:
    static constexpr auto m_numNodesPerCell = SpaceDim+1;
    static constexpr auto m_numDofsPerCell  = m_numNodesPerCell;


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
        for( Plato::OrdinalType iDof=0; iDof<m_numDofsPerCell; iDof++){
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
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          sgrad(cellOrdinal,iDim) += s(cellOrdinal,iNode)*gradient(cellOrdinal,iNode,iDim);
        }
      }
    }
};
#endif
