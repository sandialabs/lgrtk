#ifndef PRESSURE_GRADIENT_HPP
#define PRESSURE_GRADIENT_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Pressure gradient functor.

    Given a gradient matrix, b, and state array, s, compute the pressure gradient, g.

    g_{i} = s_{e,I} b_{e,I,i}

        e:  element index
        I:  local node index
        i:  dimension index

*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class PressureGradient : public Plato::Simplex<SpaceDim>
{
  private:

    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;

  public:

    template<typename ResultScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< ResultScalarType   > const& pgrad,
                Plato::ScalarMultiVectorT< StateScalarType    > const& state,
                Plato::ScalarArray3DT<     GradientScalarType > const& gradient) const {

      // compute pgrad
      //
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        pgrad(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          pgrad(cellOrdinal,iDof) += state(cellOrdinal,iNode)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};

} // namespace Plato

#endif
