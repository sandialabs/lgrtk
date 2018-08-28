#ifndef APPLY_WEIGHTING_HPP
#define APPLY_WEIGHTING_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Weighting functor.
  
    Given a voigt tensor and density, apply weighting to the voigt tensor.
    Assumes single point integration.
*/
/******************************************************************************/
template<int SpaceDim, int NumVoigtTerms, typename PenaltyFunction>
class ApplyWeighting : public Simplex<SpaceDim>
{
  private:
    using Simplex<SpaceDim>::m_numNodesPerCell;

    PenaltyFunction m_penaltyFunction;

  public:

    ApplyWeighting(PenaltyFunction penaltyFunction) :
      m_penaltyFunction(penaltyFunction) {}

    template<typename TensorScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<TensorScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& voigtTensor,
                Kokkos::View<WeightScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( int iNode=0; iNode<m_numNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/m_numNodesPerCell);
      for( int iVoigt=0; iVoigt<NumVoigtTerms; iVoigt++){
        voigtTensor(cellOrdinal,iVoigt) *= m_penaltyFunction(cellDensity);
      }
    }
    template<typename ResultScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> const& result,
                Plato::ScalarMultiVectorT<WeightScalarType> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( int iNode=0; iNode<m_numNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/m_numNodesPerCell);
      result(cellOrdinal) *= m_penaltyFunction(cellDensity);
    }
};

#endif
