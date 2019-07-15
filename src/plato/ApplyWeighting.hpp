#ifndef APPLY_WEIGHTING_HPP
#define APPLY_WEIGHTING_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Weighting functor.
  
    Given a voigt tensor and density, apply weighting to the voigt tensor.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumVoigtTerms, typename PenaltyFunction>
class ApplyWeighting : public Plato::Simplex<SpaceDim>
{
  private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

    PenaltyFunction mPenaltyFunction;

  public:

    ApplyWeighting(PenaltyFunction penaltyFunction) :
      mPenaltyFunction(penaltyFunction) {}

    template<typename TensorScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<TensorScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& voigtTensor,
                Kokkos::View<WeightScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/mNumNodesPerCell);
      for( Plato::OrdinalType iVoigt=0; iVoigt<NumVoigtTerms; iVoigt++){
        voigtTensor(cellOrdinal,iVoigt) *= mPenaltyFunction(cellDensity);
      }
    }
    template<typename InputScalarType,
             typename OutputScalarType, 
             typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<InputScalarType> const& input,
                Plato::ScalarMultiVectorT<OutputScalarType> const& output,
                Plato::ScalarMultiVectorT<WeightScalarType> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/mNumNodesPerCell);
      for( Plato::OrdinalType iVoigt=0; iVoigt<NumVoigtTerms; iVoigt++){
        output(cellOrdinal,iVoigt) = mPenaltyFunction(cellDensity)*input(cellOrdinal, iVoigt);
      }
    }
    template<typename ResultScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> const& result,
                Plato::ScalarMultiVectorT<WeightScalarType> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/mNumNodesPerCell);
      result(cellOrdinal) *= mPenaltyFunction(cellDensity);
    }
    template<typename InputScalarType, 
             typename OutputScalarType, 
             typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<InputScalarType> const& input,
                Plato::ScalarVectorT<OutputScalarType>& output,
                Plato::ScalarMultiVectorT<WeightScalarType> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/mNumNodesPerCell);
      output(cellOrdinal) = mPenaltyFunction(cellDensity)*input(cellOrdinal);
    }
};
// class ApplyWeighting

} // namespace Plato

#endif
