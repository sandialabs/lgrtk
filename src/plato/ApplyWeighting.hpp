#ifndef APPLY_WEIGHTING_HPP
#define APPLY_WEIGHTING_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Weighting functor.
  
    Given an input view and density, apply weighting to the input view.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumTerms, typename PenaltyFunction>
class ApplyWeighting : public Plato::Simplex<SpaceDim>
{
  private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

    PenaltyFunction mPenaltyFunction;

  public:

    ApplyWeighting(PenaltyFunction penaltyFunction) :
      mPenaltyFunction(penaltyFunction) {}

    template<typename InputScalarType, typename WeightScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<InputScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& inputOutput,
                Kokkos::View<WeightScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& rho) const {

      // apply weighting
      //
      WeightScalarType cellDensity = 0.0;
      for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
        cellDensity += rho(cellOrdinal, iNode);
      }
      cellDensity = (cellDensity/mNumNodesPerCell);
      for( Plato::OrdinalType iTerm=0; iTerm<NumTerms; iTerm++){
        inputOutput(cellOrdinal,iTerm) *= mPenaltyFunction(cellDensity);
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
      for( Plato::OrdinalType iTerm=0; iTerm<NumTerms; iTerm++){
        output(cellOrdinal,iTerm) = mPenaltyFunction(cellDensity)*input(cellOrdinal, iTerm);
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
