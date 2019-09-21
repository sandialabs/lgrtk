#ifndef THERMAL_FLUX_HPP
#define THERMAL_FLUX_HPP

#include "plato/PlatoStaticsTypes.hpp"
#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Thermal flux functor.
  
    given a temperature gradient, compute the thermal flux
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ThermalFlux
{
  private:
    static constexpr auto mNumNodesPerCell = SpaceDim+1;
    static constexpr auto mNumDofsPerCell  = mNumNodesPerCell;

    const Omega_h::Matrix<SpaceDim,SpaceDim> mCellConductivity;

  public:

    ThermalFlux( const Omega_h::Matrix<SpaceDim,SpaceDim> cellConductivity) :
            mCellConductivity(cellConductivity) {}

    template<typename TGradScalarType, typename TFluxScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<TFluxScalarType**, Kokkos::LayoutRight, Plato::MemSpace> tflux,
                Kokkos::View<TGradScalarType**, Kokkos::LayoutRight, Plato::MemSpace> tgrad) const {

      // compute thermal flux
      //
      for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        tflux(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpaceDim; jDim++){
          tflux(cellOrdinal,iDim) += tgrad(cellOrdinal,jDim)*mCellConductivity[iDim][jDim];
        }
      }
    }
};
// class ThermalFlux

} // namespace Plato
#endif
