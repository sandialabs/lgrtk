#ifndef THERMAL_CONTENT_HPP
#define THERMAL_CONTENT_HPP

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Thermal content functor.
  
    given a temperature value, compute the thermal content
*/
/******************************************************************************/
class ThermalContent
{
  private:
    const Plato::Scalar m_cellDensity;
    const Plato::Scalar m_cellSpecificHeat;

  public:
    ThermalContent(Plato::Scalar cellDensity, Plato::Scalar cellSpecificHeat) :
            m_cellDensity(cellDensity),
            m_cellSpecificHeat(cellSpecificHeat) {}

    template<typename TScalarType, typename TContentScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<TContentScalarType**, Kokkos::LayoutRight, Plato::MemSpace> tcontent,
                Kokkos::View<TScalarType**, Kokkos::LayoutRight, Plato::MemSpace> temperature) const {

      // compute thermal content
      //
      tcontent(cellOrdinal,0) = temperature(cellOrdinal,0)*m_cellDensity*m_cellSpecificHeat;
    }
};
#endif
