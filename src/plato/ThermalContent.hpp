#ifndef THERMAL_CONTENT_HPP
#define THERMAL_CONTENT_HPP

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermal content functor.
  
    given a temperature value, compute the thermal content
*/
/******************************************************************************/
class ThermalContent
{
  private:
    const Plato::Scalar mCellDensity;
    const Plato::Scalar mCellSpecificHeat;

  public:
    ThermalContent(Plato::Scalar cellDensity, Plato::Scalar cellSpecificHeat) :
            mCellDensity(cellDensity),
            mCellSpecificHeat(cellSpecificHeat) {}

    template<typename TScalarType, typename TContentScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<TContentScalarType> tcontent,
                Plato::ScalarVectorT<TScalarType> temperature) const {

      // compute thermal content
      //
      tcontent(cellOrdinal) = temperature(cellOrdinal)*mCellDensity*mCellSpecificHeat;
    }
};
// class ThermalContent

} // namespace Plato

#endif
