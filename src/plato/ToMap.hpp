#ifndef PLATO_TO_MAP_HPP
#define PLATO_TO_MAP_HPP

#include <string>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

template <typename InputType>
inline void toMap(Plato::DataMap& aDataMap, InputType aInput, std::string aEntryName)
{
  // don't add to map
}

template <>
inline void 
toMap(Plato::DataMap& aDataMap, Plato::ScalarVector aInput, std::string aEntryName)
{
    aDataMap.scalarVectors[aEntryName] = aInput;
}

template <>
inline void 
toMap(Plato::DataMap& aDataMap, Plato::ScalarMultiVector aInput, std::string aEntryName)
{
    aDataMap.scalarMultiVectors[aEntryName] = aInput;
}

template <>
inline void 
toMap(Plato::DataMap& aDataMap, Plato::ScalarArray3D aInput, std::string aEntryName)
{
    aDataMap.scalarArray3Ds[aEntryName] = aInput;
}

} // end namespace Plato

#endif
