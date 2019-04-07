#ifndef PLATO_TO_MAP_HPP
#define PLATO_TO_MAP_HPP

#include <string>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Null operation for Sacado Types
 * @param [in/out] aDataMap output data storage
 * @param [in] aInput Sacado-type container
 * @param [in] aEntryName output data name
 **********************************************************************************/
template<typename InputType>
inline void toMap(Plato::DataMap& aDataMap, const InputType aInput, const std::string & aEntryName)
{
    // don't add to map
}
// function toMap

/******************************************************************************//**
 * @brief Store 1D container in data map. The data map is used for output purposes
 * @param [in/out] aDataMap output data storage
 * @param [in] aInput 1D container
 * @param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void toMap(Plato::DataMap& aDataMap, const Plato::ScalarVector aInput, const std::string & aEntryName)
{
    aDataMap.scalarVectors[aEntryName] = aInput;
}
// function toMap

/******************************************************************************//**
 * @brief Store 2D container in data map. The data map is used for output purposes
 * @param [in/out] aDataMap output data storage
 * @param [in] aInput 2D container
 * @param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void toMap(Plato::DataMap& aDataMap, const Plato::ScalarMultiVector aInput, const std::string & aEntryName)
{
    aDataMap.scalarMultiVectors[aEntryName] = aInput;
}
// function toMap

/******************************************************************************//**
 * @brief Store 3D container in data map. The data map is used for output purposes
 * @param [in/out] aDataMap output data storage
 * @param [in] aInput 3D container
 * @param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void toMap(Plato::DataMap& aDataMap, const Plato::ScalarArray3D aInput, const std::string & aEntryName)
{
    aDataMap.scalarArray3Ds[aEntryName] = aInput;
}
// function toMap

}// end namespace Plato

#endif
