#ifndef PLATO_PARSE_TOOLS
#define PLATO_PARSE_TOOLS


namespace Plato {

namespace ParseTools {

/**************************************************************************//**
 * @brief Get a parameter from a sublist if it exists, otherwise return the default.
 * @tparam T Type of the requested parameter.
 * @param [in] aInputParams The containing ParameterList
 * @param [in] aSubListName The name of the sublist within aInputParams
 * @param [in] aParamName The name of the desired parameter
 * @return The requested parameter value if it exists, otherwise the default
 *****************************************************************************/

template < typename T >
T getSubParam(
    Teuchos::ParameterList& aInputParams,
    const std::string aSubListName,
    const std::string aParamName,
    T aDefaultValue )
{
    if( aInputParams.isSublist(aSubListName) == true )
    {
        return aInputParams.sublist(aSubListName).get<T>(aParamName, aDefaultValue);
    }
    else
    {
        return aDefaultValue;
    }
}

} // namespace ParseTools

} // namespace Plato

#endif
