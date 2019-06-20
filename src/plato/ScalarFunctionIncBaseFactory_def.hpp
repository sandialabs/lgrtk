#pragma once

#include "plato/ScalarFunctionBase.hpp"
#include "plato/PhysicsScalarFunctionInc.hpp"

namespace Plato
{
    /******************************************************************************//**
     * @brief Create method
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::ScalarFunctionIncBase> 
    ScalarFunctionIncBaseFactory<PhysicsT>::create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName)
    {
        auto tProblemFunction = aInputParams.sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunctionInc<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            throw std::runtime_error(tErrorString);
        }
    }


}
// namespace Plato