#pragma once

#include "plato/ScalarFunctionBase.hpp"
#include "plato/WeightedSumFunction.hpp"
#include "plato/PhysicsScalarFunction.hpp"

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
    std::shared_ptr<Plato::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           const std::string aFunctionName)
    {
        auto tProblemSpecs = aInputParams.sublist("Plato Problem");
        auto tProblemFunction = tProblemSpecs.sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "");

        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else if(tFunctionType == "Physics Based")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function 'Type' specified in function name ") + 
                                             aFunctionName + " ParameterList";
            throw std::runtime_error(tErrorString);
        }
    }


}
// namespace Plato