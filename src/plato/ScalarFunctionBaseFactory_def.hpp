#pragma once

#include "plato/ScalarFunctionBase.hpp"
#include "plato/WeightedSumFunction.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/DivisionFunction.hpp"
#include "plato/LeastSquaresFunction.hpp"
#include "plato/MassPropertiesFunction.hpp"
#include "plato/AnalyzeMacros.hpp"

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
           std::string& aFunctionName)
    {
        auto tProblemFunction = aInputParams.sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else if(tFunctionType == "Mass Properties")
        {
            return std::make_shared<MassPropertiesFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            THROWERR(tErrorString);
        }
    }


}
// namespace Plato