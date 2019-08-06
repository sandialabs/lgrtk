#ifndef PLATO_STABILIZED_MECHANICS_HPP
#define PLATO_STABILIZED_MECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/SimplexProjection.hpp"
#include "plato/SimplexStabilizedMechanics.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/StabilizedElastostaticResidual.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

namespace StabilizedMechanicsFactory
{

/******************************************************************************//**
 * @brief Create elastostatics residual equation
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
stabilized_elastostatics_residual(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap& aDataMap,
                       Teuchos::ParameterList& aInputParams,
                       std::string aFuncName)
{
    std::shared_ptr<AbstractVectorFunctionVMS<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::StabilizedElastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function stabilized_elastostatics_residual


/******************************************************************************//**
 * @brief Factory for linear mechanics problem
**********************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * @brief Create a PLATO vector function (i.e. residual equation)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
    createVectorFunctionVMS(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aInputParams,
                         std::string aFuncName)
    {

        if(aFuncName == "Elliptic")
        {
            return (Plato::StabilizedMechanicsFactory::stabilized_elastostatics_residual<EvaluationType>
                     (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<AbstractScalarFunctionInc<EvaluationType>>
    createScalarFunctionInc(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string strScalarFunctionType,
                            std::string aStrScalarFunctionName )
    /******************************************************************************/
    {
        THROWERR("Not yet implemented");
    }

    /******************************************************************************//**
     * @brief Create a PLATO scalar function (i.e. optimization criterion)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName scalar function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aInputParams,
                         std::string aFuncType,
                         std::string aFuncName)
    {
        {
            THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
};
// struct FunctionFactory

} // namespace StabilizedMechanicsFactory

/****************************************************************************//**
 * @brief Concrete class for use as the SimplexPhysics template argument in
 *        EllipticVMSProblem
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class StabilizedMechanics: public Plato::SimplexStabilizedMechanics<SpaceDimParam>
{
public:
    typedef Plato::StabilizedMechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT        = SimplexStabilizedMechanics<SpaceDimParam>;

    using ProjectorT = typename Plato::Projection<SpaceDimParam,
                                                  SimplexT::mNumDofsPerNode,
                                                  SimplexT::mPDofOffset,
                                                  /* numProjectionDofs=*/ 1>;

    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};


} // namespace Plato

#endif
