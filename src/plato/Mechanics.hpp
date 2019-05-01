#ifndef PLATO_MECHANICS_HPP
#define PLATO_MECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Plato_AugLagStressCriterionGeneral.hpp"
#include "plato/Plato_AugLagStressCriterion.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/ElastostaticResidual.hpp"
#include "plato/InternalElasticEnergy.hpp"
#include "plato/EffectiveEnergy.hpp"
#include "plato/Volume.hpp"
#include "plato/StressPNorm.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

namespace MechanicsFactory
{

/******************************************************************************//**
 * @brief Create elastostatics residual equation
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
elastostatics_residual(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap& aDataMap,
                       Teuchos::ParameterList& aInputParams,
                       const std::string & aFuncType)
{
    std::shared_ptr<AbstractVectorFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist("Elastostatics").sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::ElastostaticResidual<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function elastostatics_residual

/******************************************************************************//**
 * @brief Create augmented Lagrangian stress constraint criterion tailored for linear problems
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_constraint_linear(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList & aInputParams)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared< Plato::AugLagStressCriterion<EvaluationType> >(aMesh, aMeshSets, aDataMap, aInputParams);
    return (tOutput);
}

/******************************************************************************//**
 * @brief Create augmented Lagrangian stress constraint criterion tailored for general problems
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_constraint_general(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList & aInputParams)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    tOutput = std::make_shared < AugLagStressCriterionGeneral<EvaluationType> > (aMesh, aMeshSets, aDataMap, aInputParams);
    return (tOutput);
}

/******************************************************************************//**
 * @brief Create internal elastic energy criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
internal_elastic_energy(Omega_h::Mesh& aMesh,
                        Omega_h::MeshSets& aMeshSets,
                        Plato::DataMap& aDataMap,
                        Teuchos::ParameterList & aInputParams,
                        const std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::InternalElasticEnergy<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function internal_elastic_energy

/******************************************************************************//**
 * @brief Create stress p-norm criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
stress_p_norm(Omega_h::Mesh& aMesh,
              Omega_h::MeshSets& aMeshSets,
              Plato::DataMap& aDataMap,
              Teuchos::ParameterList & aInputParams,
              const std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::StressPNorm<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function stress_p_norm

/******************************************************************************//**
 * @brief Create effective energy criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
effective_energy(Omega_h::Mesh& aMesh,
                 Omega_h::MeshSets& aMeshSets,
                 Plato::DataMap& aDataMap,
                 Teuchos::ParameterList & aInputParams,
                 const std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::EffectiveEnergy<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function effective_energy

/******************************************************************************//**
 * @brief Create volume criterion
 * @param [in] aMesh mesh database
 * @param [in] aMeshSets side sets database
 * @param [in] aDataMap PLATO Analyze physics-based database
 * @param [in] aInputParams input parameters
 * @param [in] aFuncType vector function name
**********************************************************************************/
template<typename EvaluationType>
inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
volume(Omega_h::Mesh& aMesh,
       Omega_h::MeshSets& aMeshSets,
       Plato::DataMap& aDataMap,
       Teuchos::ParameterList & aInputParams,
       const std::string & aFuncName)
{
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>> tOutput;
    auto tPenaltyParams = aInputParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    if(tPenaltyType == "SIMP")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "RAMP")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    else
    if(tPenaltyType == "Heaviside")
    {
        tOutput = std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aInputParams, tPenaltyParams);
    }
    return (tOutput);
}
// function volume

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
    std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aInputParams,
                         std::string aFuncName)
    {

        if(aFuncName == "Elastostatics")
        {
            return (Plato::MechanicsFactory::elastostatics_residual<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
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
                         std::string aFuncName)
    {

        if(aFuncName == "Internal Elastic Energy")
        {
            return (Plato::MechanicsFactory::internal_elastic_energy<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncName == "Stress P-Norm")
        {
            return (Plato::MechanicsFactory::stress_p_norm<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncName == "Effective Energy")
        {
            return (Plato::MechanicsFactory::effective_energy<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else if(aFuncName == "Stress Constraint")
        {
            return (Plato::MechanicsFactory::stress_constraint_linear<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams));
        }
        else if(aFuncName == "Stress Constraint General")
        {
            return (Plato::MechanicsFactory::stress_constraint_general<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams));
        }
        else if(aFuncName == "Volume")
        {
            return (Plato::MechanicsFactory::volume<EvaluationType>(aMesh, aMeshSets, aDataMap, aInputParams, aFuncName));
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
};
// struct FunctionFactory

} // namespace MechanicsFactory

/******************************************************************************//**
 * @brief Factory interface for linear mechanics problem
**********************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Mechanics: public Plato::SimplexMechanics<SpaceDimParam>
{
public:
    using FunctionFactory = typename Plato::MechanicsFactory::FunctionFactory;
    using SimplexT = SimplexMechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
// class Mechanics

} // namespace Plato

#endif
