#ifndef PLATO_THERMOMECHANICS_HPP
#define PLATO_THERMOMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simplex.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/AbstractLocalMeasure.hpp"
#include "plato/ThermoelastostaticResidual.hpp"
#include "plato/TransientThermomechResidual.hpp"
#include "plato/InternalThermoelasticEnergy.hpp"
#include "plato/AnalyzeMacros.hpp"
#include "plato/Volume.hpp"
//#include "plato/TMStressPNorm.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/AnalyzeMacros.hpp"
#include "plato/J2PlasticityLocalResidual.hpp"
#include "plato/Plato_AugLagStressCriterionQuadratic.hpp"
#include "plato/ThermalVonMisesLocalMeasure.hpp"

namespace Plato
{

namespace ThermomechanicsFactory
{
    /******************************************************************************//**
    * @brief Create a local measure for use in augmented lagrangian quadratic
    * @param [in] aInputParams input parameters
    * @param [in] aFuncName scalar function name
    **********************************************************************************/
    template <typename EvaluationType>
    inline std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>> 
    create_local_measure(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aInputParams.sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
          return std::make_shared<ThermalVonMisesLocalMeasure<EvaluationType, Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>>(aInputParams, "VonMises");
        }
        else
        {
          THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }

    /******************************************************************************//**
     * @brief Create augmented Lagrangian local constraint criterion with quadratic constraint formulation
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    inline std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
    stress_constraint_quadratic(Omega_h::Mesh& aMesh,
                                Omega_h::MeshSets& aMeshSets,
                                Plato::DataMap& aDataMap,
                                Teuchos::ParameterList & aInputParams,
                                std::string & aFuncName)
    {
        auto EvalMeasure = Plato::ThermomechanicsFactory::create_local_measure<EvaluationType>(aInputParams, aFuncName);
        using Residual = typename Plato::ResidualTypes<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>>;
        auto PODMeasure = Plato::ThermomechanicsFactory::create_local_measure<Residual>(aInputParams, aFuncName);

        using SimplexT = Plato::SimplexThermomechanics<EvaluationType::SpatialDim>;
        std::shared_ptr<Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT>> tOutput;
        tOutput = std::make_shared< Plato::AugLagStressCriterionQuadratic<EvaluationType,SimplexT> >
                    (aMesh, aMeshSets, aDataMap, aInputParams, aFuncName);
        //THROWERR("Not finished implementing this for thermomechanics... need local measure that is compatible.")
        tOutput->setLocalMeasure(EvalMeasure, PODMeasure);
        return (tOutput);
    }

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aParamList,
                         std::string aStrVectorFunctionType)
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Elliptic")
        {
            auto tPenaltyParams = aParamList.sublist(aStrVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<AbstractVectorFunctionInc<EvaluationType>>
    createVectorFunctionInc(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets, 
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string strVectorFunctionType )
    /******************************************************************************/
    {
        if( strVectorFunctionType == "Parabolic" )
        {
            auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::RAMP>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aParamList, 
                         std::string aStrScalarFunctionType,
                         std::string aStrScalarFunctionName)
    /******************************************************************************/
    {

        if(aStrScalarFunctionType == "Internal Thermoelastic Energy")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else if(aStrScalarFunctionType == "Stress Constraint Quadratic")
        {
            return (Plato::ThermomechanicsFactory::stress_constraint_quadratic<EvaluationType>
                   (aMesh, aMeshSets, aDataMap, aParamList, aStrScalarFunctionName));
        }
#ifdef NOPE
        else 
        if(aStrScalarFunctionType == "Stress P-Norm")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
#endif
        if(aStrScalarFunctionType == "Volume")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
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
        if( strScalarFunctionType == "Internal Thermoelastic Energy" ){
            auto penaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, aStrScalarFunctionName);
            } else
            if( penaltyType == "RAMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::RAMP>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, aStrScalarFunctionName);
            } else
            if( penaltyType == "Heaviside" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::Heaviside>>
                         (aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, aStrScalarFunctionName);
            } else {
                THROWERR("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        } else {
            THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

}; // struct FunctionFactory

} // namespace ThermomechanicsFactory


/****************************************************************************//**
 * @brief Concrete class for use as the SimplexPhysics template argument in
 *        EllipticProblem and ParabolicProblem
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Thermomechanics: public Plato::SimplexThermomechanics<SpaceDimParam>
{
public:
    typedef Plato::ThermomechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexThermomechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
} // namespace Plato

#endif
