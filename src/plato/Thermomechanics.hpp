#ifndef PLATO_THERMOMECHANICS_HPP
#define PLATO_THERMOMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simplex.hpp"
#include "plato/SimplexProjection.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/ThermoelastostaticResidual.hpp"
#include "plato/Projection.hpp"
#include "plato/StabilizedThermoelastostaticResidual.hpp"
#include "plato/PressureGradientProjectionResidual.hpp"
#include "plato/TransientThermomechResidual.hpp"
#include "plato/InternalThermoelasticEnergy.hpp"
#include "plato/Volume.hpp"
//#include "plato/TMStressPNorm.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

namespace ThermomechanicsFactory
{

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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<EvaluationType>>
    createVectorFunctionVMS(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string aStrVectorFunctionType)
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Stabilized Elliptic")
        {
            auto tPenaltyParams = aParamList.sublist(aStrVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
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

#ifdef NOPE
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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            printf("Name = '%s' and Type = '%s'\n", aStrScalarFunctionName.c_str(), aStrScalarFunctionType.c_str());
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
#else
        throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
#endif
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
#ifdef NOPE
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
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        } else {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
#else
        throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
#endif
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
    using FunctionFactory = typename Plato::ThermomechanicsFactory::FunctionFactory;
    using SimplexT = SimplexThermomechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};


/****************************************************************************//**
 * @brief Concrete class for use as the SimplexPhysics template argument in
 *        EllipticVMSProblem
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class StabilizedThermomechanics: public Plato::SimplexStabilizedThermomechanics<SpaceDimParam>
{
public:
    using FunctionFactory = typename Plato::ThermomechanicsFactory::FunctionFactory;
    using SimplexT        = SimplexStabilizedThermomechanics<SpaceDimParam>;

    using ProjectorT = typename Plato::Projection<SpaceDimParam,
                                                  SimplexT::m_numDofsPerNode,
                                                  SimplexT::m_PDofOffset,
                                                  /* numProjectionDofs=*/ 1>;


    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif
