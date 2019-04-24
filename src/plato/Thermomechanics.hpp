#ifndef PLATO_THERMOMECHANICS_HPP
#define PLATO_THERMOMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/Simplex.hpp"
#include "plato/SimplexThermomechanics.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/ThermoelastostaticResidual.hpp"
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

        if(aStrVectorFunctionType == "Thermoelastostatics")
        {
            auto tPenaltyParams = aParamList.sublist("Thermoelastostatics").sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, ::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, ::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
        if( strVectorFunctionType == "First Order" )
        {
            auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, ::SIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
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
                         std::string aStrScalarFunctionType)
    /******************************************************************************/
    {

        if(aStrScalarFunctionType == "Internal Thermoelastic Energy")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, ::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, ::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
#ifdef NOPE
        else 
        if(aStrScalarFunctionType == "Stress P-Norm")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, ::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, ::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
#endif
        if(aStrScalarFunctionType == "Volume")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, ::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Volume<EvaluationType, ::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<AbstractScalarFunctionInc<EvaluationType>>
    createScalarFunctionInc(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string strScalarFunctionType )
    /******************************************************************************/
    {
        if( strScalarFunctionType == "Internal Thermoelastic Energy" ){
            auto penaltyParams = aParamList.sublist(strScalarFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        } else {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
}; // struct FunctionFactory

} // namespace ThermomechanicsFactory

template<Plato::OrdinalType SpaceDimParam>
class Thermomechanics: public Plato::SimplexThermomechanics<SpaceDimParam>
{
public:
    using FunctionFactory = typename Plato::ThermomechanicsFactory::FunctionFactory;
    using SimplexT = SimplexThermomechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif
