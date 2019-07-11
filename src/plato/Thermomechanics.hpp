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
#include "plato/StabilizedThermoelastostaticResidual.hpp"
#include "plato/PressureGradientProjectionResidual.hpp"
#include "plato/TransientThermomechResidual.hpp"
//#include "plato/TransientStabilizedThermomechResidual.hpp"
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
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::ThermoelastostaticResidual<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::StabilizedThermoelastostaticResidual<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<TransientThermomechResidual<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
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

#ifdef NOPE
        if(aStrScalarFunctionType == "Internal Thermoelastic Energy")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::InternalThermoelasticEnergy<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
        if(aStrScalarFunctionType == "Stress P-Norm")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::TMStressPNorm<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else 
        if(aStrScalarFunctionType == "Volume")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
                            std::string strScalarFunctionType )
    /******************************************************************************/
    {
#ifdef NOPE
        if( strScalarFunctionType == "Internal Thermoelastic Energy" ){
            auto penaltyParams = aParamList.sublist(strScalarFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "RAMP" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
            } else
            if( penaltyType == "Heaviside" ){
                return std::make_shared<InternalThermoelasticEnergyInc<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
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

namespace ProjectionFactory
{

/******************************************************************************/
struct FunctionFactory
{
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

        if(aStrVectorFunctionType == "State Gradient Projection")
        {
            auto tPenaltyParams = aParamList.sublist(aStrVectorFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::PressureGradientProjectionResidual<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
}; // struct FunctionFactory

} // namespace ProjectionFactory

template<Plato::OrdinalType SpaceDimParam>
class Thermomechanics: public Plato::SimplexThermomechanics<SpaceDimParam>
{
public:
    using FunctionFactory = typename Plato::ThermomechanicsFactory::FunctionFactory;
    using SimplexT = SimplexThermomechanics<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

template<
  Plato::OrdinalType SpaceDimParam,
  Plato::OrdinalType TotalDofsParam = SpaceDimParam,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDofs = 1>
class Projection: public Plato::SimplexProjection<SpaceDimParam>
{
public:
    using FunctionFactory = typename Plato::ProjectionFactory::FunctionFactory;
    using SimplexT        = SimplexProjection<SpaceDimParam, TotalDofsParam, ProjectionDofOffset, NumProjectionDofs>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

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
