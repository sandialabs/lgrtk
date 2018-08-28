#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "plato/Simplex.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ThermostaticResidual.hpp"
#include "plato/InternalThermalEnergy.hpp"
#include "plato/FluxPNorm.hpp"
#include "plato/Volume.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"


namespace Plato {

namespace ThermalFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
  template <typename EvaluationType>
  std::shared_ptr<AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets, 
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& paramList,
    std::string strVectorFunctionType )
  {

    if( strVectorFunctionType == "Thermostatics" ){
      auto penaltyParams = paramList.sublist(strVectorFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, ::SIMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
  template <typename EvaluationType>
  std::shared_ptr<AbstractScalarFunction<EvaluationType>>
  createScalarFunction( 
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& paramList,
    std::string strScalarFunctionType )
  {

    if( strScalarFunctionType == "Internal Thermal Energy" ){
      auto penaltyParams = paramList.sublist(strScalarFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<InternalThermalEnergy<EvaluationType, ::SIMP>>(aMesh, aMeshSets, aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<InternalThermalEnergy<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<InternalThermalEnergy<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else
    if( strScalarFunctionType == "Flux P-Norm" ){
      auto penaltyParams = paramList.sublist(strScalarFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<FluxPNorm<EvaluationType, ::SIMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<FluxPNorm<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<FluxPNorm<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else 
    if( strScalarFunctionType == "Volume" ){
      auto penaltyParams = paramList.sublist(strScalarFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Volume<EvaluationType, ::SIMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Volume<EvaluationType, ::RAMP>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Volume<EvaluationType, ::Heaviside>>(aMesh,aMeshSets,aDataMap,paramList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
};
}

template <Plato::OrdinalType SpaceDimParam>
class Thermal : public SimplexThermal<SpaceDimParam> {
  public:
    using FunctionFactory = typename Plato::ThermalFactory::FunctionFactory<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

}

#endif
