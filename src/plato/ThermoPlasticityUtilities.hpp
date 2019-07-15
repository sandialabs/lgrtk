#pragma once

#include "plato/Simp.hpp"

#include "plato/SimplexFadTypes.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/SimplexPlasticity.hpp"
#include "plato/SimplexThermoPlasticity.hpp"

namespace Plato
{
/**************************************************************************//**
* @brief Thermo-Plasticity Utilities Class
******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename SimplexPhysicsT>
class ThermoPlasticityUtilities
{
  private:
    using SimplexPhysicsT<SpaceDim>::mNumNodesPerCell;
    using SimplexPhysicsT<SpaceDim>::mNumDofsPerNode;

    Plato::Scalar mThermalExpansionCoefficient;
    Plato::Scalar mReferenceTemperature;
  public:
    /**************************************************************************//**
    * @brief Constructor
    * @param [in] aThermalExpansionCoefficient thermal expansion coefficient
    * @param [in] aReferenceTemperature reference temperature
    ******************************************************************************/
    ThermoPlasticityUtilities(Plato::Scalar aThermalExpansionCoefficient, Plato::Scalar aReferenceTemperature) :
      mThermalExpansionCoefficient(aThermalExpansionCoefficient),
      mReferenceTemperature(aReferenceTemperature)
    {
    }

    /**************************************************************************//**
    * @brief Destructor
    ******************************************************************************/
    ~ThermoPlasticityUtilities() :
    {
    }

    /******************************************************************************//**
     * @brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aGlobalState 2D container of global state variables
     * @param [in] aLocalState 2D container of local state variables
     * @param [in] aBasisFunctions 1D container of shape function values at the single quadrature point
     * @param [in] aGradient 3D container of basis function gradients
     * @param [out] aElasticStrain 2D container of elastic strain tensor components
    **********************************************************************************/
    template<typename GlobalStateT, typename LocalStateT, typename ConfigT, typename ElasticStrainT>
    DEVICE_TYPE inline void
    computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarArray3DT< ConfigT >             & aGradient,
                      Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain);

};
// class ThermoPlasticityUtilities


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 2D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename ConfigT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexPlasticity>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarArray3DT< ConfigT >             & aGradient,
                      Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain)
  {

    // Compute total strain
    Plato::OrdinalType tVoigtTerm = 0;
    for(Plato::OrdinalType tDofI = 0; tDofI < 2; ++tDofI){
      aElasticStrain(aCellOrdinal,tVoigtTerm) = 0.0;
      for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
        Plato::OrdinalType tLocalOrdinal = tNode*mNumDofsPerNode + tDofI;
        aElasticStrain(aCellOrdinal,tVoigtTerm) += aGlobalState(aCellOrdinal,tLocalOrdinal) * aGradient(aCellOrdinal,tNode,tDofI);
      }
      ++tVoigtTerm;
    }
    for (Plato::OrdinalType tDofJ = 1; tDofJ >= 1; --tDofJ){
      for (Plato::OrdinalType tDofI = tDofJ-1; tDofI >= 0; --tDofI){
        for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
          Plato::OrdinalType tLocalOrdinalI = tNode*mNumDofsPerNode + tDofI;
          Plato::OrdinalType tLocalOrdinalJ = tNode*mNumDofsPerNode + tDofJ;
          aElasticStrain(aCellOrdinal,tVoigtTerm) +=( aGlobalState(aCellOrdinal,tLocalOrdinalJ) * aGradient(aCellOrdinal,tNode,tDofI)
                                                    + aGlobalState(aCellOrdinal,tLocalOrdinalI) * aGradient(aCellOrdinal,tNode,tDofJ));
        }
        ++tVoigtTerm;
      }
    }

    // Subtract the plastic strain
    aElasticStrain(aCellOrdinal, 0) -= aLocalState(aCellOrdinal, 2);
    aElasticStrain(aCellOrdinal, 1) -= aLocalState(aCellOrdinal, 3);
    aElasticStrain(aCellOrdinal, 2) -= aLocalState(aCellOrdinal, 4);
  }

  /******************************************************************************//**
   * @brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 3D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename ConfigT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexPlasticity>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarArray3DT< ConfigT >             & aGradient,
                      Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain)
  {
    // Compute total strain
    Plato::OrdinalType tVoigtTerm = 0;
    for(Plato::OrdinalType tDofI = 0; tDofI < 3; ++tDofI){
      aElasticStrain(aCellOrdinal,tVoigtTerm) = 0.0;
      for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
        Plato::OrdinalType tLocalOrdinal = tNode*mNumDofsPerNode + tDofI;
        aElasticStrain(aCellOrdinal,tVoigtTerm) += aGlobalState(aCellOrdinal,tLocalOrdinal) * aGradient(aCellOrdinal,tNode,tDofI);
      }
      ++tVoigtTerm;
    }
    for (Plato::OrdinalType tDofJ = 2; tDofJ >= 1; --tDofJ){
      for (Plato::OrdinalType tDofI = tDofJ-1; tDofI >= 0; --tDofI){
        for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
          Plato::OrdinalType tLocalOrdinalI = tNode*mNumDofsPerNode + tDofI;
          Plato::OrdinalType tLocalOrdinalJ = tNode*mNumDofsPerNode + tDofJ;
          aElasticStrain(aCellOrdinal,tVoigtTerm) +=( aGlobalState(aCellOrdinal,tLocalOrdinalJ) * aGradient(aCellOrdinal,tNode,tDofI)
                                                    + aGlobalState(aCellOrdinal,tLocalOrdinalI) * aGradient(aCellOrdinal,tNode,tDofJ));
        }
        ++tVoigtTerm;
      }
    }

    // Subtract plastic strain
    aElasticStrain(aCellOrdinal, 0) -= aLocalState(aCellOrdinal, 2);
    aElasticStrain(aCellOrdinal, 1) -= aLocalState(aCellOrdinal, 3);
    aElasticStrain(aCellOrdinal, 2) -= aLocalState(aCellOrdinal, 4);
    aElasticStrain(aCellOrdinal, 3) -= aLocalState(aCellOrdinal, 5);
    aElasticStrain(aCellOrdinal, 4) -= aLocalState(aCellOrdinal, 6);
    aElasticStrain(aCellOrdinal, 5) -= aLocalState(aCellOrdinal, 7);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 2D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename ConfigT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexThermoPlasticity>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarArray3DT< ConfigT >             & aGradient,
                      Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain)
  {

    // Compute total strain
    Plato::OrdinalType tVoigtTerm = 0;
    for(Plato::OrdinalType tDofI = 0; tDofI < 2; ++tDofI){
      aElasticStrain(aCellOrdinal,tVoigtTerm) = 0.0;
      for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
        Plato::OrdinalType tLocalOrdinal = tNode*mNumDofsPerNode + tDofI;
        aElasticStrain(aCellOrdinal,tVoigtTerm) += aGlobalState(aCellOrdinal,tLocalOrdinal) * aGradient(aCellOrdinal,tNode,tDofI);
      }
      ++tVoigtTerm;
    }
    for (Plato::OrdinalType tDofJ = 1; tDofJ >= 1; --tDofJ){
      for (Plato::OrdinalType tDofI = tDofJ-1; tDofI >= 0; --tDofI){
        for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
          Plato::OrdinalType tLocalOrdinalI = tNode*mNumDofsPerNode + tDofI;
          Plato::OrdinalType tLocalOrdinalJ = tNode*mNumDofsPerNode + tDofJ;
          aElasticStrain(aCellOrdinal,tVoigtTerm) +=( aGlobalState(aCellOrdinal,tLocalOrdinalJ) * aGradient(aCellOrdinal,tNode,tDofI)
                                                    + aGlobalState(aCellOrdinal,tLocalOrdinalI) * aGradient(aCellOrdinal,tNode,tDofJ));
        }
        ++tVoigtTerm;
      }
    }

    // Subtract plastic strain
    aElasticStrain(aCellOrdinal, 0) -= aLocalState(aCellOrdinal, 2);
    aElasticStrain(aCellOrdinal, 1) -= aLocalState(aCellOrdinal, 3);
    aElasticStrain(aCellOrdinal, 2) -= aLocalState(aCellOrdinal, 4);

    // Compute the temperature
    GlobalStateT tTemperature = 0.0;
    for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
    {
      Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + 2;
      tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
    }

    // Subtract thermal strain
    GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
    aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
  }

  /******************************************************************************//**
   * @brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 3D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename ConfigT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexThermoPlasticity>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarArray3DT< ConfigT >             & aGradient,
                      Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain)
  {
    // Compute total strain
    Plato::OrdinalType tVoigtTerm = 0;
    for(Plato::OrdinalType tDofI = 0; tDofI < 3; ++tDofI){
      aElasticStrain(aCellOrdinal,tVoigtTerm) = 0.0;
      for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
        Plato::OrdinalType tLocalOrdinal = tNode*mNumDofsPerNode + tDofI;
        aElasticStrain(aCellOrdinal,tVoigtTerm) += aGlobalState(aCellOrdinal,tLocalOrdinal) * aGradient(aCellOrdinal,tNode,tDofI);
      }
      ++tVoigtTerm;
    }
    for (Plato::OrdinalType tDofJ = 2; tDofJ >= 1; --tDofJ){
      for (Plato::OrdinalType tDofI = tDofJ-1; tDofI >= 0; --tDofI){
        for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode){
          Plato::OrdinalType tLocalOrdinalI = tNode*mNumDofsPerNode + tDofI;
          Plato::OrdinalType tLocalOrdinalJ = tNode*mNumDofsPerNode + tDofJ;
          aElasticStrain(aCellOrdinal,tVoigtTerm) +=( aGlobalState(aCellOrdinal,tLocalOrdinalJ) * aGradient(aCellOrdinal,tNode,tDofI)
                                                    + aGlobalState(aCellOrdinal,tLocalOrdinalI) * aGradient(aCellOrdinal,tNode,tDofJ));
        }
        ++tVoigtTerm;
      }
    }

    // Subtract plastic strain
    aElasticStrain(aCellOrdinal, 0) -= aLocalState(aCellOrdinal, 2);
    aElasticStrain(aCellOrdinal, 1) -= aLocalState(aCellOrdinal, 3);
    aElasticStrain(aCellOrdinal, 2) -= aLocalState(aCellOrdinal, 4);
    aElasticStrain(aCellOrdinal, 3) -= aLocalState(aCellOrdinal, 5);
    aElasticStrain(aCellOrdinal, 4) -= aLocalState(aCellOrdinal, 6);
    aElasticStrain(aCellOrdinal, 5) -= aLocalState(aCellOrdinal, 7);

    // Compute the temperature
    GlobalStateT tTemperature = 0.0;
    for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
    {
      Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + 2;
      tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
    }

    // Subtract thermal strain
    GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
    aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 2) -= tThermalStrain;
  }

} // namespace Plato
