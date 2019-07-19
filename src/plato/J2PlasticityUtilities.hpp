#pragma once

#include "plato/Simp.hpp"

#include "plato/SimplexFadTypes.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{
/**************************************************************************//**
* @brief J2 Plasticity Utilities Class
******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class J2PlasticityUtilities
{
  private:
    static constexpr mSqrt3Over2 = std::sqrt(3.0/2.0);
    static constexpr mSqrt2Over3 = std::sqrt(2.0/3.0);

  public:
    /**************************************************************************//**
    * @brief Constructor
    ******************************************************************************/
    J2PlasticityUtilities() :
    {
    }

    /**************************************************************************//**
    * @brief Destructor
    ******************************************************************************/
    ~J2PlasticityUtilities() :
    {
    }

    /******************************************************************************//**
     * @brief Update the plastic strain and backstress for a plastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * @param [in] aHardeningModulusKinematic penalized kinematic hardening modulus
     * @param [out] aLocalState 2D container of local state variables to update
    **********************************************************************************/
    DEVICE_TYPE inline void
    updatePlasticStrainAndBackstressPlasticStep( 
                const Plato::OrdinalType       & aCellOrdinal,
                const Plato::ScalarMultiVector & aPrevLocalState,
                const Plato::ScalarMultiVector & aYieldSurfaceNormal,
                const Plato::Scalar            & aHardeningModulusKinematic,
                      Plato::ScalarMultiVector & aLocalState);

    /******************************************************************************//**
     * @brief Update the plastic strain and backstress for an elastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [out] aLocalState 2D container of local state variables to update
    **********************************************************************************/
    DEVICE_TYPE inline void
    updatePlasticStrainAndBackstressElasticStep( 
                const Plato::OrdinalType       & aCellOrdinal,
                const Plato::ScalarMultiVector & aPrevLocalState,
                      Plato::ScalarMultiVector & aLocalState);

    /******************************************************************************//**
     * @brief Compute the yield surface normal and the norm of the deviatoric stress minus the backstress
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aDeviatoricStress deviatoric stress tensor
     * @param [in] aLocalState 2D container of local state variables to update
     * @param [out] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * @param [out] aDevStressMinusBackstressNorm norm(deviatoric_stress - backstress)
    **********************************************************************************/
    template<typename LocalStateT, typename StressT>
    DEVICE_TYPE inline void
    computeDeviatoricStressMinusBackstressNormalized(
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                      Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
                      Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm);

    /******************************************************************************//**
     * @brief Compute the deviatoric stress
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aElasticStrain elastic strain tensor
     * @param [in] aPenalizedShearModulus penalized elastic shear modulus
     * @param [out] aDeviatoricStress deviatoric stress tensor
    **********************************************************************************/
    template<typename ElasticStrainT, typename ControlT, typename StressT>
    DEVICE_TYPE inline void
    computeDeviatoricStress(
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain,
                const ControlT                                     & aPenalizedShearModulus,
                      Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress);

    /******************************************************************************//**
     * @brief Fill the local residual vector with the plastic strain residual equation for plastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aLocalState 2D container of local state variables
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * @param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
    DEVICE_TYPE inline void
    fillPlasticStrainTensorResidualPlasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                      Plato::ScalarMultiVectorT< ResultT >          & aResult );

    /******************************************************************************//**
     * @brief Fill the local residual vector with the backstress residual equation for plastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aHardeningModulusKinematic penalized kinematic hardening modulus
     * @param [in] aLocalState 2D container of local state variables
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * @param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
             typename YieldSurfNormalT, typename ResultT>
    DEVICE_TYPE inline void
    fillBackstressTensorResidualPlasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const ControlT                                      & aHardeningModulusKinematic,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                      Plato::ScalarMultiVectorT< ResultT >          & aResult );

    /******************************************************************************//**
     * @brief Fill the local residual vector with the plastic strain residual equation for elastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aLocalState 2D container of local state variables
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
    DEVICE_TYPE inline void
    fillPlasticStrainTensorResidualElasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                      Plato::ScalarMultiVectorT< ResultT >          & aResult );

    /******************************************************************************//**
     * @brief Fill the local residual vector with the backstress residual equation for plastic step
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aLocalState 2D container of local state variables
     * @param [in] aPrevLocalState 2D container of previous local state variables
     * @param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
    DEVICE_TYPE inline void
    fillBackstressTensorResidualElasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                      Plato::ScalarMultiVectorT< ResultT >          & aResult );
};
// class J2PlasticityUtilities


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Update the plastic strain and backstress for a plastic step in 2D
  **********************************************************************************/
  template<>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::updatePlasticStrainAndBackstressPlasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aYieldSurfaceNormal,
              const Plato::Scalar            & aHardeningModulusKinematic,
                    Plato::ScalarMultiVector & aLocalState)
  {
    Plato::Scalar tMultiplier1 = aLocalState(aCellOrdinal, 1) * mSqrt3Over2;
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4) + 2.0 * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 2);

    Plato::Scalar tMultiplier2 = aLocalState(aCellOrdinal, 1) * mSqrt2Over3 * aHardeningModulusKinematic;
    // Backstress Tensor
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 2);
  }

  /******************************************************************************//**
   * @brief Update the plastic strain and backstress for a plastic step in 3D
  **********************************************************************************/
  template<>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::updatePlasticStrainAndBackstressPlasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aYieldSurfaceNormal,
              const Plato::Scalar            & aHardeningModulusKinematic,
                    Plato::ScalarMultiVector & aLocalState)
  {
    Plato::Scalar tMultiplier1 = aLocalState(aCellOrdinal, 1) * mSqrt3Over2;
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5) + 2.0 * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6) + 2.0 * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 4);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7) + 2.0 * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 5);

    Plato::Scalar tMultiplier2 = aLocalState(aCellOrdinal, 1) * mSqrt2Over3 * aHardeningModulusKinematic;
    // Backstress Tensor
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal,10) = aPrevLocalState(aCellOrdinal,10) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal,11) = aPrevLocalState(aCellOrdinal,11) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 3);
    aLocalState(aCellOrdinal,12) = aPrevLocalState(aCellOrdinal,12) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 4);
    aLocalState(aCellOrdinal,13) = aPrevLocalState(aCellOrdinal,13) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 5);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Update the plastic strain and backstress for an elastic step in 2D
  **********************************************************************************/
  template<>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::updatePlasticStrainAndBackstressElasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
                    Plato::ScalarMultiVector & aLocalState)
  {
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4);

    // Backstress Tensor
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7);
  }

  /******************************************************************************//**
   * @brief Update the plastic strain and backstress for an elastic step in 3D
  **********************************************************************************/
  template<>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::updatePlasticStrainAndBackstressElasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
                    Plato::ScalarMultiVector & aLocalState)
  {
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7);

    // Backstress Tensor
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9);
    aLocalState(aCellOrdinal,10) = aPrevLocalState(aCellOrdinal,10);
    aLocalState(aCellOrdinal,11) = aPrevLocalState(aCellOrdinal,11);
    aLocalState(aCellOrdinal,12) = aPrevLocalState(aCellOrdinal,12);
    aLocalState(aCellOrdinal,13) = aPrevLocalState(aCellOrdinal,13);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Compute the yield surface normal and the norm of the deviatoric stress minus the backstress for 2D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename StressT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::computeDeviatoricStressMinusBackstressNormalized(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
              const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                    Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
                    Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm)
  {
    // Subtract the backstress from the deviatoric stress
    aYieldSurfaceNormal(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 5);
    aYieldSurfaceNormal(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 6);
    aYieldSurfaceNormal(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 7);

    // Compute the norm || stress_deviator - backstress ||
    aDevStressMinusBackstressNorm(aCellOrdinal) = sqrt(pow(aYieldSurfaceNormal(aCellOrdinal, 0), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 1), 2) +
                                                 2.0 * pow(aYieldSurfaceNormal(aCellOrdinal, 2), 2));

    // Normalize the yield surface normal
    aYieldSurfaceNormal(aCellOrdinal, 0) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 1) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 2) /= aDevStressMinusBackstressNorm(aCellOrdinal);
  }

  /******************************************************************************//**
   * @brief Compute the yield surface normal and the norm of the deviatoric stress minus the backstress for 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename StressT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::computeDeviatoricStressMinusBackstressNormalized(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
              const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                    Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
                    Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm)
  {
    // Subtract the backstress from the deviatoric stress
    aYieldSurfaceNormal(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 8);
    aYieldSurfaceNormal(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 9);
    aYieldSurfaceNormal(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2) - aLocalState(aCellOrdinal,10);
    aYieldSurfaceNormal(aCellOrdinal, 3) = aDeviatoricStress(aCellOrdinal, 3) - aLocalState(aCellOrdinal,11);
    aYieldSurfaceNormal(aCellOrdinal, 4) = aDeviatoricStress(aCellOrdinal, 4) - aLocalState(aCellOrdinal,12);
    aYieldSurfaceNormal(aCellOrdinal, 5) = aDeviatoricStress(aCellOrdinal, 5) - aLocalState(aCellOrdinal,13);

    // Compute the norm || stress_deviator - backstress ||
    aDevStressMinusBackstressNorm(aCellOrdinal) = sqrt(pow(aYieldSurfaceNormal(aCellOrdinal, 0), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 1), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 2), 2) +
                                                 2.0 * pow(aYieldSurfaceNormal(aCellOrdinal, 3), 2) +
                                                 2.0 * pow(aYieldSurfaceNormal(aCellOrdinal, 4), 2) +
                                                 2.0 * pow(aYieldSurfaceNormal(aCellOrdinal, 5), 2));

    // Normalize the yield surface normal
    aYieldSurfaceNormal(aCellOrdinal, 0) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 1) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 2) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 3) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 4) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 5) /= aDevStressMinusBackstressNorm(aCellOrdinal);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Compute the deviatoric stress for 2D
  **********************************************************************************/
  template<>
  template<typename ElasticStrainT, typename ControlT, typename StressT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::computeDeviatoricStress(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain,
              const ControlT                                     & aPenalizedShearModulus,
                    Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress)
  {
    ElasticStrainT tTraceOver3 = (aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1)) / 3.0;
    aDeviatoricStress(aCellOrdinal, 0) = (2.0 * aPenalizedShearModulus) * (aElasticStrain(aCellOrdinal, 0) -
                                                                           tTraceOver3);
    aDeviatoricStress(aCellOrdinal, 1) = (2.0 * aPenalizedShearModulus) * (aElasticStrain(aCellOrdinal, 1) -
                                                                           tTraceOver3);
    aDeviatoricStress(aCellOrdinal, 2) = aPenalizedShearModulus * aElasticStrain(aCellOrdinal, 2);
  }

  /******************************************************************************//**
   * @brief Compute the deviatoric stress for 3D
  **********************************************************************************/
  template<>
  template<typename ElasticStrainT, typename ControlT, typename StressT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::computeDeviatoricStress(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain,
              const ControlT                                     & aPenalizedShearModulus,
                    Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress)
  {
    ElasticStrainT tTraceOver3 = (  aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1)
                                  + aElasticStrain(aCellOrdinal, 2) ) / 3.0;
    aDeviatoricStress(aCellOrdinal, 0) = (2.0 * aPenalizedShearModulus) * (aElasticStrain(aCellOrdinal, 0) -
                                                                           tTraceOver3);
    aDeviatoricStress(aCellOrdinal, 1) = (2.0 * aPenalizedShearModulus) * (aElasticStrain(aCellOrdinal, 1) -
                                                                           tTraceOver3);
    aDeviatoricStress(aCellOrdinal, 2) = (2.0 * aPenalizedShearModulus) * (aElasticStrain(aCellOrdinal, 2) -
                                                                           tTraceOver3);
    aDeviatoricStress(aCellOrdinal, 3) = aPenalizedShearModulus * aElasticStrain(aCellOrdinal, 3);
    aDeviatoricStress(aCellOrdinal, 4) = aPenalizedShearModulus * aElasticStrain(aCellOrdinal, 4);
    aDeviatoricStress(aCellOrdinal, 5) = aPenalizedShearModulus * aElasticStrain(aCellOrdinal, 5);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/
  
  /******************************************************************************//**
   * @brief Fill the local residual vector with the plastic strain residual equation for plastic step in 2D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::fillPlasticStrainTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 2);
  }

  /******************************************************************************//**
   * @brief Fill the local residual vector with the plastic strain residual equation for plastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::fillPlasticStrainTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 2);
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 3);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 4);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 5);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Fill the local residual vector with the backstress residual equation for plastic step in 2D
  **********************************************************************************/
  template<>
  template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
           typename YieldSurfNormalT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::fillBackstressTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const ControlT                                      & aHardeningModulusKinematic,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 2);
  }

  /******************************************************************************//**
   * @brief Fill the local residual vector with the backstress residual equation for plastic step in 3D
  **********************************************************************************/
  template<>
  template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
           typename YieldSurfNormalT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::fillBackstressTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const ControlT                                      & aHardeningModulusKinematic,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal,10) = aLocalState(aCellOrdinal,10) - aPrevLocalState(aCellOrdinal,10);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 2);
    aResult(aCellOrdinal,11) = aLocalState(aCellOrdinal,11) - aPrevLocalState(aCellOrdinal,11);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 3);
    aResult(aCellOrdinal,12) = aLocalState(aCellOrdinal,12) - aPrevLocalState(aCellOrdinal,12);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 4);
    aResult(aCellOrdinal,13) = aLocalState(aCellOrdinal,13) - aPrevLocalState(aCellOrdinal,13);
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 5);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Fill the local residual vector with the plastic strain residual equation for elastic step in 2D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::fillPlasticStrainTensorResidualElasticStep( 
              const Plato::OrdinalType                              & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >        & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >    & aPrevLocalState,
                    Plato::ScalarMultiVectorT< ResultT >            & aResult )
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4);
  }

  /******************************************************************************//**
   * @brief Fill the local residual vector with the plastic strain residual equation for elastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::fillPlasticStrainTensorResidualElasticStep( 
              const Plato::OrdinalType                              & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >        & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >    & aPrevLocalState,
                    Plato::ScalarMultiVectorT< ResultT >            & aResult )
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4);
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * @brief Fill the local residual vector with the backstress residual equation for elastic step in 2D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<2>::fillBackstressTensorResidualElasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7);
  }

  /******************************************************************************//**
   * @brief Fill the local residual vector with the backstress residual equation for elastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  DEVICE_TYPE inline void
  J2PlasticityUtilities<3>::fillBackstressTensorResidualElasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult )
  {
    aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8);
    aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9);
    aResult(aCellOrdinal,10) = aLocalState(aCellOrdinal,10) - aPrevLocalState(aCellOrdinal,10);
    aResult(aCellOrdinal,11) = aLocalState(aCellOrdinal,11) - aPrevLocalState(aCellOrdinal,11);
    aResult(aCellOrdinal,12) = aLocalState(aCellOrdinal,12) - aPrevLocalState(aCellOrdinal,12);
    aResult(aCellOrdinal,13) = aLocalState(aCellOrdinal,13) - aPrevLocalState(aCellOrdinal,13);
  }

} // namespace Plato
