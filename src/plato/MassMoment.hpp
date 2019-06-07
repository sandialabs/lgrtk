#pragma once

#include "plato/Simplex.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class MassMoment : public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                   public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    
    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mCellMaterialDensity;

    std::string mCalculationType;

  public:
    /**************************************************************************/
    MassMoment(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList& aInputParams) :
           Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "MassMoment"),
           mCellMaterialDensity(1.0),
           mCalculationType("")
    /**************************************************************************/
    {
      auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
      mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
    }

    /**************************************************************************
     * Unit testing constructor
    /**************************************************************************/
    MassMoment(Omega_h::Mesh& aMesh, 
               Omega_h::MeshSets& aMeshSets,
               Plato::DataMap& aDataMap) :
               Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "MassMoment"),
               mCellMaterialDensity(1.0),
               mCalculationType("")
    /**************************************************************************/
    {

    }

    /**************************************************************************/
    void setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mCellMaterialDensity = aMaterialDensity;
    }

    /**************************************************************************/
    void setCalculationType(const std::string & aCalculationType)
    /**************************************************************************/
    {
      mCalculationType = aCalculationType;
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const 
    /**************************************************************************/
    {
      if (mCalculationType == "Mass")
        computeStructuralMass(aControl, aConfig, aResult, aTimeStep);
      else if (mCalculationType == "FirstX")
        computeFirstMoment(aControl, aConfig, aResult, 0, aTimeStep);
      else if (mCalculationType == "FirstY")
        computeFirstMoment(aControl, aConfig, aResult, 1, aTimeStep);
      else if (mCalculationType == "FirstZ")
        computeFirstMoment(aControl, aConfig, aResult, 2, aTimeStep);
      else if (mCalculationType == "SecondXX")
        computeSecondMoment(aControl, aConfig, aResult, 0, 0, aTimeStep);
      else if (mCalculationType == "SecondYY")
        computeSecondMoment(aControl, aConfig, aResult, 1, 1, aTimeStep);
      else if (mCalculationType == "SecondZZ")
        computeSecondMoment(aControl, aConfig, aResult, 2, 2, aTimeStep);
      else
        throw std::runtime_error("In 'MassMoment.hpp' specified calculation type not implemented.");
    }

    /**************************************************************************/
    void computeStructuralMass(const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                               const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                               Plato::ScalarVectorT<ResultScalarType> & aResult,
                               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);

        aResult(aCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume );

      }, "mass calculation");
    }

    /**************************************************************************/
    void computeFirstMoment(const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                            const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                            Plato::ScalarVectorT<ResultScalarType> & aResult,
                            Plato::OrdinalType aComponent,
                            Plato::Scalar aTimeStep = 0.0) const 
    /**************************************************************************/
    {
      assert(aComponent < mSpaceDim);

      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubPoint  = tCubatureRule.getCubPointsCoords();
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();
      auto tNumPoints = tCubatureRule.getNumCubPoints();

      Plato::ScalarMultiVectorT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, mSpaceDim);
      mapQuadraturePoint(tCubPoint, aConfig, tMappedPoints);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(tCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControl);

        ConfigScalarType tMomentArm = tMappedPoints(tCellOrdinal, aComponent);

        aResult(tCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume * tMomentArm );

      }, "first moment calculation");
    }


    /**************************************************************************/
    void computeSecondMoment(const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                             const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                             Plato::ScalarVectorT<ResultScalarType> & aResult,
                             Plato::OrdinalType aComponent1,
                             Plato::OrdinalType aComponent2,
                             Plato::Scalar aTimeStep = 0.0) const 
    /**************************************************************************/
    {
      assert(aComponent1 < mSpaceDim);
      assert(aComponent2 < mSpaceDim);

      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubPoint  = tCubatureRule.getCubPointsCoords();
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();
      auto tNumPoints = tCubatureRule.getNumCubPoints();

      Plato::ScalarMultiVectorT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, mSpaceDim);
      mapQuadraturePoint(tCubPoint, aConfig, tMappedPoints);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(tCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControl);

        ConfigScalarType tMomentArm1 = tMappedPoints(tCellOrdinal, aComponent1);
        ConfigScalarType tMomentArm2 = tMappedPoints(tCellOrdinal, aComponent2);
        ConfigScalarType tSecondMoment  = tMomentArm1 * tMomentArm2;

        aResult(tCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume * tSecondMoment );

      }, "second moment calculation");
    }

    /******************************************************************************/
    void mapQuadraturePoint(const Plato::ScalarVector & aRefPoint,
                            const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                            Plato::ScalarMultiVectorT<ConfigScalarType> & aMappedPoints) const
    /******************************************************************************/
    {
      Plato::OrdinalType tNumCells  = mMesh.nelems();

      Kokkos::deep_copy(aMappedPoints, static_cast<ConfigScalarType>(0.0));

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal) {
        Plato::OrdinalType tNodeOrdinal;
        Plato::Scalar tFinalNodeValue = 1.0;
        for (tNodeOrdinal = 0; tNodeOrdinal < mSpaceDim; ++tNodeOrdinal)
        {
          Plato::Scalar tNodeValue = aRefPoint(tNodeOrdinal);
          tFinalNodeValue -= tNodeValue;
          for (Plato::OrdinalType tDim = 0; tDim < mSpaceDim; ++tDim)
          {
            aMappedPoints(tCellOrdinal,tDim) += tNodeValue * aConfig(tCellOrdinal,tNodeOrdinal,tDim);
          }
        }
        tNodeOrdinal = mSpaceDim;
        for (Plato::OrdinalType tDim = 0; tDim < mSpaceDim; ++tDim)
        {
          aMappedPoints(tCellOrdinal,tDim) += tFinalNodeValue * aConfig(tCellOrdinal,tNodeOrdinal,tDim);
        }
      }, "map single quadrature point to physical domain");
    }
};
// class MassMoment

}
// namespace Plato

#ifdef PLATO_1D
extern template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif