#pragma once

#include "plato/Simplex.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include <Teuchos_ParameterList.hpp>

#include <math.h> // need PI

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class IntermediateDensityPenalty : public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                                   public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    
    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mPenaltyAmplitude;

  public:
    /**************************************************************************/
    IntermediateDensityPenalty(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList& aInputParams,
           std::string aFunctionName) :
           Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
           mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {
      auto tInputs = aInputParams.get<Teuchos::ParameterList>(aFunctionName);
      mPenaltyAmplitude = tInputs.get<Plato::Scalar>("Penalty Amplitude", 1.0);
    }

    /**************************************************************************
     * Unit testing constructor
    /**************************************************************************/
    IntermediateDensityPenalty(Omega_h::Mesh& aMesh, 
               Omega_h::MeshSets& aMeshSets,
               Plato::DataMap& aDataMap) :
               Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "IntermediateDensityPenalty"),
               mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {

    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const 
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      auto tPenaltyAmplitude = mPenaltyAmplitude;

      Plato::Scalar tOne = 1.0;
      Plato::Scalar tTwo = 2.0;
      Plato::Scalar tPi  = M_PI;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tBasisFunc = tCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);

        ResultScalarType tResult = tPenaltyAmplitude / tTwo * (tOne - cos(tTwo * tPi * tCellMass));

        aResult(aCellOrdinal) = tResult;

      }, "density penalty calculation");
    }
};
// class IntermediateDensityPenalty

}
// namespace Plato

#ifdef PLATO_1D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif