#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_vector.hpp>
#include <Omega_h_eigen.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/DivisionFunction.hpp"
#include "plato/LeastSquaresFunction.hpp"
#include "plato/WeightedSumFunction.hpp"
#include "plato/MassMoment.hpp"
#include "plato/AnalyzeMacros.hpp"
#include "plato/PlatoMathHelpers.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Mass properties function class
 **********************************************************************************/
template<typename PhysicsT>
class MassPropertiesFunction : public Plato::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::LeastSquaresFunction<PhysicsT>> mLeastSquaresFunction; /*!< Least squares function object */

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    Plato::Scalar mMaterialDensity; /*!< material density */

    Omega_h::Tensor<3> mInertiaRotationMatrix;
    Omega_h::Vector<3> mInertiaPrincipalValues;

    Omega_h::Tensor<3> mMinusRotatedParallelAxisTheoremMatrix;

    Plato::Scalar mMeshExtentX;
    Plato::Scalar mMeshExtentY;
    Plato::Scalar mMeshExtentZ;

	/******************************************************************************//**
     * @brief Initialization of Mass Properties Function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
        mMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

        createLeastSquaresFunction(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Create the least squares mass properties function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void createLeastSquaresFunction(Omega_h::Mesh& aMesh, 
                                    Omega_h::MeshSets& aMeshSets,
                                    Teuchos::ParameterList & aInputParams)
    {
        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tPropertyNamesTeuchos      = tProblemFunctionName.get<Teuchos::Array<std::string>>("Properties");
        auto tPropertyWeightsTeuchos    = tProblemFunctionName.get<Teuchos::Array<Plato::Scalar>>("Weights");
        auto tPropertyGoldValuesTeuchos = tProblemFunctionName.get<Teuchos::Array<Plato::Scalar>>("Gold Values");

        auto tPropertyNames      = tPropertyNamesTeuchos.toVector();
        auto tPropertyWeights    = tPropertyWeightsTeuchos.toVector();
        auto tPropertyGoldValues = tPropertyGoldValuesTeuchos.toVector();

        if (tPropertyNames.size() != tPropertyWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Properties' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            THROWERR(tErrorString)
        }

        if (tPropertyNames.size() != tPropertyGoldValues.size())
        {
            const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Properties'";
            THROWERR(tErrorString)
        }

        const bool tAllPropertiesSpecifiedByUser = allPropertiesSpecified(tPropertyNames);

        computeMeshExtent(aMesh);

        if (tAllPropertiesSpecifiedByUser)
            createAllMassPropertiesLeastSquaresFunction(
                aMesh, aMeshSets, tPropertyNames, tPropertyWeights, tPropertyGoldValues);
        else
            createItemizedLeastSquaresFunction(
                aMesh, aMeshSets, tPropertyNames, tPropertyWeights, tPropertyGoldValues);
    }

    /******************************************************************************//**
     * @brief Check if all properties were specified by user
     * @param [in] aPropertyNames names of properties specified by user 
     * @return bool indicating if all properties were specified by user
    **********************************************************************************/
    bool
    allPropertiesSpecified(const std::vector<std::string>& aPropertyNames)
    {
        // copy the vector since we sort it and remove items in this function
        std::vector<std::string> tPropertyNames(aPropertyNames.begin(), aPropertyNames.end());

        const unsigned int tUserSpecifiedNumberOfProperties = tPropertyNames.size();

        // Sort and erase duplicate entries
        std::sort( tPropertyNames.begin(), tPropertyNames.end() );
        tPropertyNames.erase( std::unique( tPropertyNames.begin(), tPropertyNames.end() ), tPropertyNames.end());

        // Check for duplicate entries from the user
        const unsigned int tUniqueNumberOfProperties = tPropertyNames.size();
        if (tUserSpecifiedNumberOfProperties != tUniqueNumberOfProperties)
        { THROWERR("User specified mass properties vector contains duplicate entries!") }

        if (tUserSpecifiedNumberOfProperties < 10) return false;

        std::vector<std::string> tAllPropertiesVector = 
                                 {"Mass","CGx","CGy","CGz","Ixx","Iyy","Izz","Ixy","Ixz","Iyz"};
        std::sort(tAllPropertiesVector.begin(), tAllPropertiesVector.end());
        
        std::set<std::string> tAllPropertiesSet(tAllPropertiesVector.begin(), tAllPropertiesVector.end());
        std::set<std::string>::iterator tSetIterator;

        // if number of unqiue user-specified properties does not equal all of them, return false
        if (tPropertyNames.size() != tAllPropertiesVector.size()) return false;

        for (Plato::OrdinalType tIndex = 0; tIndex < tPropertyNames.size(); ++tIndex)
        {
            const std::string tCurrentProperty = tPropertyNames[tIndex];

            // Check to make sure it is a valid property
            tSetIterator = tAllPropertiesSet.find(tCurrentProperty);
            if (tSetIterator == tAllPropertiesSet.end())
            {
                const std::string tErrorString = std::string("Specified mass property '") +
                tCurrentProperty + "' not implemented. Options are: Mass, CGx, CGy, CGz, " 
                                 + "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
                THROWERR(tErrorString)
            }

            // property vectors were sorted so check that the properties match in sequence
            if (tCurrentProperty != tAllPropertiesVector[tIndex])
            {
                printf("Property %s does not equal property %s \n", 
                       tCurrentProperty.c_str(), tAllPropertiesVector[tIndex].c_str());
                printf("If user specifies all mass properties, better performance may be experienced.\n");
                return false;
            }
        }

        return true;
    }


    /******************************************************************************//**
     * @brief Create a least squares function for all mass properties (inertia about gold CG)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aPropertyNames names of properties specified by user 
     * @param [in] aPropertyWeights weights of properties specified by user 
     * @param [in] aPropertyGoldValues gold values of properties specified by user 
    **********************************************************************************/
    void
    createAllMassPropertiesLeastSquaresFunction(Omega_h::Mesh& aMesh, 
                                                Omega_h::MeshSets& aMeshSets,
                                                const std::vector<std::string>& aPropertyNames,
                                                const std::vector<Plato::Scalar>& aPropertyWeights,
                                                const std::vector<Plato::Scalar>& aPropertyGoldValues)
    {
        printf("Creating all mass properties function.\n");
        mLeastSquaresFunction = std::make_shared<Plato::LeastSquaresFunction<PhysicsT>>(aMesh, mDataMap);
        std::map<std::string, Plato::Scalar> tWeightMap;
        std::map<std::string, Plato::Scalar> tGoldValueMap;
        for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < aPropertyNames.size(); ++tPropertyIndex)
        {
            const std::string   tPropertyName      = aPropertyNames[tPropertyIndex];
            const Plato::Scalar tPropertyWeight    = aPropertyWeights[tPropertyIndex];
            const Plato::Scalar tPropertyGoldValue = aPropertyGoldValues[tPropertyIndex];

            tWeightMap.insert(    std::pair<std::string, Plato::Scalar>(tPropertyName, tPropertyWeight   ) );
            tGoldValueMap.insert( std::pair<std::string, Plato::Scalar>(tPropertyName, tPropertyGoldValue) );
        }

        computeRotationAndParallelAxisTheoremMatrices(tGoldValueMap);

        // Mass
        mLeastSquaresFunction->allocateScalarFunctionBase(getMassFunction(aMesh, aMeshSets));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Mass")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("Mass")]);

        // CGx
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstX"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGx")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGx")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentX);

        // CGy
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGy")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentY);

        // CGz
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGz")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentZ);

        // Ixx
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "XX"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixx")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(0));

        // Iyy
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "YY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Iyy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(1));

        // Izz
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "ZZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Izz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(2));


        // Minimum Principal Moment of Inertia
        Plato::Scalar tMinPrincipalMoment = std::min(mInertiaPrincipalValues(0),
                                            std::min(mInertiaPrincipalValues(1), mInertiaPrincipalValues(2)));

        // Ixy
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "XY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);

        // Ixz
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "XZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);

        // Iyz
        mLeastSquaresFunction->allocateScalarFunctionBase(
              getMomentOfInertiaRotatedAboutCG(aMesh, aMeshSets, "YZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Iyz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);

        mLeastSquaresFunction->setGradientWRTStateIsZeroFlag(true);
    }

    /******************************************************************************//**
     * @brief Compute rotation and parallel axis theorem matrices
     * @param [in] aGoldValueMap gold value map
    **********************************************************************************/
    void
    computeRotationAndParallelAxisTheoremMatrices(std::map<std::string, Plato::Scalar>& aGoldValueMap)
    {
        const Plato::Scalar Mass = aGoldValueMap[std::string("Mass")];

        const Plato::Scalar Ixx = aGoldValueMap[std::string("Ixx")];
        const Plato::Scalar Iyy = aGoldValueMap[std::string("Iyy")];
        const Plato::Scalar Izz = aGoldValueMap[std::string("Izz")];
        const Plato::Scalar Ixy = aGoldValueMap[std::string("Ixy")];
        const Plato::Scalar Ixz = aGoldValueMap[std::string("Ixz")];
        const Plato::Scalar Iyz = aGoldValueMap[std::string("Iyz")];

        const Plato::Scalar CGx = aGoldValueMap[std::string("CGx")];
        const Plato::Scalar CGy = aGoldValueMap[std::string("CGy")]; 
        const Plato::Scalar CGz = aGoldValueMap[std::string("CGz")];

        Omega_h::Vector<3> tCGVector = Omega_h::vector_3(CGx, CGy, CGz);

        const Plato::Scalar tNormSquared = tCGVector * tCGVector;

        Omega_h::Tensor<3> tParallelAxisTheoremMatrix = 
            (tNormSquared * Omega_h::identity_tensor<3>()) - Omega_h::outer_product(tCGVector, tCGVector);

        Omega_h::Tensor<3> tGoldInertiaTensor = Omega_h::tensor_3(Ixx,Ixy,Ixz,
                                                                  Ixy,Iyy,Iyz,
                                                                  Ixz,Iyz,Izz);
        Omega_h::Tensor<3> tGoldInertiaTensorAboutCG = tGoldInertiaTensor - (Mass * tParallelAxisTheoremMatrix);
    
        auto tEigenPair = Omega_h::decompose_eigen_jacobi<3>(tGoldInertiaTensorAboutCG);
        mInertiaRotationMatrix = tEigenPair.q;
        mInertiaPrincipalValues = tEigenPair.l;

        printf("Eigenvalues of GoldInertiaTensor : %f, %f, %f\n", mInertiaPrincipalValues(0), 
            mInertiaPrincipalValues(1), mInertiaPrincipalValues(2));

        mMinusRotatedParallelAxisTheoremMatrix = -1.0 *
            (Omega_h::transpose<3,3>(mInertiaRotationMatrix) * (tParallelAxisTheoremMatrix * mInertiaRotationMatrix));
    }

    /******************************************************************************//**
     * @brief Create an itemized least squares function for user specified mass properties
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aPropertyNames names of properties specified by user 
     * @param [in] aPropertyWeights weights of properties specified by user 
     * @param [in] aPropertyGoldValues gold values of properties specified by user 
    **********************************************************************************/
    void
    createItemizedLeastSquaresFunction(Omega_h::Mesh& aMesh, 
                                       Omega_h::MeshSets& aMeshSets,
                                       const std::vector<std::string>& aPropertyNames,
                                       const std::vector<Plato::Scalar>& aPropertyWeights,
                                       const std::vector<Plato::Scalar>& aPropertyGoldValues)
    {
        printf("Creating itemized mass properties function.\n");
        mLeastSquaresFunction = std::make_shared<Plato::LeastSquaresFunction<PhysicsT>>(aMesh, mDataMap);
        for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < aPropertyNames.size(); ++tPropertyIndex)
        {
            const std::string   tPropertyName      = aPropertyNames[tPropertyIndex];
            const Plato::Scalar tPropertyWeight    = aPropertyWeights[tPropertyIndex];
            const Plato::Scalar tPropertyGoldValue = aPropertyGoldValues[tPropertyIndex];

            if (tPropertyName == "Mass")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMassFunction(aMesh, aMeshSets));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "CGx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstX"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentX);
            }
            else if (tPropertyName == "CGy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentY);
            }
            else if (tPropertyName == "CGz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentZ);
            }
            else if (tPropertyName == "Ixx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XX"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Iyy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "YY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Izz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "ZZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Ixy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Ixz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Iyz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "YZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else
            {
                const std::string tErrorString = std::string("Specified mass property '") +
                tPropertyName + "' not implemented. Options are: Mass, CGx, CGy, CGz, " 
                              + "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
                THROWERR(tErrorString)
            }
        }
        mLeastSquaresFunction->setGradientWRTStateIsZeroFlag(true);
    }

    /******************************************************************************//**
     * @brief Create the mass function only
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @return physics scalar function
    **********************************************************************************/
    std::shared_ptr<PhysicsScalarFunction<PhysicsT>>
    getMassFunction(Omega_h::Mesh& aMesh, 
                    Omega_h::MeshSets& aMeshSets)
    {
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tMassFunction =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, mDataMap);
        tMassFunction->setFunctionName("Mass Function");

        std::string tCalculationType = std::string("Mass");

        std::shared_ptr<Plato::MassMoment<Residual>> tValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, mDataMap);
        tValue->setMaterialDensity(mMaterialDensity);
        tValue->setCalculationType(tCalculationType);
        tMassFunction->allocateValue(tValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, mDataMap);
        tGradientU->setMaterialDensity(mMaterialDensity);
        tGradientU->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientU(tGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, mDataMap);
        tGradientZ->setMaterialDensity(mMaterialDensity);
        tGradientZ->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientZ(tGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, mDataMap);
        tGradientX->setMaterialDensity(mMaterialDensity);
        tGradientX->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientX(tGradientX);
        return tMassFunction;
    }

    /******************************************************************************//**
     * @brief Create the 'first mass moment divided by the mass' function (CG)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getFirstMomentOverMassRatio(Omega_h::Mesh& aMesh, 
                           Omega_h::MeshSets& aMeshSets, 
                           const std::string & aMomentType)
    {
        const std::string tNumeratorName = std::string("CG Numerator (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tNumerator =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, mDataMap);
        tNumerator->setFunctionName(tNumeratorName);

        std::shared_ptr<Plato::MassMoment<Residual>> tNumeratorValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, mDataMap);
        tNumeratorValue->setMaterialDensity(mMaterialDensity);
        tNumeratorValue->setCalculationType(aMomentType);
        tNumerator->allocateValue(tNumeratorValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tNumeratorGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, mDataMap);
        tNumeratorGradientU->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientU->setCalculationType(aMomentType);
        tNumerator->allocateGradientU(tNumeratorGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tNumeratorGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, mDataMap);
        tNumeratorGradientZ->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientZ->setCalculationType(aMomentType);
        tNumerator->allocateGradientZ(tNumeratorGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tNumeratorGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, mDataMap);
        tNumeratorGradientX->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientX->setCalculationType(aMomentType);
        tNumerator->allocateGradientX(tNumeratorGradientX);

        const std::string tDenominatorName = std::string("CG Mass Denominator (Moment type = ")
                                           + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tDenominator = 
             getMassFunction(aMesh, aMeshSets);
        tDenominator->setFunctionName(tDenominatorName);

        std::shared_ptr<Plato::DivisionFunction<PhysicsT>> tMomentOverMassRatioFunction =
             std::make_shared<Plato::DivisionFunction<PhysicsT>>(aMesh, mDataMap);
        tMomentOverMassRatioFunction->allocateNumeratorFunction(tNumerator);
        tMomentOverMassRatioFunction->allocateDenominatorFunction(tDenominator);
        tMomentOverMassRatioFunction->setFunctionName(std::string("CG ") + aMomentType);
        return tMomentOverMassRatioFunction;
    }

    /******************************************************************************//**
     * @brief Create the second mass moment function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aMomentType second mass moment type (XX, XY, YY, ...)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getSecondMassMoment(Omega_h::Mesh& aMesh, 
                        Omega_h::MeshSets& aMeshSets, 
                        const std::string & aMomentType)
    {
        const std::string tInertiaName = std::string("Second Mass Moment (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tSecondMomentFunction =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, mDataMap);
        tSecondMomentFunction->setFunctionName(tInertiaName);

        std::shared_ptr<Plato::MassMoment<Residual>> tValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, mDataMap);
        tValue->setMaterialDensity(mMaterialDensity);
        tValue->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateValue(tValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, mDataMap);
        tGradientU->setMaterialDensity(mMaterialDensity);
        tGradientU->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientU(tGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, mDataMap);
        tGradientZ->setMaterialDensity(mMaterialDensity);
        tGradientZ->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientZ(tGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, mDataMap);
        tGradientX->setMaterialDensity(mMaterialDensity);
        tGradientX->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientX(tGradientX);

        return tSecondMomentFunction;
    }


    /******************************************************************************//**
     * @brief Create the moment of inertia function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getMomentOfInertia(Omega_h::Mesh& aMesh, 
                       Omega_h::MeshSets& aMeshSets, 
                       const std::string & aAxes)
    {
        std::shared_ptr<Plato::WeightedSumFunction<PhysicsT>> tMomentOfInertiaFunction = 
               std::make_shared<Plato::WeightedSumFunction<PhysicsT>>(aMesh, mDataMap);
        tMomentOfInertiaFunction->setFunctionName(std::string("Inertia ") + aAxes);

        if (aAxes == "XX")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYY"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "YY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "ZZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYY"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "XY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXY"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "XZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "YZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else
        {
            const std::string tErrorString = std::string("Specified axes '") +
            aAxes + "' not implemented for moment of inertia calculation. " 
                          + "Options are: XX, YY, ZZ, XY, XZ, YZ";
            THROWERR(tErrorString)
        }

        return tMomentOfInertiaFunction;
    }

    /******************************************************************************//**
     * @brief Create the moment of inertia function about the CG in the principal coordinate frame
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getMomentOfInertiaRotatedAboutCG(Omega_h::Mesh& aMesh, 
                                     Omega_h::MeshSets& aMeshSets, 
                                     const std::string & aAxes)
    {
        std::shared_ptr<Plato::WeightedSumFunction<PhysicsT>> tMomentOfInertiaFunction = 
               std::make_shared<Plato::WeightedSumFunction<PhysicsT>>(aMesh, mDataMap);
        tMomentOfInertiaFunction->setFunctionName(std::string("InertiaRot ") + aAxes);

        std::vector<Plato::Scalar> tInertiaWeights(6);
        Plato::Scalar tMassWeight;

        getInertiaAndMassWeights(tInertiaWeights, tMassWeight, aAxes);
        for (unsigned int tIndex = 0; tIndex < 6; ++tIndex)
            tMomentOfInertiaFunction->appendFunctionWeight(tInertiaWeights[tIndex]);

        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "XX"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "YY"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "ZZ"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "XY"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "XZ"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aMesh, aMeshSets, "YZ"));

        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMassFunction(aMesh, aMeshSets));
        tMomentOfInertiaFunction->appendFunctionWeight(tMassWeight);

        return tMomentOfInertiaFunction;
    }

    /******************************************************************************//**
     * @brief Compute the inertia weights and mass weight for the inertia about the CG rotated into principal frame
     * @param [out] aInertiaWeights inertia weights
     * @param [out] aMassWeight mass weight
     * @param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
    **********************************************************************************/
    void
    getInertiaAndMassWeights(std::vector<Plato::Scalar> & aInertiaWeights, 
                             Plato::Scalar & aMassWeight, 
                             const std::string & aAxes)
    {
        const Plato::Scalar Q11 = mInertiaRotationMatrix(0,0);
        const Plato::Scalar Q12 = mInertiaRotationMatrix(0,1);
        const Plato::Scalar Q13 = mInertiaRotationMatrix(0,2);

        const Plato::Scalar Q21 = mInertiaRotationMatrix(1,0);
        const Plato::Scalar Q22 = mInertiaRotationMatrix(1,1);
        const Plato::Scalar Q23 = mInertiaRotationMatrix(1,2);

        const Plato::Scalar Q31 = mInertiaRotationMatrix(2,0);
        const Plato::Scalar Q32 = mInertiaRotationMatrix(2,1);
        const Plato::Scalar Q33 = mInertiaRotationMatrix(2,2);

        if (aAxes == "XX")
        {
            aInertiaWeights[0] = Q11*Q11;
            aInertiaWeights[1] = Q21*Q21;
            aInertiaWeights[2] = Q31*Q31;
            aInertiaWeights[3] = 2.0*Q11*Q21;
            aInertiaWeights[4] = 2.0*Q11*Q31;
            aInertiaWeights[5] = 2.0*Q21*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,0);
        }
        else if (aAxes == "YY")
        {
            aInertiaWeights[0] =  Q12*Q12;
            aInertiaWeights[1] =  Q22*Q22;
            aInertiaWeights[2] =  Q32*Q32;
            aInertiaWeights[3] =  2.0*Q12*Q22;
            aInertiaWeights[4] =  2.0*Q12*Q32;
            aInertiaWeights[5] =  2.0*Q22*Q32;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1,1);
        }
        else if (aAxes == "ZZ")
        {
            aInertiaWeights[0] =  Q13*Q13;
            aInertiaWeights[1] =  Q23*Q23;
            aInertiaWeights[2] =  Q33*Q33;
            aInertiaWeights[3] =  2.0*Q13*Q23;
            aInertiaWeights[4] =  2.0*Q13*Q33;
            aInertiaWeights[5] =  2.0*Q23*Q33;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(2,2);
        }
        else if (aAxes == "XY")
        {
            aInertiaWeights[0] =  Q11*Q12;
            aInertiaWeights[1] =  Q21*Q22;
            aInertiaWeights[2] =  Q31*Q32;
            aInertiaWeights[3] =  Q11*Q22 + Q12*Q21;
            aInertiaWeights[4] =  Q11*Q32 + Q12*Q31;
            aInertiaWeights[5] =  Q21*Q32 + Q22*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,1);
        }
        else if (aAxes == "XZ")
        {
            aInertiaWeights[0] =  Q11*Q13;
            aInertiaWeights[1] =  Q21*Q23;
            aInertiaWeights[2] =  Q31*Q33;
            aInertiaWeights[3] =  Q11*Q23 + Q13*Q21;
            aInertiaWeights[4] =  Q11*Q33 + Q13*Q31;
            aInertiaWeights[5] =  Q21*Q33 + Q23*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,2);
        }
        else if (aAxes == "YZ")
        {
            aInertiaWeights[0] =  Q12*Q13;
            aInertiaWeights[1] =  Q22*Q23;
            aInertiaWeights[2] =  Q32*Q33;
            aInertiaWeights[3] =  Q12*Q23 + Q13*Q22;
            aInertiaWeights[4] =  Q12*Q33 + Q13*Q32;
            aInertiaWeights[5] =  Q22*Q33 + Q23*Q32;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1,2);
        }
        else
        {
            const std::string tErrorString = std::string("Specified axes '") +
            aAxes + "' not implemented for inertia and mass weights calculation. " 
                          + "Options are: XX, YY, ZZ, XY, XZ, YZ";
            THROWERR(tErrorString)
        }
    }

public:
    /******************************************************************************//**
     * @brief Primary Mass Properties Function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    MassPropertiesFunction(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap & aDataMap,
                           Teuchos::ParameterList& aInputParams,
                           std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName),
            mMaterialDensity(1.0)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Compute the X, Y, and Z extents of the mesh (e.g. (X_max - X_min))
     * @param [in] aMesh mesh database
    **********************************************************************************/
    void
    computeMeshExtent(Omega_h::Mesh& aMesh)
    {
        Omega_h::Reals tNodeCoordinates = aMesh.coords();
        Omega_h::Int   tSpaceDim        = aMesh.dim();
        Omega_h::LO    tNumVertices     = aMesh.nverts();

        assert(tSpaceDim == 3);

        Plato::ScalarVector tXCoordinates("X-Coordinates", tNumVertices);
        Plato::ScalarVector tYCoordinates("Y-Coordinates", tNumVertices);
        Plato::ScalarVector tZCoordinates("Z-Coordinates", tNumVertices);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVertices), LAMBDA_EXPRESSION(const Plato::OrdinalType & tVertexIndex)
        {
            const Plato::Scalar x_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 0];
            const Plato::Scalar y_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 1];
            const Plato::Scalar z_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 2];

            tXCoordinates(tVertexIndex) = x_coordinate;
            tYCoordinates(tVertexIndex) = y_coordinate;
            tZCoordinates(tVertexIndex) = z_coordinate;
        }, "Fill vertex coordinate views");

        Plato::Scalar tXmin;
        Plato::Scalar tXmax;
        Plato::min(tXCoordinates, tXmin);
        Plato::max(tXCoordinates, tXmax);

        Plato::Scalar tYmin;
        Plato::Scalar tYmax;
        Plato::min(tYCoordinates, tYmin);
        Plato::max(tYCoordinates, tYmax);

        Plato::Scalar tZmin;
        Plato::Scalar tZmax;
        Plato::min(tZCoordinates, tZmin);
        Plato::max(tZCoordinates, tZmax);

        mMeshExtentX = std::abs(tXmax - tXmin);
        mMeshExtentY = std::abs(tYmax - tYmin);
        mMeshExtentZ = std::abs(tZmax - tZmin);
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        mLeastSquaresFunction->updateProblem(aState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate Mass Properties Function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::Scalar tFunctionValue = mLeastSquaresFunction->value(aState, aControl, aTimeStep);
        return tFunctionValue;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientX = mLeastSquaresFunction->gradient_x(aState, aControl, aTimeStep);
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientU = mLeastSquaresFunction->gradient_u(aState, aControl, aTimeStep);
        return tGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientZ = mLeastSquaresFunction->gradient_z(aState, aControl, aTimeStep);
        return tGradientZ;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class MassPropertiesFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<3>>;
#endif