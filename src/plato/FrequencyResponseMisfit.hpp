/*
 * FrequencyResponseMisfit.hpp
 *
 *  Created on: May 24, 2018
 */

#ifndef SRC_PLATO_FREQUENCYRESPONSEMISFIT_HPP_
#define SRC_PLATO_FREQUENCYRESPONSEMISFIT_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_defines.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"

#include "plato/WorksetBase.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/SimplexStructuralDynamics.hpp"
#include "plato/ComputeFrequencyResponseMisfit.hpp"

namespace Plato
{

template<typename EvaluationType>
class FrequencyResponseMisfit :
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public AbstractScalarFunction<EvaluationType>
{
private:
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mComplexSpaceDim;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerCell;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numDofsPerNode;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::m_numNodesPerCell;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Plato::ScalarMultiVector mExpStates;
    std::vector<Plato::Scalar> mTimeSteps;

    Plato::VectorEntryOrdinal<EvaluationType::SpatialDim, m_numDofsPerNode> mStateEntryOrdinal;

public:
    /*************************************************************************/
    explicit FrequencyResponseMisfit(Omega_h::Mesh& aMesh,
                                     Omega_h::MeshSets& aMeshSets, 
                                     Plato::DataMap aDataMap,
                                     Teuchos::ParameterList & aParamList) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Frequency Response Misfit"),
            mExpStates(),
            mTimeSteps(),
            mStateEntryOrdinal(Plato::VectorEntryOrdinal<EvaluationType::SpatialDim, m_numDofsPerNode>(&aMesh))
    /*************************************************************************/
    {
        this->readTimeSteps(aParamList);
        this->readExperimentalData(aMesh, aParamList);
    }

    /*************************************************************************/
    explicit FrequencyResponseMisfit(Omega_h::Mesh& aMesh,       
                                     Omega_h::MeshSets& aMeshSets,
                                     Plato::DataMap aDataMap,
                                     const std::vector<Plato::Scalar> & aTimeSteps,
                                     const Plato::ScalarMultiVector & aExpStates) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Frequency Response Misfit"),
            mExpStates(aExpStates),
            mTimeSteps(aTimeSteps),
            mStateEntryOrdinal(Plato::VectorEntryOrdinal<EvaluationType::SpatialDim, m_numDofsPerNode>(&aMesh))
    /*************************************************************************/
    {
    }

    /*************************************************************************/
    virtual ~FrequencyResponseMisfit()
    /*************************************************************************/
    {
    }

    /*************************************************************************
     * Evaluate f(u,z)=\frac{1}{2}(u_i, where u denotes
     * states, z denotes controls, K denotes the stiffness matrix and M denotes
     * the mass matrix.
     **************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aStates,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControls,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResults,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        // Find input frequency argument in frequency array and return its index
        Plato::OrdinalType tIndex =
                std::distance(mTimeSteps.begin(), std::find(mTimeSteps.begin(), mTimeSteps.end(), aTimeStep));

        // Get experimental measurements for this frequency and construct its workset
        auto tNumCells = aStates.extent(0);
        auto tMyExpStates = Kokkos::subview(mExpStates, tIndex, Kokkos::ALL());
        Plato::ScalarMultiVector tExpStatesWorkSet("ExpStatesWorkSet", tNumCells, m_numDofsPerCell);
        Plato::workset_state_scalar_scalar<m_numDofsPerNode, m_numNodesPerCell>
            (tNumCells, mStateEntryOrdinal, tMyExpStates, tExpStatesWorkSet);
        assert(tExpStatesWorkSet.size() == aStates.size());

        Plato::ComputeFrequencyResponseMisfit<EvaluationType::SpatialDim> tComputeMisfit;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeMisfit(aCellOrdinal, tExpStatesWorkSet, aStates, aResults);
        }, "Objective::FrequencyResponseMisfit");
    }

    /**************************************************************************/
    void readExperimentalData(Omega_h::Mesh& aMesh, Teuchos::ParameterList & aParamList)
    /**************************************************************************/
    {
        if(aParamList.isSublist("Experimental Data") == true)
        {
            auto tExpDataParams = aParamList.sublist("Experimental Data");
            assert(tExpDataParams.isParameter("Names"));
            assert(tExpDataParams.isParameter("Index"));
            if(tExpDataParams.isParameter("Names") == false)
            {
                std::ostringstream tErrorMessage;
                tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                        << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__
                        << ", MESSAGE: USER DID NOT DEFINE NAMES ARRAY INSIDE SUBLIST = EXPERIMENTAL DATA."
                        << " CHECK INPUT FILE. **************\n\n";
                throw std::runtime_error(tErrorMessage.str().c_str());
            }
            auto tNames = tExpDataParams.get<Teuchos::Array<std::string>>("Names");

            if(tExpDataParams.isParameter("Index") == false)
            {
                std::ostringstream tErrorMessage;
                tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                        << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__
                        << ", MESSAGE: USER DID NOT DEFINE INDEX ARRAY INSIDE SUBLIST = EXPERIMENTAL DATA."
                        << " CHECK INPUT FILE. **************\n\n";
                throw std::runtime_error(tErrorMessage.str().c_str());
            }
            auto tIndices = tExpDataParams.get<Teuchos::Array<Plato::OrdinalType>>("Index");

            if(tIndices.size() != tNames.size())
            {
                std::ostringstream tErrorMessage;
                tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                        << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__
                        << ", MESSAGE: DIMENSION MISSMATCH. USER DEFINED INDEX AND NAMES ARRAYS IN SUBLIST = "
                        << " EXPERIMENTAL DATA SHOULD HAVE THE SAME SIZE. CHECK INPUT FILE. **************\n\n";
                throw std::runtime_error(tErrorMessage.str().c_str());
            }

            this->readExperimentalFields(aMesh, tNames, tIndices);
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: ARRAY WITH EXPERIMENTAL DATA FIELD NAMES WAS NOT DEFINED IN THE INPUT FILE."
                    << " USER SHOULD PROVIDE EXPERIMENTAL DATA FIELD NAMES INFORMATION IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }

    /**************************************************************************/
    void readExperimentalFields(Omega_h::Mesh& aMesh,
                                const Teuchos::Array<std::string> & aNames,
                                const Teuchos::Array<Plato::OrdinalType> & aIndices)
    /**************************************************************************/
    {
        const Plato::OrdinalType tNumInputExpFields = aNames.size();
        const Plato::OrdinalType tExpectedNumInputExpFields = mComplexSpaceDim * EvaluationType::SpatialDim;
        if(tNumInputExpFields != tExpectedNumInputExpFields)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << ", MESSAGE: PLATO EXPECTED " << tExpectedNumInputExpFields
                    << " EXPERIMENTAL DATA FIELDS. USER DEFINED " << tNumInputExpFields
                    << " EXPERIMENTAL DATA FIELDS IN THE INPUT FILE. USER SHOULD DEFINE " << tExpectedNumInputExpFields
                    << " EXPERIMENTAL DATA FIELD NAMES IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        const Plato::OrdinalType tNumVertices = aMesh.nverts();
        const Plato::OrdinalType tNumTimeSteps = mTimeSteps.size();
        const Plato::OrdinalType tNumStates = tNumVertices * m_numDofsPerNode;
        mExpStates = Plato::ScalarMultiVector("ExpStates", tNumTimeSteps, tNumStates);

        for(Plato::OrdinalType tFieldIndex = 0; tFieldIndex < tNumInputExpFields; tFieldIndex++)
        {
            auto tExpStates = mExpStates;
            auto tNumDofsPerNode = m_numDofsPerNode;
            auto tMyName = aNames[tFieldIndex];
            auto tMyDof = aIndices[tFieldIndex];
            // TODO: FINISH IMPLEMENTATION, I NEED TO MAKE SURE THAT I AM READING EACH TIME STEPS
            Plato::OrdinalType tMyTimeStep = 0;
            auto tMyExpStates = Kokkos::subview(mExpStates, tMyTimeStep, Kokkos::ALL());
            auto tMyInputExpData = aMesh.get_array < Omega_h::Real > (Omega_h::VERT, tMyName).view();

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVertices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {
                Plato::OrdinalType tStride = tNumDofsPerNode * aOrdinal;
                tMyExpStates(tStride + tMyDof) = tMyInputExpData(aOrdinal);
            }, "FRF_Objective::readFields");
        }
    }

private:
    /**************************************************************************/
    void readTimeSteps(Teuchos::ParameterList & aParamList)
    /**************************************************************************/
    {
        if(aParamList.isSublist("Frequency Steps") == true)
        {
            auto tFreqParams = aParamList.sublist("Frequency Steps");
            assert(tFreqParams.isParameter("Values"));
            auto tFreqValues = tFreqParams.get<Teuchos::Array<Plato::Scalar>>("Values");

            const Plato::OrdinalType tNumFrequencies = tFreqValues.size();
            mTimeSteps.resize(tNumFrequencies);
            for(Plato::OrdinalType tIndex = 0; tIndex < tNumFrequencies; tIndex++)
            {
                mTimeSteps[tIndex] = tFreqValues[tIndex];
            }
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << ", MESSAGE: FREQUENCY ARRAY WAS NOT DEFINED IN THE INPUT FILE."
                    << " USER SHOULD PROVIDE FREQUENCY ARRAY INFORMATION IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
};
// class FrequencyResponseMisfit

} // namespace Plato

#endif /* SRC_PLATO_FREQUENCYRESPONSEMISFIT_HPP_ */
