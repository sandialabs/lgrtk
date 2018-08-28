/*
 * StructuralDynamicsOutput.hpp
 *
 *  Created on: Aug 13, 2018
 */

#ifndef STRUCTURALDYNAMICSOUTPUT_HPP_
#define STRUCTURALDYNAMICSOUTPUT_HPP_

#include <memory>
#include <vector>
#include <cassert>
#include <unistd.h>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>

#include "plato/SimplexStructuralDynamics.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class StructuralDynamicsOutput: public Plato::SimplexStructuralDynamics<SpaceDim, NumControls>
{
private:
    static constexpr Plato::OrdinalType mSpatialDim = Plato::SimplexStructuralDynamics<SpaceDim>::m_numSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode = Plato::SimplexStructuralDynamics<SpaceDim>::m_numDofsPerNode;

    std::shared_ptr<Omega_h::vtk::Writer> mWriter;

private:
    void insert(const Omega_h::Int & aEntity,
                const std::string & aName,
                const Omega_h::Write<Omega_h::Real> & aData,
                Omega_h::Mesh& aMesh)
    {
        if(aMesh.has_tag(aEntity, aName) == false)
        {
            aMesh.add_tag(aEntity, aName, mSpatialDim /*num_dof_per_node*/, Omega_h::Reals(aData));
        }
        else
        {
            aMesh.set_tag(aEntity, aName, Omega_h::Reals(aData));
        }
    }

public:
    StructuralDynamicsOutput(Omega_h::Mesh& aMesh, Plato::Scalar aRestartFreq = 0) :
            mWriter(nullptr)
    {
        char tTemp[FILENAME_MAX];
        auto tFilePath = getcwd(tTemp, FILENAME_MAX) ? std::string( tTemp ) : std::string("");
        assert(tFilePath.empty() == false);
        mWriter = std::make_shared<Omega_h::vtk::Writer>(tFilePath, &aMesh, mSpatialDim, aRestartFreq);
    }

    StructuralDynamicsOutput(Omega_h::Mesh& aMesh, const std::string & aFilePath, Plato::Scalar aRestartFreq = 0) :
            mWriter(std::make_shared<Omega_h::vtk::Writer>(aFilePath, &aMesh, mSpatialDim, aRestartFreq))
    {
    }

    ~StructuralDynamicsOutput()
    {
    }

    template<typename ArrayT>
    void output(const ArrayT& tFreqArray,
                const Plato::ScalarMultiVector& aState,
                Omega_h::Mesh& aMesh)
    {
        auto tNumVertices = aMesh.nverts();
        auto tOutputNumDofs = tNumVertices * mSpatialDim;
        Omega_h::Write<Omega_h::Real> tRealDisp(tOutputNumDofs, "RealDisp");
        Omega_h::Write<Omega_h::Real> tImagDisp(tOutputNumDofs, "ImagDisp");

        auto tNumFrequencies = tFreqArray.size();
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumFrequencies; tIndex++)
        {
            auto tMyState = Kokkos::subview(aState, tIndex, Kokkos::ALL());
            Plato::copy<mNumDofsPerNode /*input_num_dof_per_node*/, mSpatialDim /*output_num_dof_per_node*/>
                (0 /*stride*/, tNumVertices, tMyState, tRealDisp);
            this->insert(Omega_h::VERT, "RealDisp", tRealDisp, aMesh);

            Plato::copy<mNumDofsPerNode /*input_num_dof_per_node*/, mSpatialDim /*output_num_dof_per_node*/>
                (mSpatialDim /*stride*/, tNumVertices, tMyState, tImagDisp);
            this->insert(Omega_h::VERT, "ImagDisp", tImagDisp, aMesh);

            auto tMyFreq = tFreqArray[tIndex];
            Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, mSpatialDim);
            auto tFreqIndex = tIndex + static_cast<Plato::OrdinalType>(1);
            mWriter->write(tFreqIndex, tMyFreq, tTags);
        }
    }
};
// class StructuralDynamicsOutput

} // namespace Plato

#endif /* STRUCTURALDYNAMICSOUTPUT_HPP_ */
