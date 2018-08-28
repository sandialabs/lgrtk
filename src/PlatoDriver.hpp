
/*!
 *  todo:
 *   1.  Organize source.  what goes in plato directory?
 **/

#ifndef PLATO_DRIVER_HPP
#define PLATO_DRIVER_HPP

#include <string>
#include <vector>
#include <memory>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Omega_h_tag.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/PlatoUtilities.hpp"
#include "plato/PlatoProblemFactory.hpp"
//#include "plato/StructuralDynamicsOutput.hpp"

namespace Plato
{

template<const Plato::OrdinalType SpatialDim>
void output(Teuchos::ParameterList & aParamList,
            const std::string & aOutputFilePath,
            const Plato::ScalarMultiVector & aState,
            Omega_h::Mesh& aMesh)
{
    auto tProblemSpecs = aParamList.sublist("Plato Problem");
    assert(tProblemSpecs.isParameter("Physics"));
    auto tPhysics = tProblemSpecs.get < std::string > ("Physics");

    if(tPhysics == "Mechanical")
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        auto tNumDisp = tNumVertices * SpatialDim;
        Omega_h::Write<Omega_h::Real> tDisp(tNumDisp, "Displacement");
        
        const Plato::OrdinalType tStride = 0;
        Plato::copy<SpatialDim /*input_num_dof_per_node*/, SpatialDim /*output_num_dof_per_node*/>
            (tStride, tNumVertices, tSubView, tDisp);

        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);
        aMesh.add_tag(Omega_h::VERT, "Displacements", SpatialDim /*output_num_dof_per_node*/, Omega_h::Reals(tDisp));
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/1, /*current_time=*/1.0, tTags);
    }
    if(tPhysics == "StructuralDynamics")
    {
        /*assert(tPlatoProblemSpec.isSublist("Frequency Steps"));
        auto tFreqParams = tPlatoProblemSpec.sublist("Frequency Steps");
        auto tFrequencies = tFreqParams.get<Teuchos::Array<Plato::Scalar>>("Values");
        Plato::StructuralDynamicsOutput<SpatialDim> tOutput(aMesh, aOutputFilePath, tRestartTime);
        tOutput.output(tFrequencies, aState, aMesh);*/
    }
    else if(tPhysics == "Thermal")
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tSubView = Kokkos::subview(aState, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tNumVertices = aMesh.nverts();
        Omega_h::Write<Omega_h::Real> tTemp(tNumVertices, "Temperature");
        
        const Plato::OrdinalType tStride = 0;
        Plato::copy<1 /*input_num_dof_per_node*/, 1 /*output_num_dof_per_node*/>
            (tStride, tNumVertices, tSubView, tTemp);
        
        const Plato::Scalar tRestartTime = 0.;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aOutputFilePath, &aMesh, SpatialDim, tRestartTime);
        aMesh.add_tag(Omega_h::VERT, "Temperature", 1 /*output_num_dof_per_node*/, Omega_h::Reals(tTemp));
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpatialDim);
        tWriter.write(/*time_index*/1, /*current_time=*/1.0, tTags);
    }
}

template<const Plato::OrdinalType SpatialDim>
void run(Teuchos::ParameterList& aProblemSpec,
         Omega_h::Mesh& aMesh,
         Omega_h::MeshSets& aMeshSets,
         const std::string & aVizFilePath)
{
    // create mesh based density from host data
    std::vector<Plato::Scalar> tControlHost(aMesh.nverts(), 1.0);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tControlHostView(tControlHost.data(), tControlHost.size());
    auto tControl = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tControlHostView);

    // Solve Plato problem
    Plato::ProblemFactory<SpatialDim> tProblemFactory;
    std::shared_ptr<::Plato::AbstractProblem> tPlatoProblem = tProblemFactory.create(aMesh, aMeshSets, aProblemSpec);
    auto tSolution = tPlatoProblem->solution(tControl);

    Plato::output<SpatialDim>(aProblemSpec, aVizFilePath, tSolution, aMesh);
}

template<const Plato::OrdinalType SpatialDim>
void driver(Omega_h::Library* aLibOSH,
            Teuchos::ParameterList & aProblemSpec,
            const std::string& aInputFilename,
            const std::string& aVizFilePath)
{
    auto& tAssocParamList = aProblemSpec.sublist("Associations");
    auto tMesh = Omega_h::binary::read(aInputFilename, aLibOSH);
    tMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

    Omega_h::Assoc tAssoc;
    Omega_h::update_assoc(&tAssoc, tAssocParamList);
    auto tMeshSets = Omega_h::invert(&tMesh, tAssoc);
    
    Plato::run<SpatialDim>(aProblemSpec, tMesh, tMeshSets, aVizFilePath);
}

void driver(Omega_h::Library* aLibOmegaH,
            Teuchos::ParameterList & aProblemSpec,
            const std::string& aInputFilename,
            const std::string& aVizFilePath)
{
    const Plato::OrdinalType tSpaceDim = aProblemSpec.get<Plato::OrdinalType>("Spatial Dimension", 3);

    // Run Plato problem
    if(tSpaceDim == static_cast<Plato::OrdinalType>(3))
    {
        driver<3>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(2))
    {
        driver<2>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
    else if(tSpaceDim == static_cast<Plato::OrdinalType>(1))
    {
        driver<1>(aLibOmegaH, aProblemSpec, aInputFilename, aVizFilePath);
    }
}

} // namespace Plato

#endif /* #ifndef PLATO_DRIVER_HPP */

