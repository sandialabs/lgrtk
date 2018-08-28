/*
 * PlatoTestHelpers.hpp
 *
 *  Created on: Mar 31, 2018
 */

#ifndef PLATOTESTHELPERS_HPP_
#define PLATOTESTHELPERS_HPP_

#include <fstream>

#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "FEMesh.hpp"
#include "Fields.hpp"

#include "LGRTestHelpers.hpp"

#include "plato/PlatoStaticsTypes.hpp"

namespace PlatoUtestHelpers
{

/******************************************************************************/
//! returns all nodes matching x=0 on the boundary of the provided mesh
inline 
Omega_h::LOs getBoundaryNodes_x0(Teuchos::RCP<Omega_h::Mesh> & aMesh)
/******************************************************************************/
{
    Omega_h::Int tSpaceDim = aMesh->dim();
    try
    {
        if (tSpaceDim != static_cast<Omega_h::Int>(3))
        {
            std::ostringstream tErrorMsg;
            tErrorMsg << "\n\n ************* ERROR IN FILE: " << __FILE__<< ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << ", MESSAGE: " << "THIS METHOD IS ONLY IMPLEMENTED FOR 3D USE CASES."
                    << " *************\n\n";
            throw std::invalid_argument(tErrorMsg.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::cout << tErrorMsg.what() << std::flush;
        std::abort();
    }

    // because of the way that build_box does things, the x=0 nodes end up on a face which has label (2,12); the
    // x=1 nodes end up with label (2,14)
    const Omega_h::Int tVertexDim = 0;
    const Omega_h::Int tFaceDim = tSpaceDim - static_cast<Omega_h::Int>(1);
    Omega_h::Read<Omega_h::I8> x0Marks = Omega_h::mark_class_closure(aMesh.get(), tVertexDim, tFaceDim, 12);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(x0Marks);

    return (tLocalOrdinals);
}

/******************************************************************************/
// This one Tpetra likes; will have to check whether this works with Magma
// Sparse and AmgX or if we need to do something to factor this out

/*! Return a box (cube) mesh.

 @param spaceDim Spatial dimensions of the mesh to be created.
 @param meshWidth Number of mesh intervals through the thickness.
 */
inline
Teuchos::RCP<Omega_h::Mesh> getBoxMesh(Omega_h::Int aSpaceDim,
                                       Omega_h::Int aMeshWidth,
                                       Plato::Scalar aX_scaling = 1.0,
                                       Plato::Scalar aY_scaling = -1.0,
                                       Plato::Scalar aZ_scaling = -1.0)
/******************************************************************************/
{
    if(aY_scaling == -1.0)
    {
        aY_scaling = aX_scaling;
    }
    if(aZ_scaling == -1.0)
    {
        aZ_scaling = aY_scaling;
    }

    Omega_h::Int tNumX = 0, tNumY = 0, tNumZ = 0;
    if(aSpaceDim == 1)
    {
        tNumX = aMeshWidth;
    }
    else if(aSpaceDim == 2)
    {
        tNumX = aMeshWidth;
        tNumY = aMeshWidth;
    }
    else if(aSpaceDim == 3)
    {
        tNumX = aMeshWidth;
        tNumY = aMeshWidth;
        tNumZ = aMeshWidth;
    }

    Teuchos::RCP<Omega_h::Library> tLibOmegaH = lgr::getLibraryOmegaH();
    auto tOmegaH_mesh = Teuchos::rcp(new Omega_h::Mesh(Omega_h::build_box(tLibOmegaH->world(),
                                                                          OMEGA_H_SIMPLEX,
                                                                          aX_scaling,
                                                                          aY_scaling,
                                                                          aZ_scaling,
                                                                          tNumX,
                                                                          tNumY,
                                                                          tNumZ)));
    return (tOmegaH_mesh);
}

/******************************************************************************/
/*! Create an lgr::FEMesh from a Omega_h::Mesh

 @param meshOmegaH Input Omega_h mesh.
 */
template<int SpaceDim>
inline
lgr::FEMesh<SpaceDim> createFEMesh(const Teuchos::RCP<Omega_h::Mesh> & aMeshOmegaH)
/******************************************************************************/
{
    using DefaultFields = lgr::Fields<SpaceDim>;
    lgr::FEMesh<SpaceDim> tMesh;
    {
        // hopefully the following is enough to get things set up in the FEMesh

        tMesh.omega_h_mesh = aMeshOmegaH.get();
        tMesh.machine = lgr::getCommMachine();

        tMesh.resetSizes();

        // Element counts:

        const size_t tElemCount = tMesh.omega_h_mesh->nelems();
        const size_t tFaceCount = Omega_h::FACE <= tMesh.omega_h_mesh->dim() ? tMesh.omega_h_mesh->nfaces() : 0;

        //! list of total nodes on the rank
        size_t tNodeCount = tMesh.omega_h_mesh->nverts();

        // Allocate the initial arrays. Note that the mesh.reAlloc() function does the resize
        if(tNodeCount)
        {
            tMesh.node_coords = typename DefaultFields::node_coords_type("node_coords", tMesh.geom_layout);
        }

        if(tElemCount)
        {
            tMesh.elem_node_ids = typename DefaultFields::elem_node_ids_type("elem_node_ids", tElemCount);
	    tMesh.elem_face_ids = typename DefaultFields::elem_face_ids_type("elem_face_ids", tElemCount );
	    tMesh.elem_face_orientations = typename DefaultFields::elem_face_orient_type("elem_face_orientations", tElemCount );
        }
        if(tFaceCount)
        {
            tMesh.face_node_ids = typename DefaultFields::face_node_ids_type("face_node_ids", tFaceCount);
        }

        tMesh.updateMesh();
    }
    return (tMesh);
}

/******************************************************************************/
inline
void writeConnectivity(const Teuchos::RCP<Omega_h::Mesh> & aMeshOmegaH,
                       const std::string & aName,
                       const Omega_h::Int & aSpaceDim)
/******************************************************************************/
{
#ifndef KOKKOS_ENABLE_CUDA

    std::ofstream tOutfile(aName);

    auto tNumCells = aMeshOmegaH->nelems();
    auto tCells2nodes = aMeshOmegaH->ask_elem_verts();
    const Omega_h::Int tNumNodesPerCell = aSpaceDim + 1;
    for(Omega_h::Int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Omega_h::Int tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            tOutfile << tCells2nodes[tCellIndex * tNumNodesPerCell + tNodeIndex] << " ";
        }
        tOutfile << std::endl;
    }
    tOutfile.close();
#else
    (void)aMeshOmegaH;
    (void)aName;
    (void)aSpaceDim;
#endif
}

/******************************************************************************/
inline
void writeMesh(const Teuchos::RCP<Omega_h::Mesh> & aMeshOmegaH,
               const std::string & aName,
               const Omega_h::Int & aSpaceDim)
/******************************************************************************/
{
#ifndef KOKKOS_ENABLE_CUDA
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aName, aMeshOmegaH.get(), aSpaceDim);
    auto tTags = Omega_h::vtk::get_all_vtk_tags(aMeshOmegaH.get(),aSpaceDim);
    tWriter.write(static_cast<Omega_h::Real>(1.0), tTags);
#else
    (void)aMeshOmegaH;
    (void)aName;
    (void)aSpaceDim;
#endif
}

/******************************************************************************//**
 *
 * @brief Build 1D box mesh
 *
 * @param[in] aX x-dimension length
 * @param[in] aNx number of space in x-dimension
 *
**********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_1d_box_mesh(Omega_h::Real aX, Omega_h::LO aNx)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = lgr::getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh =
        std::make_shared<Omega_h::Mesh>(Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, 0., 0., aNx, 0, 0));
    return (tMesh);
}

/******************************************************************************//**
 *
 * @brief Build 2D box mesh
 *
 * @param[in] aX x-dimension length
 * @param[in] aY y-dimension length
 * @param[in] aNx number of space in x-dimension
 * @param[in] aNy number of space in y-dimension
 *
**********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_2d_box_mesh(Omega_h::Real aX, Omega_h::Real aY, Omega_h::LO aNx, Omega_h::LO aNy)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = lgr::getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh =
        std::make_shared<Omega_h::Mesh>(Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, aY, 0., aNx, aNy, 0));
    return (tMesh);
}

/******************************************************************************//**
 *
 * @brief Build 3D box mesh
 *
 * @param[in] aX x-dimension length
 * @param[in] aY y-dimension length
 * @param[in] aZ z-dimension length
 * @param[in] aNx number of space in x-dimension
 * @param[in] aNy number of space in y-dimension
 * @param[in] aNz number of space in z-dimension
 *
**********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_3d_box_mesh(Omega_h::Real aX,
                                                        Omega_h::Real aY,
                                                        Omega_h::Real aZ,
                                                        Omega_h::LO aNx,
                                                        Omega_h::LO aNy,
                                                        Omega_h::LO aNz)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = lgr::getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh =
        std::make_shared<Omega_h::Mesh>(Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, aY, aZ, aNx, aNy, aNz));
    return (tMesh);
}

/******************************************************************************//**
 *
 * @brief Get node ordinals associated with the boundary
 * @param[in] aMesh mesh data base
 *
**********************************************************************************/
inline Omega_h::LOs get_boundary_nodes(Omega_h::Mesh & aMesh)
{
    auto tSpaceDim  = aMesh.dim();

    Omega_h::Read<Omega_h::I8> tInteriorMarks = Omega_h::mark_by_class_dim(&aMesh, Omega_h::VERT, tSpaceDim);
    Omega_h::Read<Omega_h::I8> tBoundaryMarks = Omega_h::invert_marks(tInteriorMarks);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tBoundaryMarks);

    return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * @brief Get node ordinals associated with boundary edge y=0
 * @param[in] aMesh mesh data base
 *
**********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_y0(Omega_h::Mesh& aMesh)
{
  // the y=0 nodes end up on a face which has label (1,1);
  Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 1);
  Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
  return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * @brief Get node ordinals associated with boundary edge x=0
 * @param[in] aMesh mesh data base
 *
**********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_x0(Omega_h::Mesh& aMesh)
{
  // the x=0 nodes end up on an edge which has label (1,3);
  Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 3);
  Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
  return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * @brief Get node ordinals associated with boundary edge x=1
 * @param[in] aMesh mesh data base
 *
**********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_x1(Omega_h::Mesh& aMesh)
{
  // the x=1 nodes end up on an edge which has label (1,5),
  Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 5);
  Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
  return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * @brief Get node ordinals associated with boundary edge y=1
 * @param[in] aMesh mesh data base
 *
**********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_y1(Omega_h::Mesh& aMesh)
{
  // the x=1 nodes end up on an edge which has label (1,7).
  Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 7);
  Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
  return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * @brief Set point load
 * @param[in] aNodeOrdinal node ordinal associated with point load
 * @param[in] aNodeOrdinals collection of node ordinals associated with the entity (point, edge or surface) were point load is applied
 * @param[in] aValues values associated with point load
 * @param[in,out] aOutput global point load
 *
**********************************************************************************/
inline void set_point_load(const Omega_h::LO& aNodeOrdinal,
                           const Omega_h::LOs& aNodeOrdinals,
                           const Plato::ScalarMultiVector& aValues,
                           Plato::ScalarVector& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aValues.extent(0)), LAMBDA_EXPRESSION(const Plato::OrdinalType& aIndex)
    {
        auto tOffset = aIndex * aValues.extent(1);
        auto tNumDofsPerNode = aValues.extent(0) * aValues.extent(1);
        auto tMyNodeDof = tNumDofsPerNode * aNodeOrdinals[aNodeOrdinal];
        for(Plato::OrdinalType tDim = 0; tDim <  aValues.extent(1); tDim++)
        {
            auto tOutputIndex = tMyNodeDof + tOffset + tDim;
            aOutput(tOutputIndex) = aValues(aIndex, tDim);
        }
    }, "set point load");
}

/******************************************************************************//**
 *
 * @brief Set Dirichlet boundary conditions.
 *
 * @param[in] aNumDofsPerNode number of degrees of freedom per node
 * @param[in] aValue constant value associated with Dirichlet boundary conditions (only constant values are supported)
 * @param[in] aCoords coordinates associated with Dirichlet boundary conditions
 * @param[in,out] aDirichletValues values associated with Dirichlet boundary conditions
 *
**********************************************************************************/
inline void set_dirichlet_boundary_conditions(const Plato::OrdinalType& aNumDofsPerNode,
                                              const Plato::Scalar& aValue,
                                              const Omega_h::LOs& aCoords,
                                              Plato::LocalOrdinalVector& aDirichletDofs,
                                              Plato::ScalarVector& aDirichletValues)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aCoords.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType& aIndex)
    {
        auto tOffset = aIndex*aNumDofsPerNode;
        for (Plato::OrdinalType tDof = 0; tDof < aNumDofsPerNode; tDof++)
        {
            aDirichletDofs[tOffset + tDof] = aNumDofsPerNode*aCoords[aIndex] + tDof;
            aDirichletValues[tOffset + tDof] = aValue;
        }
    },"Dirichlet BC");
}

/******************************************************************************//**
 *
 * @brief Print ordinals' values
 * @param[in] aInput array of ordinals
 *
**********************************************************************************/
inline void print_ordinals(const Omega_h::LOs& aInput)
{
    auto tRange = aInput.size();
    Kokkos::parallel_for("print ordinals", tRange, LAMBDA_EXPRESSION(const int & aIndex)
    {
        printf("[%d]\n", aInput[aIndex]);
    });
}

/******************************************************************************//**
 *
 * @brief Print 1D coordinates associated with node ordinals
 * @param[in] aMesh mesg data base
 * @param[in] aNodeOrdinals array of node ordinals
 *
**********************************************************************************/
inline void print_1d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(1));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        printf("(%f)\n", tX);
    });
}

/******************************************************************************//**
 *
 * @brief Print 2D coordinates associated with node ordinals
 * @param[in] aMesh mesg data base
 * @param[in] aNodeOrdinals array of node ordinals
 *
**********************************************************************************/
inline void print_2d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(2));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        auto tY = tCoords[tEntryOffset + 1];
        printf("(%f,%f)\n", tX, tY);
    });
}

/******************************************************************************//**
 *
 * @brief Print 3D coordinates associated with node ordinals
 * @param[in] aMesh mesg data base
 * @param[in] aNodeOrdinals array of node ordinals
 *
**********************************************************************************/
inline void print_3d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(2));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        auto tY = tCoords[tEntryOffset + 1];
        auto tZ = tCoords[tEntryOffset + 2];
        printf("(%f,%f,%f)\n", tX, tY, tZ);
    });
}

} // namespace PlatoUtestHelpers

#endif /* PLATOTESTHELPERS_HPP_ */
