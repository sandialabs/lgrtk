/*
//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINEAR_ELASTOSTATICS_DRIVER_HPP
#define LINEAR_ELASTOSTATICS_DRIVER_HPP

#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <memory>

#include <Teuchos_ParameterList.hpp>
#include <AdaptRecon.hpp>
#include <Omega_h_teuchos.hpp>


#include <CrsLinearProblem.hpp>
#include "VizOutput.hpp"
#include <MeshFixture.hpp>
#include "ExplicitFunctors.hpp"
#include "MaterialModels.hpp"
#include "Fields.hpp"
#include "FieldDB.hpp"
#include "ElastostaticSolve.hpp"
#include "EssentialBCs.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

//----------------------------------------------------------------------------

namespace lgr {

namespace LinearElastostatics {

template<int SpatialDim>
void run( 
    Teuchos::ParameterList&   problem,
    const lgr::FEMesh<SpatialDim>& mesh,
    lgr::MeshIO&                   mesh_io,
    std::string const&        viz_path)
{

  mesh_io.computeSets();

  Teuchos::ParameterList fieldData;
  typedef lgr::Fields<SpatialDim> Fields;
  auto mesh_fields = Teuchos::rcp( new Fields( mesh, fieldData ) );
  const comm::Machine machine = mesh.machine ;

  Teuchos::RCP<Plato::LinearElastostatics::ElastostaticSolve<SpatialDim>>
    elastostaticSolve = Teuchos::rcp( new Plato::LinearElastostatics::ElastostaticSolve<SpatialDim>(problem, mesh_fields, machine) );


  // initialize storage for globals (stiffness, forcing, solution)
  //
  elastostaticSolve->initialize();


  // Parse and create essential boundary conditions
  //
  Plato::EssentialBCs<Plato::SimplexMechanics<SpatialDim>> ebc(problem.sublist("Essential Boundary Conditions", false));
  ebc.get(mesh_io.mesh_sets, 
          elastostaticSolve->getConstrainedDofs(),
          elastostaticSolve->getConstrainedValues());


  // Parse and create body loads
  //
  ::Plato::BodyLoads<SpatialDim> bl(problem.sublist("Body Loads", false));
  bl.get(*(mesh_io.getMesh()), elastostaticSolve->getRHS());


  // parse and apply natural boundary conditions
  //
  Plato::NaturalBCs<SpatialDim> nbc(problem.sublist("Natural Boundary Conditions", false));
  nbc.get(mesh_io.getMesh(), mesh_io.mesh_sets, elastostaticSolve->getRHS());

  elastostaticSolve->assemble();

  Teuchos::ParameterList &viz_pl = problem.sublist("Visualization", false);
  VizOutput ioWriter(mesh_io.getMesh(), viz_path, viz_pl, /*restart_time=*/0);

  // once ic's are handled through the IC machinery:
  // initialize_element<...>::apply(*mesh_fields, ic);

  lgr::initialize_node<SpatialDim>::apply(*mesh_fields);

  // get linear solver
  typedef lgr::CrsLinearProblem<Plato::OrdinalType> LinearSolver;
  Teuchos::RCP<LinearSolver>
    linearSolver = elastostaticSolve->getDefaultSolver(/*cgTol=*/1e-12, /*cgMaxIters=*/10000);

  linearSolver->solve();

  // copy into field
  auto lhs = elastostaticSolve->getLHS();
  const typename Fields::geom_array_type disp(Displacement<Fields>());
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,lhs.size()), LAMBDA_EXPRESSION(int dofOrdinal) {
    disp(dofOrdinal/SpatialDim, dofOrdinal%SpatialDim) = lhs(dofOrdinal);
  },"copy from LHS");

  mesh_fields->copyGeomToMesh("displacement", Displacement<Fields>());
  ioWriter.writeOutputFile( *mesh_fields,
                            /*cycle=*/1,
                            /*next_state=*/1,
                            /*current_time=*/1.0);

}
  
template <int SpatialDim>
void driver( 
    Omega_h::Library*       lib_osh,
    Teuchos::ParameterList &problem,
    lgr::comm::Machine           machine ,
    const std::string&      input_filename,
    const std::string&      viz_path )
{
  typedef MeshFixture<SpatialDim> fixture_type ;

  typedef typename fixture_type::FEMeshType mesh_type;

  auto& assoc_pl = problem.sublist("Associations");

  lgr::MeshIO mesh_io(
      lib_osh, input_filename, assoc_pl, machine,
      /*read_restart=*/false);

  mesh_type mesh = fixture_type::create( mesh_io, machine );

  run<SpatialDim>(problem, mesh, mesh_io, viz_path);
}


void driver( 
    Omega_h::Library*       lib_osh,
    Teuchos::ParameterList &problem,
    lgr::comm::Machine           machine ,
    const std::string&      input_filename,
    const std::string&      viz_path )
{
  const int spaceDim = problem.get<int>("Spatial Dimension",3);

  if (spaceDim == 3)
  {
    driver<3>(lib_osh, problem, machine, input_filename, viz_path);
    ::lgr::FieldDB_Finalize<3>();
  } else
  if (spaceDim == 2)
  {
    driver<2>(lib_osh, problem, machine, input_filename, viz_path);
    ::lgr::FieldDB_Finalize<2>();
  } else
  if (spaceDim == 1)
  {
    driver<1>(lib_osh, problem, machine, input_filename, viz_path);
    ::lgr::FieldDB_Finalize<1>();
  }
}

} // namespace LinearElastostatics

} // namespace lgr

#endif /* #ifndef LINEAR_ELASTOSTATICS_DRIVER_HPP */
