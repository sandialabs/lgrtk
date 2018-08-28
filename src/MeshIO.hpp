//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LGR_MESH_IO_HPP
#define LGR_MESH_IO_HPP

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <ParallelComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <string>

namespace lgr {

class MeshIO {
 public:
  MeshIO(
      Omega_h::Library*       lib_osh,
      std::string const&      input_filepath,
      Teuchos::ParameterList& assoc_pl,
      comm::Machine           machine,
      bool                    should_load_restart);

  MeshIO( Omega_h::Mesh& mesh );

  Omega_h::Mesh* getMesh() { return &mesh; }
  void           computeSets();
  void           markFixedVelocity(std::string const& ns_name, int comp);

  Omega_h::MeshSets mesh_sets;

  comm::Machine const&   getMachine() { return machine; }

 private:
  comm::Machine        machine;
  Omega_h::Mesh        mesh;
  Omega_h::Assoc       assoc;
};

}  // end namespace lgr

#endif
