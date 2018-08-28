//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MeshIO.hpp"
#include <iostream>
#include <Omega_h_teuchos.hpp>

namespace lgr {

MeshIO::MeshIO(
    Omega_h::Library*       lib_osh,
    std::string const&      input_filepath,
    Teuchos::ParameterList& assoc_pl,
    comm::Machine           machine_,
    bool                    should_load_restart)
    : machine(machine_), mesh(lib_osh) {
  if (comm::rank(machine) == 0) {
    std::cout << "Opening mesh file: " << input_filepath << '\n';
  }
  Omega_h::binary::read(input_filepath, lib_osh->world(), &mesh);
  if (!should_load_restart) {
    //load balance after reading the file
    mesh.balance();
  }
  mesh.set_parting(OMEGA_H_GHOSTED);
  Omega_h::update_assoc(&assoc, assoc_pl);
}

MeshIO::MeshIO( Omega_h::Mesh& mesh_in ) : mesh(mesh_in) {}

void MeshIO::computeSets() { mesh_sets = Omega_h::invert(&mesh, assoc); }

void MeshIO::markFixedVelocity(std::string const& ns_name, int comp) {
  auto& class_pairs = assoc[Omega_h::NODE_SET][ns_name];
  Omega_h::fix_momentum_velocity_verts(&mesh, class_pairs, comp);
}

} // end namespace lgr
