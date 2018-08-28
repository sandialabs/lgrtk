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

#include "ParallelComm.hpp"
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace lgr {
namespace comm {

Machine::Machine(int *argc, char ***argv) {
  mpiSession = Teuchos::rcp(new Teuchos::GlobalMPISession(argc, argv));
  teuchosComm = Teuchos::DefaultComm<int>::getComm();
}

Machine::Machine(MPI_Comm& localComm) {
  mpiSession = Teuchos::null;
  teuchosComm = Teuchos::rcp(new Teuchos::MpiComm<int>(localComm));
}

Machine::Machine() : mpiSession(), teuchosComm() {}

unsigned size(Machine const& machine) {
  return (machine.teuchosComm)->getSize();
}

unsigned rank(Machine const& machine) {
  return (machine.teuchosComm)->getRank();
}

double max(Machine const& machine, double local) {
  double global = 0;
  Teuchos::reduceAll(
      *(machine.teuchosComm), Teuchos::REDUCE_MAX, 1, &local, &global);
  return global;
}

double min(Machine const& machine, double local) {
  double global = 0;
  Teuchos::reduceAll(
      *(machine.teuchosComm), Teuchos::REDUCE_MIN, 1, &local, &global);
  return global;
}

double sum(Machine const& machine, double local) {
  double global = 0;
  Teuchos::reduceAll(
      *(machine.teuchosComm), Teuchos::REDUCE_SUM, 1, &local, &global);
  return global;
}

void allReduce(
    Machine const& machine, int n, const double *local, double *global) {
  std::copy(local, local + n, global);
  Teuchos::reduceAll(
      *(machine.teuchosComm), Teuchos::REDUCE_SUM, n, local, global);
}

}}  //end namespace lgr::comm
