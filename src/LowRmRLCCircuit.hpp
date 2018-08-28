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

#ifndef LOW_RM_RLC_CIRCUIT_HPP
#define LOW_RM_RLC_CIRCUIT_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <CellTools.hpp>
#include <CrsMatrix.hpp>
#include <Fields.hpp>
#include <MeshIO.hpp>
#include <ParallelComm.hpp>

namespace lgr {
using Scalar = double;

class LowRmRLCCircuit {
  /*
     
     We have an RLC circuit of the following form:
     
        1   i    2    3
     V0 •---L----•--R--•
        |              |
        C              |
        |              |
        •-----LGR----|
        4
     
     C is a capacitor, L is an inductor, and R is a resistor.
     
     The degrees of freedom are voltages corresponding to the numbered nodes, namely 
     v1, v2, v3, and v4, as well as the current i within the inductor.
     
     At time 0, we set initial conditions v1 = V0, v2=v3=v4=i=0.
     
     At each time step, we use backward Euler to solve using the current lumped circuit model element value, K11 (see LowRmPotentialSolve).
     
     It is the caller's responsibility to set K11 anytime it has changed.
     */
 private:
  Scalar _G, _L, _C;  // G is defined as 1/R
  Scalar _V0;

  Scalar _K11;

  // degrees of freedom:
  Scalar _v1, _v2, _v3, _v4, _i;

 public:
  LowRmRLCCircuit(Scalar R, Scalar L, Scalar C, Scalar V0);

  // ! set the lumped circuit element upper-left value (seet LowRmPotentialSolve).
  void setK11(Scalar K11) { _K11 = K11; }

  void setValues(Scalar V1, Scalar V2, Scalar V3, Scalar V4, Scalar I) {
    _v1 = V1;
    _v2 = V2;
    _v3 = V3;
    _v4 = V4;
    _i = I;
  }

  // ! Perform backward Euler time step of the given size
  void takeTimeStep(Scalar dt);

  Scalar v1() { return _v1; }
  Scalar v2() { return _v2; }
  Scalar v3() { return _v3; }
  Scalar v4() { return _v4; }

  Scalar i() { return _i; }
};
}  // namespace lgr

#endif
