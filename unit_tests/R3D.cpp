//@HEADER
// ************************************************************************
//
//                        LGR v. 1.0
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

#include <iostream>

#include "Teuchos_UnitTestHarness.hpp"

#include <r3d.hpp>

namespace {

template <typename T, int n>
bool equal(const r3d::Few<T,n> v1, const r3d::Few<T,n> v2) {
  bool r = true;
  for (int i=0; i<n && r; ++i) r = v1[i] == v2[i];
  return r;
}

TEUCHOS_UNIT_TEST(r3d, intersect)
{
  constexpr int dim  = 3;
  constexpr int vert = dim+1;
  constexpr int moment  =  2;  
  typedef r3d::Few<r3d::Vector<dim>,vert> Tet;
  typedef r3d::Polytope<dim>              Polytope;
  typedef r3d::Polynomial<dim,moment>     Polynomial;
  Tet tet1 = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Tet tet2 = {{1, 0, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}};
  Tet ints = {{0, 0, 0}, {1, 0, 0}, {.5,.5,0}, {.5,0,.5}};
  Polytope intersection;
  r3d::intersect_simplices(intersection, tet1, tet2);

  const r3d::Vertex<dim> *verts = intersection.verts;

  for (int i=0; i<intersection.nverts; ++i) {
    bool r3d_intersection_correct = false;
    for (int j=0; j<ints.size && !r3d_intersection_correct; ++j) {
      r3d_intersection_correct = equal(ints[j],verts[i].pos);
    }   
    TEST_ASSERT(r3d_intersection_correct);
  }
  double moments[Polynomial::nterms] = {}; 
  r3d::reduce<moment>(intersection, moments);

  const double tol=1e-10;
  double check[Polynomial::nterms] = // From Mathematica
   {0.0416666666667,
    0.0208333333333,
    0.0052083333333,
    0.0052083333333,
    0.0114583333333,
    0.0026041666667,
    0.0026041666667,
    0.0010416666667,
    0.0005208333333,
    0.0010416666667};
  for (int i=0; i<Polynomial::nterms; ++i)
    TEST_FLOATING_EQUALITY(check[i],moments[i],tol);
}
}
