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

#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "LGRTestHelpers.hpp"

#include <FEMesh.hpp>
#include <Fields.hpp>
#include <FieldDB.hpp>

namespace {

template <typename T, int n>
bool equal(const r3d::Few<T,n> v1, const r3d::Few<T,n> v2) {
  bool r = true;
  for (int i=0; i<n && r; ++i) r = v1[i] == v2[i];
  return r;
}

TEUCHOS_UNIT_TEST(r3d, intersect_two_tets)
{
  constexpr int dim  = 3;
  constexpr int vert = dim+1;
  constexpr int moment  =  2;  
  constexpr Plato::Scalar tol=1e-10;
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
  Plato::Scalar moments[Polynomial::nterms] = {}; 
  r3d::reduce<moment>(intersection, moments);

  Plato::Scalar check[Polynomial::nterms] = // From Mathematica
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

r3d::Few<r3d::Vector<3>,4> 
extract_tet(const Omega_h::Reals &coords, const Omega_h::LOs &elem_verts, const int e) {
  constexpr int dim  = 3;
  constexpr int vert = dim+1;
  r3d::Few<r3d::Vector<dim>,vert> tet;
  for (int v=0; v<vert; ++v) 
    for (int d=0; d<dim; ++d) 
      tet[v][d] = coords.get(dim*elem_verts.get(vert*e+v)+d);
  return tet;
}
Plato::Scalar dot (const Plato::Scalar *moments,const Plato::Scalar *quad_poly) {
  constexpr int dim     = 3;
  constexpr int moment  =  2;  
  typedef r3d::Polynomial<dim,moment> Polynomial;
  Plato::Scalar d = 0;
  for (int i=0; i<Polynomial::nterms; ++i) 
    d += moments[i]*quad_poly[i];
  return d;
}

TEUCHOS_UNIT_TEST(r3d, intersect_two_cavities)
{
  constexpr int dim     = 3;
  constexpr int vert    = dim+1;
  constexpr int moment  =  2;  
  constexpr Plato::Scalar tol=1e-10;
  typedef r3d::Few<r3d::Vector<dim>,vert> Tet;
  typedef r3d::Polytope<dim>              Polytope;
  typedef r3d::Polynomial<dim,moment>     Polynomial;
  
  // 1+2x+3y+4z+5xx+6xy+7xz+8yy+9yz+10zz
  const Plato::Scalar quad_poly[Polynomial::nterms] = {1,2,3,4,5,6,7,8,9,10}; 
  const Plato::Scalar check[3] = {69.3333333333333, 63.8333333333333, 18.6666666666667};

  const Teuchos::RCP<Omega_h::Library> libOmegaH = lgr::getLibraryOmegaH();
  Omega_h::Mesh omega_h_mesh_1 = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,1,2,2,2,2);
  Omega_h::Mesh omega_h_mesh_2 = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,2,1,3,3,3);

  const Omega_h::LOs elem_verts_1 = omega_h_mesh_1.ask_elem_verts();
  const Omega_h::LOs elem_verts_2 = omega_h_mesh_2.ask_elem_verts();
  
  const Omega_h::Reals coords_1 = omega_h_mesh_1.coords();
  const Omega_h::Reals coords_2 = omega_h_mesh_2.coords();

  const int nelem_1 = omega_h_mesh_1.nelems();
  const int nelem_2 = omega_h_mesh_2.nelems();

  Plato::Scalar volume = 0;
  for (int e1=0; e1<nelem_1; ++e1) {
    const Tet tet1 = extract_tet(coords_1, elem_verts_1, e1);
    Polytope intersection;
    r3d::intersect_simplices(intersection, tet1, tet1);
    Plato::Scalar moments[Polynomial::nterms] = {}; 
    r3d::reduce<moment>(intersection, moments);
    volume += dot (moments,quad_poly);
  }
  TEST_FLOATING_EQUALITY(volume,check[0],tol);

  volume = 0;
  for (int e2=0; e2<nelem_2; ++e2) {
    const Tet tet2 = extract_tet(coords_2, elem_verts_2, e2);
    Polytope intersection;
    r3d::intersect_simplices(intersection, tet2, tet2);
    Plato::Scalar moments[Polynomial::nterms] = {}; 
    r3d::reduce<moment>(intersection, moments);
    volume += dot (moments,quad_poly);
  }
  TEST_FLOATING_EQUALITY(volume,check[1],tol);

  volume = 0;
  for (int e1=0; e1<nelem_1; ++e1) {
    const Tet tet1 = extract_tet(coords_1, elem_verts_1, e1);
    for (int e2=0; e2<nelem_2; ++e2) {
      const Tet tet2 = extract_tet(coords_2, elem_verts_2, e2);
      Polytope intersection;
      r3d::intersect_simplices(intersection, tet1, tet2);
      Plato::Scalar moments[Polynomial::nterms] = {}; 
      r3d::reduce<moment>(intersection, moments);
      volume += dot (moments,quad_poly);
    }
  }
  TEST_FLOATING_EQUALITY(volume,check[2],tol);
}

template<class T>
double intersect(const T src, const T trg) {
  constexpr int spaceDim= 2;
  constexpr int moment  = 1;
  typedef r3d::Polytope<spaceDim>                Polytope;
  typedef r3d::Polynomial<spaceDim,moment>     Polynomial;

  Polytope intersection;
  r3d::intersect_simplices(intersection, src, trg);
  double moments[Polynomial::nterms] = {0.}; 
  r3d::reduce<moment>(intersection, moments);

  const double measure = r3d::measure(intersection);
  std::cout<<" **** measure:"<<measure<<" nverts:"<<intersection.nverts<<std::endl;

  return measure;
}
template<class T>
void print(const T src, const T trg) {
  std::cout<<" Intersect src:"<<"{{"
      <<src[0][0]<<","<<src[0][1]<<"},{"
      <<src[1][0]<<","<<src[1][1]<<"},{"
      <<src[2][0]<<","<<src[2][1]<<"}}"
      <<std::endl;
  std::cout<<" Intersect trg:"<<"{{"
      <<trg[0][0]<<","<<trg[0][1]<<"},{"
      <<trg[1][0]<<","<<trg[1][1]<<"},{"
      <<trg[2][0]<<","<<trg[2][1]<<"}}"
      <<std::endl;
}

TEUCHOS_UNIT_TEST( r3d, 2D )
{
  constexpr double tol=1e-10;
  constexpr int spaceDim= 2;
  constexpr int vert    = spaceDim+1;
  typedef r3d::Few<r3d::Vector<spaceDim>,vert> Tri3;
  std::cout<<std::endl;
  {
    const Tri3 src = {{0.0,1.5},{0.5,3.0},{0.0l,3.0}};
    const Tri3 trg = {{0.5,1.5},{0.0,3.0},{0.0l,1.5}};
    const double check = 0.1875;
    print (src,trg);
      
    const double moment = intersect(src,trg);
    
    TEST_FLOATING_EQUALITY(moment,check,tol);
  }
  {
    const Tri3 src = {{0.0,0.0},{0.0,1.0},{1.0,1.0}};
    const Tri3 trg = {{0.0,0.0},{0.0,1.0},{1.0,1.0}};
    const double check = 0.5;
    print (src,trg);
      
    const double moment = intersect(src,trg);

    TEST_FLOATING_EQUALITY(moment,check,tol);
  }
  {
    const Tri3 src = {{0.0,1.0},{0.0,2.0},{1.0,2.0}};
    const Tri3 trg = {{0.0,1.0},{0.0,2.0},{0.5,2.0}};
    print (src,trg);
      
    const double moment = intersect(src,trg);

    TEST_FLOATING_EQUALITY(moment,0.5,tol);
  }
  {
    const Tri3 src = {{0.5,0.0},{0.5,1.0},{0.0l,0.0}};
    const Tri3 trg = {{0.5,1.0},{0.0,0.0},{1.0l,1.0}};
    const double check = 0.125;
    print (src,trg);
      
    const double moment = intersect(src,trg);
    
    TEST_FLOATING_EQUALITY(moment,check,tol);
  }
}
}
