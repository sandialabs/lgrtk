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

#include <FieldDB.hpp>
#include <Fields.hpp>

namespace {
TEUCHOS_UNIT_TEST(FieldDB, Init)
{
  static const int spaceDim = 3;
  typedef lgr::Fields<spaceDim> FieldT;
  typedef lgr::FieldDB<FieldT::array_type>       db1T;
  typedef lgr::FieldDB<FieldT::elem_vector_type> db2T;
  
  lgr::FieldDB_Finalize<spaceDim>();

  db1T &db1 = lgr::FieldDB<FieldT::array_type>::Self();
  db2T &db2 = lgr::FieldDB<FieldT::elem_vector_type>::Self();
  db1T &db3 = lgr::FieldDB<FieldT::array_type>::Self();

  FieldT::array_type       A1("A1",10);
  FieldT::array_type       A2("A2",10);
  FieldT::array_type       A3("A3",10);
  FieldT::elem_vector_type B1("B1",10);

  // Standard insert as defined on std::map:
  db2.insert(std::make_pair("UB1",B1));

  db1["A1"]=A1;
  db1["A2"]=A2;
  db1["A3"]=A3;

  // What do we get..
  TEST_EQUALITY_CONST(db1.size(),3);
  TEST_EQUALITY_CONST(db2.size(),1);
  TEST_EQUALITY_CONST(db3.size(),3);
  TEST_ASSERT(db3.find("A1")!=db3.end())
  lgr::FieldDB_Finalize<spaceDim>();
}
}
