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

#ifndef LGR_ERROR_HANDLING_HPP
#define LGR_ERROR_HANDLING_HPP

#include "Teuchos_Assert.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#define LGR_ASSERT(condition) TEUCHOS_ASSERT(condition)
// Use LGR_THROW_IF as a convenient wrapper to standard exception
// throwing. The message can uses a stream's operator<< so that a complicated
// message can be constructed.
/*
  Example:
   LGR_THROW_IF (!nearest.count(e.id()),
       "Processor:"<<comm::rank(machine)<<" Failed to find nearest id:("
       <<e.id().first<<":"<<e.id().second<<":"<<e.proc()<<")\n");
*/
#define LGR_THROW_IF(condition, message)                                     \
  TEUCHOS_TEST_FOR_EXCEPTION(condition, std::logic_error, message)
#define LGR_CATCH_STATEMENTS(verbose, success)                               \
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success)

namespace lgr {
inline void enable_floating_point_exceptions();
}

#ifdef LGR_CHECK_FPE
#ifdef LGR_USE_GNU_FPE
#define _GNU_SOURCE 1
#include <fenv.h>
inline void lgr::enable_floating_point_exceptions() {
  feclearexcept(FE_ALL_EXCEPT);
  // FE_INEXACT inexact result: rounding was necessary to store the result of an earlier floating-point operation
  // sounds like the above would happen in almost any floating point operation involving non-whole numbers ???
  // As for underflow, there are plenty of cases where we will have things like ((a + eps) - (a)) -> eps,
  // where eps can be arbitrarily close to zero (usually it would have been zero with infinite precision).
  feenableexcept(FE_ALL_EXCEPT - FE_INEXACT - FE_UNDERFLOW);
}
#else  // not GCC, fall back on XMM intrinsics
#include <xmmintrin.h>
// Intel system
inline void lgr::enable_floating_point_exceptions() {
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
}
#endif
#else  // don't check FPE
inline void lgr::enable_floating_point_exceptions() {}
#endif

#endif  // LGR_ERROR_HANDLING_HPP
