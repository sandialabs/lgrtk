/*
 * PlatoTypes.hpp
 *
 *  Created on: Jul 12, 2018
 */

#ifndef SRC_PLATO_PLATOTYPES_HPP_
#define SRC_PLATO_PLATOTYPES_HPP_

#include <Kokkos_Core.hpp>

namespace Plato
{

using Scalar = double;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using Layout = Kokkos::LayoutRight;

} // namespace Plato

#endif /* SRC_PLATO_PLATOTYPES_HPP_ */
