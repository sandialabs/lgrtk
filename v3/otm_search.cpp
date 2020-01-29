#include <Kokkos_Core.hpp>

namespace lgr {
namespace search {

void initialize_otm_search()
{
  Kokkos::initialize();
}

void finalize_otm_search()
{
  Kokkos::finalize();
}

}
}
