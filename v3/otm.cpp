#include <cassert>
#include <string>
#include <hpc_macros.hpp>
#include <otm_meshless.hpp>

int main()
{
  HPC_TRAP_FPE();
  if ((1)) lgr::otm_j2_uniaxial_patch_test();
  return 0;
}
