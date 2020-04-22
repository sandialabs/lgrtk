#include <cassert>
#include <string>
#include <hpc_macros.hpp>
#include <otm_apps.hpp>

int main()
{
  HPC_TRAP_FPE();
  if ((0)) lgr::otm_j2_nu_zero_patch_test();
  if ((1)) lgr::otm_j2_uniaxial_patch_test();
  return 0;
}
