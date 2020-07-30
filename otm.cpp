#include <cassert>
#include <hpc_macros.hpp>
#include <otm_apps.hpp>
#include <string>

int
main()
{
  HPC_TRAP_FPE();
  lgr::otm_scope scope;
  if ((0)) lgr::otm_j2_nu_zero_patch_test();
  if ((0)) lgr::otm_j2_uniaxial_patch_test();
  if ((1)) lgr::otm_taylor();
  return 0;
}
