#pragma once

namespace lgr {

struct otm_scope {
  HPC_NOINLINE HPC_HOST otm_scope();
  HPC_NOINLINE HPC_HOST ~otm_scope();
};

bool otm_j2_uniaxial_patch_test();
bool otm_j2_nu_zero_patch_test();
bool otm_cylindrical_flyer();

}
