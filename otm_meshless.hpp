#pragma once

namespace lgr {

class state;
class input;

void otm_initialize_V(state& s);
void otm_initialize_grad_val_N(state& s);
void otm_update_nodal_internal_force(state& s);
void otm_update_nodal_external_force(state& s);
void otm_update_nodal_force(state& s);
void otm_lump_nodal_mass(state& s);
void otm_update_reference(state& s);
}
