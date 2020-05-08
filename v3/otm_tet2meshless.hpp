#pragma once

namespace lgr {

class input;
class state;

void convert_tet_mesh_to_meshless(const input& in, state& st);
void otm_update_h(state& st);

}
