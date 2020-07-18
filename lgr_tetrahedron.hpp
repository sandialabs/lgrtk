#pragma once

namespace lgr {

class state;
class input;

void
initialize_tetrahedron_V(state& s);
void
initialize_tetrahedron_grad_N(state& s);
void
update_tetrahedron_h_min_inball(input const&, state& s);
void
update_tetrahedron_h_art(state& s);

}  // namespace lgr
