#pragma once

namespace lgr {

class state;
class input;

void
initialize_triangle_V(state& s);
void
initialize_triangle_grad_N(state& s);
void
update_triangle_h_min_inball(input const&, state& s);
void
update_triangle_h_art(state& s);

}  // namespace lgr
