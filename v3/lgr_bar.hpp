#pragma once

namespace lgr {

class state;
class input;

void initialize_bar_V(state& s);
void initialize_bar_grad_N(state& s);
void update_bar_h_min(input const&, state& s);
void update_bar_h_art(state& s);

}
