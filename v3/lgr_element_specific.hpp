#pragma once

namespace lgr {

class input;
class state;

void initialize_V(
    input const& in,
    state& s);
void initialize_grad_N(
    input const& in,
    state& s);
void update_h_min(
    input const& in,
    state& s);
void update_h_art(input const& in, state& s);
void update_nodal_mass(input const& in, state& s);

}
