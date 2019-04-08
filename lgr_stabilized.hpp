#pragma once

namespace lgr {

class input;
class state;

void update_sigma_with_p_h(state& s);
void update_v_prime(input const& in, state& s);
void update_q(input const& in, state& s);
void update_p_h_W(state& s);
void update_e_h_W(state& s);
void update_p_h_dot(state& s);
void update_e_h_dot(state& s);
void nodal_ideal_gas(input const& in, state& s);
void update_nodal_density(state& s);
void interpolate_K(state& s);
void interpolate_rho(state& s);

}
