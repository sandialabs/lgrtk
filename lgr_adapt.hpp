#pragma once

namespace lgr {

class input;
class state;

void update_Q(input const& in, state& s);
void adapt(state& s);

}
