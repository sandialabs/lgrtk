#pragma once

namespace lgr {

class input;
class state;

void update_badness(input const& in, state& s);
void adapt(state& s);

}
