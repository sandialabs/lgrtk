#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

Scope::~Scope() {
  sim.fields.print_and_clear_set_fields();
}

}
