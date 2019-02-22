#include <lgr_setup.hpp>
#include <lgr_neo_hookean.hpp>

namespace lgr {

void add_builtin_setups(Setups& setups) {
  setups.material_models.push_back(setup_neo_hookean);
}

}
