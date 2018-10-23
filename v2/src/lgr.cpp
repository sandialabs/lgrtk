#include <lgr_run.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_cmdline.hpp>

int main(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv);
  auto world = lib.world();
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("input.yaml");
  if (!cmdline.parse_final(world, &argc, argv)) {
    return -1;
  }
  auto config_path = cmdline.get<std::string>("input.yaml");
  auto params = Omega_h::read_input(config_path);
  OMEGA_H_CHECK(params.used);
  lgr::run(world, params);
}
