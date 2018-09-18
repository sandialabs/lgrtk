#include <lgr_run.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_cmdline.hpp>
#include <Omega_h_teuchos.hpp>

int main(int argc, char** argv) {
  Omega_h::Library lib(&argc, &argv);
  auto world = lib.world();
  Omega_h::CmdLine cmdline;
  cmdline.add_arg<std::string>("input.yaml");
  if (!cmdline.parse_final(world, &argc, argv)) {
    return -1;
  }
  auto config_path = cmdline.get<std::string>("input.yaml");
  auto comm_teuchos = Omega_h::make_teuchos_comm(world);
  auto params = Teuchos::ParameterList{};
  Omega_h::update_parameters_from_file(config_path, &params, *comm_teuchos);
  lgr::run(world, params);
}
