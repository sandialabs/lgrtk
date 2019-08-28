#include <Omega_h_map.hpp>
#include <lgr_ray_trace.hpp>
#include <lgr_scalar.hpp>
#include <lgr_simulation.hpp>
#include <Omega_h_simplex.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_array_ops.hpp>
#include <lgr_for.hpp>
#include <string>

namespace lgr {

#define LARGE_VALUE (1.0e8)

template <class Elem>
struct RayTracePair {
   double type_distance;
   double last_type_distance;
   double distance() {
       double rval = -1.0*std::abs(default_value)/default_value;
       return (std::abs(type_distance) < std::abs(default_value)) ? type_distance : rval;
   }
   double speed(const double dt) {
       if (dt == 0.0) return 0.0;
       if ((std::abs(type_distance)      < std::abs(default_value)) &&
           (std::abs(last_type_distance) < std::abs(default_value))) {
           return (type_distance - last_type_distance)/dt;
       }
       return 0.0;
   }
   void set_type(std::string typein) {
       type = typein;
       if (type == "max") default_value = -1.0*default_value;
       type_distance = default_value;
       last_type_distance = default_value;
   }
   double location[3] = {0.0};
   double direction[3] = {0.0};
   std::string name = "";
   std::string type = "min";
   double default_value = LARGE_VALUE;
};

template <class Elem>
struct RayTrace : public Model<Elem> {
  using Model<Elem>::sim;
  std::vector<RayTracePair<Elem>> pairs;
  RayTrace(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
      // Get user input
      if (pl.is_list("pairs")) {
          auto& pairs_pl = pl.get_list("pairs");

          for (int i=0; i < pairs_pl.size(); ++i) {
              RayTracePair<Elem> mdp;
              pairs.push_back(mdp);
          }

          for (int i=0; i < pairs_pl.size(); ++i) {
              if (pairs_pl.is_map(i)) {
                  std::size_t const k = std::size_t(i);

                  auto& pair_pl = pairs_pl.get_map(i);

                  // Name for later use
                  if (pair_pl.is<std::string>("name")) 
                      pairs.at(k).name = pair_pl.get<std::string>("name");

                  // Max or min
                  if (pair_pl.is<std::string>("type"))
                      pairs.at(k).set_type( pair_pl.get<std::string>("type") );

                  // Location
                  if (pair_pl.is_list("location")) {
                      auto& location_pl = pair_pl.get_list("location");
                      for (int j=0; j < Elem::dim; ++j) {
                          pairs.at(k).location[j] = location_pl.get<double>(j);
                      }
                  }

                  // Direction
                  if (pair_pl.is_list("direction")) {
                      auto& direction_pl = pair_pl.get_list("direction");
                      for (int j=0; j < Elem::dim; ++j) {
                          pairs.at(k).direction[j] = direction_pl.get<double>(j);
                      }
                  }
              }
          }
      }
  }
  std::uint64_t exec_stages() override final { return BEFORE_FIELD_UPDATE; }
  char const* name() override final { return "ray trace"; }
  void before_field_update() override final {

      auto const elems_to_nodes = this->get_elems_to_nodes();
      auto const nodes_to_x = sim.get(sim.position);

      const int faces_per_elem = Elem::dim+1;
      const int nodes_per_face = Elem::dim;

      // Loop over all pairs
      for (auto it = pairs.begin(); it != pairs.end(); ++it) {
          (*it).last_type_distance = (*it).type_distance;

          Omega_h::Write<double> elem_distances(this->elems(), (*it).default_value);

          auto elem_functor = OMEGA_H_LAMBDA(int const elem) {
            auto const elem_nodes = getnodes<Elem>(elems_to_nodes,elem);
            auto const x = getvecs<Elem>(nodes_to_x,elem_nodes);

            for (int face = 0; face < faces_per_elem; ++face) {
                double vert0[3] = {0.0};
                double vert1[3] = {0.0};
                double vert2[3] = {0.0};
                for (int node = 0; node < nodes_per_face; ++node) {
                    const int node_of_element = Omega_h::simplex_down_template(Elem::dim,
                                                                               Elem::dim-1,
                                                                               face,
                                                                               node);
                    auto const coordinates_of_node = x[node_of_element];

                    for (int j=0; j<Elem::dim; ++j) {
                       if (node == 0) vert0[j] = coordinates_of_node[j];
                       if (node == 1) vert1[j] = coordinates_of_node[j];
                       if (node == 2) vert2[j] = coordinates_of_node[j];
                    }
                    // In 2d, we can assume a triangle extruded in z by length of 2d side
                    if (Elem::dim == 2) {
                       double length = 0.0;
                       for (int j=0; j<Elem::dim; ++j) {
                           vert2[j] = vert1[j];
                           length += (vert1[j] - vert0[j])*(vert1[j] - vert0[j]);
                       }
                       vert2[2] = std::sqrt(length);
                    }
                }
                double distance = (*it).default_value;
                ::lgr::intersect_triangle1( (*it).location, 
                                            (*it).direction,
                                            vert0,
                                            vert1,
                                            vert2,
                                            distance);
                if ((*it).type == "max") {
                   if (distance > elem_distances[elem]) elem_distances[elem] = distance;
                } else {
                   if (distance < elem_distances[elem]) elem_distances[elem] = distance;
                }
            }
          };
          parallel_for(this->elems(), std::move(elem_functor));

          double distance;
          if ((*it).type == "max") {
             distance = Omega_h::get_max<double>(elem_distances);
          } else {
             distance = Omega_h::get_min<double>(elem_distances);
          }

          (*it).type_distance = distance;
          sim.globals.set((*it).name+" distance",(*it).distance());
          sim.globals.set((*it).name+" speed"   ,(*it).speed(sim.dt));
      }
  }
};

void setup_ray_trace(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("modifiers");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "ray trace") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new RayTrace<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
