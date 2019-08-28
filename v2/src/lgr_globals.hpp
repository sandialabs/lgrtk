#ifndef LGR_GLOBALS_HPP
#define LGR_GLOBALS_HPP

#include<string>
#include<vector>

namespace lgr {

struct Global {
  std::string name;
  double value;
  Global(std::string n, double v) : name(n), value(v) {}
};

struct Globals {
  std::vector<Global> data;
  void set(std::string name, double value) {
      // Check if name is defined
      bool notFound = true;
      for (auto it = data.begin(); it != data.end(); ++it) {
          if ((*it).name == name) {
             // This name has been found, so update the value
             notFound = false;
             (*it).value = value;
          }
      }
      // If name was not found, add new global
      if (notFound) {
         Global g(name,value);
         data.push_back(g);
      }
  }
  double get(std::string name) {
      // Check if name is defined
      for (auto it = data.begin(); it != data.end(); ++it) {
          if ((*it).name == name) {
             // This name has been found, so return value
             return (*it).value;
          }
      }
      // If name was not found, throw error
      Omega_h_fail("Global %s has not been defined \n", name.c_str());
  }
};

}  // namespace lgr

#endif
