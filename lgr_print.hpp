#pragma once

#include <iosfwd>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>

namespace lgr {

void print_one(std::ostream&, std::string const&);
void print_one(std::ostream&, char const*);
void print_one(std::ostream&, bool);
void print_one(std::ostream&, char);
void print_one(std::ostream&, signed char);
void print_one(std::ostream&, unsigned char);
void print_one(std::ostream&, signed int);
void print_one(std::ostream&, unsigned int);
void print_one(std::ostream&, signed long);
void print_one(std::ostream&, unsigned long);
void print_one(std::ostream&, signed long long);
void print_one(std::ostream&, unsigned long long);
void print_one(std::ostream&, float);
void print_one(std::ostream&, double);
void print_one(std::ostream&, vector3<double> v);
void print_one(std::ostream&, matrix3x3<double> v);
void print_one(std::ostream&, symmetric3x3<double> v);

template <class ...types>
void print_parameter_pack_expansion_receiver(types...) {}

template <class T>
int print_parameter_pack_expansion_print(std::ostream& stream, T argument) {
  print_one(stream, argument);
  return 0;
}

template <class ...types>
void print(std::ostream& stream, types... arguments) {
  print_parameter_pack_expansion_receiver(print_parameter_pack_expansion_print(stream, arguments)...);
}

}
