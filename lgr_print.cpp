#include <iostream>
#include <iomanip>

#include <lgr_print.hpp>

namespace lgr {

void print_one(std::ostream& stream, std::string const& string) {
  stream << string;
}

void print_one(std::ostream& stream, char const* c_string) {
  stream << c_string;
}

void print_one(std::ostream& stream, bool b) {
  stream << (b ? "true" : "false");
}

void print_one(std::ostream& stream, char c) {
  stream << int(c);
}

void print_one(std::ostream& stream, signed char c) {
  stream << int(c);
}

void print_one(std::ostream& stream, unsigned char c) {
  stream << int(c);
}

void print_one(std::ostream& stream, signed int i) {
  stream << std::setfill('0') << std::setw(10) << i;
}

void print_one(std::ostream& stream, unsigned int i) {
  stream << std::setfill('0') << std::setw(10) << i;
}

void print_one(std::ostream& stream, signed long i) {
  stream << i;
}

void print_one(std::ostream& stream, unsigned long i) {
  stream << i;
}

void print_one(std::ostream& stream, signed long long i) {
  stream << i;
}

void print_one(std::ostream& stream, unsigned long long i) {
  stream << i;
}

void print_one(std::ostream& stream, float f) {
  stream << std::scientific << std::setprecision(8) << f;
}

void print_one(std::ostream& stream, double f) {
  stream << std::scientific << std::setprecision(17) << f;
}

void print_one(std::ostream& stream, vector3<double> v) {
  print(stream, v(0), " ", v(1), " ", v(2));
}

void print_one(std::ostream& stream, matrix3x3<double> m) {
  print(stream, m(0, 0), " ", m(0, 1), " ", m(0, 2), "\n");
  print(stream, m(1, 0), " ", m(1, 1), " ", m(1, 2), "\n");
  print(stream, m(2, 0), " ", m(2, 1), " ", m(2, 2), "\n");
}

void print_one(std::ostream& stream, symmetric3x3<double> m) {
  print(stream, m(0, 0), " ", m(0, 1), " ", m(0, 2), "\n");
  print(stream, m(1, 0), " ", m(1, 1), " ", m(1, 2), "\n");
  print(stream, m(2, 0), " ", m(2, 1), " ", m(2, 2), "\n");
}

}
