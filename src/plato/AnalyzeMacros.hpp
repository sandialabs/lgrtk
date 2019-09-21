#pragma once

#include <string>

namespace Plato
{

#define PRINTERR(msg) \
        std::cout << std::string("\n\nFILE: ") + __FILE__ \
        + std::string("\nFUNCTION: ") + __PRETTY_FUNCTION__ \
        + std::string("\nLINE:") + std::to_string(__LINE__) \
        + std::string("\nMESSAGE: ") + msg + "\n\n";

#define THROWERR(msg) \
        throw std::runtime_error(std::string("\n\nFILE: ") + __FILE__ \
        + std::string("\nFUNCTION: ") + __PRETTY_FUNCTION__ \
        + std::string("\nLINE:") + std::to_string(__LINE__) \
        + std::string("\nMESSAGE: ") + msg + "\n\n");

}
//namespace Plato