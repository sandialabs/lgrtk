#pragma once


namespace Plato
{

#define PRINTERR(msg) \
        std::cout<< "\nFILE: " << __FILE__ \
        << "\nFUNCTION: " << __PRETTY_FUNCTION__ \
        << "\nLINE:" << __LINE__ \
        << "\nMESSAGE: " << msg;

#define THROWERR(msg) \
        std::cout << "\nFILE: " << __FILE__ \
        << "\nFUNCTION: " << __PRETTY_FUNCTION__ \
        << "\nLINE:" << __LINE__ \
        << "\nMESSAGE: " << msg; \
        throw std::runtime_error("Look above for error message.");

}
//namespace Plato