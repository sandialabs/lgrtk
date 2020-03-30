#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_print.hpp>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

namespace lgr {

inline std::ofstream make_vtk_output_stream(const std::string& prefix,
    int const file_output_index)
{
  std::stringstream filename_stream;
  filename_stream << prefix << "_" << file_output_index << ".vtk";
  std::ofstream stream(filename_stream.str().c_str());
  stream << std::scientific << std::setprecision(17);
  return stream;
}

inline void start_vtk_unstructured_grid_file(std::ostream &stream)
{
  stream << "# vtk DataFile Version 3.0\n";
  stream << "vtk output\n";
  stream << "ASCII\n";
  stream << "DATASET UNSTRUCTURED_GRID\n";
}

template <typename PointIndexType>
inline void write_vtk_point_data(std::ostream& stream, hpc::counting_range<PointIndexType> const& point_range) {
  stream << "POINT_DATA " << point_range.size() << "\n";
}

template<class Quantity, class Index>
inline void write_vtk_points(std::ostream &stream,
    hpc::pinned_array_vector<hpc::vector3<Quantity>, Index> const &x)
{
  stream << "POINTS " << x.size() << " double\n";
  for (auto ref : x)
  {
    stream << hpc::vector3<double>(ref.load()) << "\n";
  }
}

template<class Quantity, class Index>
inline void write_vtk_vectors(std::ostream &stream, char const *name,
    hpc::pinned_array_vector<hpc::vector3<Quantity>, Index> const &vec)
{
  stream << "VECTORS " << name << " double\n";
  for (auto const ref : vec)
  {
    stream << hpc::vector3<double>(ref.load()) << "\n";
  }
}

template<class Quantity, class Index>
inline void write_vtk_scalars(std::ostream &stream, std::string const &name,
    hpc::pinned_vector<Quantity, Index> const &vec)
{
  stream << "SCALARS " << name << " double 1\n";
  stream << "LOOKUP_TABLE default\n";
  for (Quantity const val : vec)
  {
    stream << double(val) << "\n";
  }
}

}
