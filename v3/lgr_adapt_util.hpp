#pragma once

#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_array_vector.hpp>

namespace lgr {

template <class Index>
HPC_NOINLINE inline void project(
    hpc::counting_range<Index> const old_things,
    hpc::device_vector<Index, Index> const& old_things_to_new_things_in,
    hpc::device_vector<Index, Index>& new_things_to_old_things_in) {
  auto const old_things_to_new_things = old_things_to_new_things_in.cbegin();
  auto const new_things_to_old_things = new_things_to_old_things_in.begin();
  auto functor = [=] HPC_DEVICE (Index const old_thing) {
    Index first = old_things_to_new_things[old_thing];
    Index const last = old_things_to_new_things[old_thing + Index(1)];
    for (; first < last; ++first) {
      new_things_to_old_things[first] = old_thing;
    }
  };
  hpc::for_each(hpc::device_policy(), old_things, functor);
}

template<class Index, class Range>
HPC_NOINLINE void interpolate_data(const hpc::counting_range<Index> &new_entities,
    const hpc::device_vector<Index, Index> &new_to_old,
    const hpc::device_vector<bool, Index> &new_entities_are_same,
    const hpc::device_array_vector<hpc::array<Index, 2, int>, Index> &interpolate_from_entities,
    Range &data)
{
  static_assert(std::is_same<Index, typename Range::size_type>::value,
      "data range incompatible with Index");
  using value_type = typename Range::value_type;
  Range new_data(new_entities.size());
  auto const old_nodes_to_data = data.cbegin();
  auto const new_nodes_to_data = new_data.begin();
  auto const new_nodes_to_old_nodes = new_to_old.cbegin();
  auto const new_nodes_are_same = new_entities_are_same.cbegin();
  auto const interpolate_from = interpolate_from_entities.cbegin();
  auto functor = [=] HPC_DEVICE (Index const i)
  {
    if (new_nodes_are_same[i])
    {
      Index const old_node = new_nodes_to_old_nodes[i];
      new_nodes_to_data[i] = value_type(old_nodes_to_data[old_node]);
    } else
    {
      auto const pair = interpolate_from[i].load();
      Index const left = pair[0];
      Index const right = pair[1];
      new_nodes_to_data[i] = 0.5 * (
          value_type(old_nodes_to_data[left])
          + value_type(old_nodes_to_data[right]));
    }
  };
  hpc::for_each(hpc::device_policy(), new_entities, functor);
  data = std::move(new_data);
}

template<class Index, class Range>
HPC_NOINLINE void distribute_data(const hpc::counting_range<Index> &new_entities,
    const hpc::device_vector<Index, Index> &new_to_old,
    const hpc::device_vector<bool, Index> &new_entities_are_same,
    const hpc::device_array_vector<hpc::array<Index, 2, int>, Index> &interpolate_from_entities,
    Range &data)
{
  static_assert(std::is_same<Index, typename Range::size_type>::value,
      "data range incompatible with Index");
  using value_type = typename Range::value_type;
  Range new_data(new_entities.size());
  auto const old_nodes_to_data = data.cbegin();
  auto const new_nodes_to_data = new_data.begin();
  auto const new_nodes_to_old_nodes = new_to_old.cbegin();
  auto const new_nodes_are_same = new_entities_are_same.cbegin();
  auto const interpolate_from = interpolate_from_entities.cbegin();
  auto functor = [=] HPC_DEVICE (Index const i)
  {
    if (new_nodes_are_same[i])
    {
      Index const old_node = new_nodes_to_old_nodes[i];
      new_nodes_to_data[i] = value_type(old_nodes_to_data[old_node]);
    } else
    {
      auto const pair = interpolate_from[i].load();
      Index const left = pair[0];
      Index const right = pair[1];
      new_nodes_to_data[i] = (1.0 / 3.0) * (
          value_type(old_nodes_to_data[left])
          + value_type(old_nodes_to_data[right]));
      new_nodes_to_data[left] = (2.0 / 3.0) * value_type(old_nodes_to_data[left]);
      new_nodes_to_data[right] = (2.0 / 3.0) * value_type(old_nodes_to_data[right]);
    }
  };
  hpc::for_each(hpc::device_policy(), new_entities, functor);
  data = std::move(new_data);
}

}


