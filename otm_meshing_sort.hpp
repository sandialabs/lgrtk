#pragma once

#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <cassert>

namespace lgr {

template<typename NodeRelRangeType, typename NodeRelToRelRangeType,
    typename NodeRelToNodesOfRelRangeType>
void otm_sort_node_relations(lgr::state &s, NodeRelRangeType &nodes_to_node_relations,
    NodeRelToRelRangeType &node_relations_to_relation_indices,
    NodeRelToNodesOfRelRangeType &node_relations_to_nodes_of_relation)
{
  auto node_rels = nodes_to_node_relations.cbegin();
  auto node_rel_to_rel = node_relations_to_relation_indices.begin();
  auto node_rel_to_nodes_of_node_rel = node_relations_to_nodes_of_relation.begin();
  auto sort_functor =
      [=] HPC_DEVICE (
          node_index const node)
          {
            auto const this_node_rel = node_rels[node];
            typename NodeRelRangeType::value_type const except_last(this_node_rel.begin(), this_node_rel.end() - 1);
            for (auto const nr : except_last)
            {
              typename NodeRelRangeType::value_type const remaining(nr + 1, this_node_rel.end());
              auto min_rel(node_rel_to_rel[nr]);
              auto min_node_rel = nr;
              for (auto const nr2 : remaining)
              {
                auto const relation = node_rel_to_rel[nr2];
                if (relation < min_rel)
                {
                  min_rel = relation;
                  min_node_rel = nr2;
                }
              }
              hpc::swap(node_rel_to_rel[nr], node_rel_to_rel[min_node_rel]);
              hpc::swap(node_rel_to_nodes_of_node_rel[nr], node_rel_to_nodes_of_node_rel[min_node_rel]);
            }
#ifndef NDEBUG
          for (auto i(*(this_node_rel.begin())); i < (*(this_node_rel.end())) - 1; ++i)
          {
            assert(node_rel_to_rel[i] < node_rel_to_rel[i + 1]);
          }
#endif
        };
  hpc::for_each(hpc::device_policy(), s.nodes, sort_functor);
}

}
