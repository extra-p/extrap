#pragma once
#include "common_types.h"

#include "calltree_node.h"
#include <msgpack.hpp>

namespace extra_prof {

class Profile {
    std::vector<containers::string> empty_gpu_metric_names;
    const char *EXTRA_PROF = "EXTRA PROF";
    const char *version_name = "1.0";
    CallTreeNode *_call_tree = nullptr;
    std::vector<containers::string> *_gpu_metric_names = &empty_gpu_metric_names;

public:
    Profile(CallTreeNode &call_tree) : _call_tree(&call_tree) {}
    Profile(CallTreeNode &call_tree, std::vector<containers::string> &gpu_metric_names)
        : _call_tree(&call_tree), _gpu_metric_names(&gpu_metric_names) {}
    template <typename Packer>
    void msgpack_pack(Packer &msgpack_pk) const {
        msgpack::type::make_define_array(EXTRA_PROF, version_name, *_call_tree, *_gpu_metric_names)
            .msgpack_pack(msgpack_pk);
    }
};
}