#include "calltree_node.h"
#include <msgpack.hpp>

namespace extra_prof {

class Profile {
    const char *EXTRA_PROF = "EXTRA PROF";
    const char *version_name = "1.0";
    CallTreeNode *_call_tree = nullptr;

public:
    Profile(CallTreeNode &call_tree) : _call_tree(&call_tree) {}
    template <typename Packer>
    void msgpack_pack(Packer &msgpack_pk) const {
        msgpack::type::make_define_array(EXTRA_PROF, version_name, *_call_tree).msgpack_pack(msgpack_pk);
    }
};
}