#include "globals.h"

namespace extra_prof {
CallTreeNode *CallTreeNode::findOrAddChild(char const *name, CallTreeNodeType type, CallTreeNodeFlags flags) {
    std::lock_guard lg(mutex);
    for (auto [node_name, node] : _children) {
        if (node_name == name) {
            return node;
        }
    }

    auto [name_, node] =
        _children.emplace_back(name, GLOBALS.calltree_nodes_allocator.construct(name, this, type, flags));
    return node;
}
}