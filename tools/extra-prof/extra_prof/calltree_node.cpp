#include "globals.h"
#include <thread>

namespace extra_prof {

CallTreeNode* CallTreeNode::findOrAddChild(RegionType region_type, const RegionID region, CallTreeNodeType type,
                                           CallTreeNodeFlags flags) {
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
    assert(owner_thread == pthread_self());
#endif

    if (!_children[region_type].empty()) {
        for (auto&& [node_region, node] : _children[region_type]) {
            if (node_region == region) {
                return node;
            }
        }

    } else {
        _children[region_type].reserve(8);
    }

    auto&& [region_, node] = _children[region_type].emplace_back(
        region, GLOBALS.calltree_nodes_allocator.construct(region_type, region, this, type, flags));

    return node;
}

CallTreeNode* CallTreeNode::findOrAddPeer(pthread_t thread, bool include_parent) {
    if (this->owner_thread == thread) {
        return this;
    }

    if (main_peer != nullptr) {
        return main_peer->findOrAddPeer(thread, include_parent);
    }

    CallTreeNode* peer;
    peers.visit_or_add(
        thread, [&](auto& v) { peer = v.second; },
        [&]() {
            CallTreeNode* parent = nullptr;
            if (include_parent && this->parent() != nullptr) {
                parent = this->parent()->findOrAddPeer(thread, false);
            }
            peer = GLOBALS.calltree_nodes_allocator.construct(region_type, region, parent, type, flags);
            peer->main_peer = this;
            return peer;
        });
    return peer;
}

CallTreeNode* CallTreeNode::findOrAddPeer(bool include_parent) { return findOrAddPeer(pthread_self(), include_parent); }

char const* CallTreeNode::name() const {
    if (region_type != RegionType::UNDEFINED_REGION) {
        return GLOBALS.name_register.get_name_ptr(region_type, region);
    }
    return nullptr;
}
} // namespace extra_prof