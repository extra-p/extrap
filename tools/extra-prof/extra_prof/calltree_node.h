#pragma once
#include "common_types.h"

#include "concurrent_map.h"
#include "concurrent_pair_vector.h"

#include "containers/pair.h"
#include "containers/string.h"
#include "containers/vector.h"

#include "memory_pool.h"
#include "msgpack_adaptors.h"
#include <array>
#include <atomic>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <msgpack.hpp>
#include <mutex>
#include <ostream>
#include <vector>

namespace extra_prof {
enum class CallTreeNodeFlags : uint16_t { NONE = 0, ASYNC = 1 << 0, OVERLAP = 1 << 1, ROOT = 1 << 2 };
inline CallTreeNodeFlags operator|(CallTreeNodeFlags lhs, CallTreeNodeFlags rhs) {
    return static_cast<CallTreeNodeFlags>(static_cast<uint16_t>(lhs) | static_cast<uint16_t>(rhs));
}
inline CallTreeNodeFlags operator&(CallTreeNodeFlags lhs, CallTreeNodeFlags rhs) {
    return static_cast<CallTreeNodeFlags>(static_cast<uint16_t>(lhs) & static_cast<uint16_t>(rhs));
}
inline CallTreeNodeFlags operator~(CallTreeNodeFlags lhs) {
    return static_cast<CallTreeNodeFlags>(~static_cast<uint16_t>(lhs));
}
enum class CallTreeNodeType : uint8_t {
    // Use 7-bits max, because of reuse in enum EventType
    NONE = 0,

    KERNEL_LAUNCH = 1,
    KERNEL = 2,
    MEMCPY = 3,
    MEMSET = 4,
    SYNCHRONIZE = 5,
    OVERHEAD = 6,
    OVERLAP = 7,
    MEMMGMT = 8,
};

class CallTreeNode;
typedef std::vector<std::pair<const RegionID, CallTreeNode*>> CallTreeNodeSublist;
class CallTreeNodeList : public std::array<CallTreeNodeSublist, RegionType::REGIONTYPES_LENGTH> {
public:
    template <typename Packer>
    void msgpack_pack(Packer& msgpack_pk) const;
};

struct Metrics {
    uint64_t duration = 0;
    uint64_t bytes = 0;
    uint64_t visits = 0;

    Metrics& operator+=(const Metrics& other) {
        duration += other.duration;
        bytes += other.bytes;
        visits += other.visits;
        return *this;
    }

    bool is_zero() { return duration == 0 && bytes == 0 && visits == 0; }
};

struct MetricAdapter {
    const std::unordered_map<pthread_t, Metrics>& map;
    std::function<uint64_t(const Metrics&)> func;
    MetricAdapter(const std::unordered_map<pthread_t, Metrics>& map_, std::function<uint64_t(const Metrics&)> func_)
        : map(map_), func(func_) {}

    template <typename Packer>
    void msgpack_pack(Packer& msgpack_pk) const {
        msgpack_pk.pack_array(map.size());
        for (const auto& [thread, metrics] : map) {
            msgpack_pk.pack(func(metrics));
        }
    }
};

class CallTreeNode {
    CallTreeNodeList _children;
    CallTreeNode* _parent = nullptr;
    Metrics metrics;

    void print_internal(std::vector<char const*>& callpath, std::ostream& stream) const {
        callpath.push_back(name());
        for (auto const& fptr : callpath) {
            stream << fptr << " ";
        }
        stream << ": " << metrics.duration << '\n';
        for (auto& children : _children) {
            for (auto [name, child] : children) {
                child->print_internal(callpath, stream);
            }
        }
        callpath.pop_back();
    }

    CallTreeNode(CallTreeNode& node) = delete;
    CallTreeNode(const CallTreeNode& node) = delete;
    CallTreeNode* findOrAddPeer(pthread_t thread, bool include_parent = true);

public:
    RegionID region;
    RegionType region_type = RegionType::UNDEFINED_REGION;
    CallTreeNodeFlags flags = CallTreeNodeFlags::NONE;
    CallTreeNodeType type = CallTreeNodeType::NONE;
    pthread_t owner_thread = pthread_self();
    std::vector<double> gpu_metrics;
#ifdef EXTRA_PROF_ENERGY
    std::atomic<uint64_t> energy_cpu = 0;
    std::atomic<uint64_t> energy_gpu = 0;
#endif
    concurrent_pair_vector<pthread_t, CallTreeNode*> peers;
    CallTreeNode* main_peer = nullptr;
    std::unordered_map<pthread_t, Metrics> per_thread_metrics;

    CallTreeNode(){};
    CallTreeNode(RegionType region_type, RegionID region, CallTreeNode* parent,
                 CallTreeNodeType type_ = CallTreeNodeType::NONE, CallTreeNodeFlags flags_ = CallTreeNodeFlags::NONE)
        : _parent(parent), region_type(region_type), region(region), type(type_), flags(flags_) {}
    ~CallTreeNode(){};

    CallTreeNode* findOrAddChild(RegionType region_type, const RegionID name,
                                 CallTreeNodeType type = CallTreeNodeType::NONE,
                                 CallTreeNodeFlags flags = CallTreeNodeFlags::NONE);

    CallTreeNode* findOrAddPeer(bool include_parent = true);

    inline CallTreeNode* parent() const { return _parent; }
    char const* name() const;

    inline void setAsync(bool is_async) {
        if (is_async) {
            flags = flags | CallTreeNodeFlags::ASYNC;
        } else {
            flags = flags & ~CallTreeNodeFlags::ASYNC;
        }
    }

    inline void setOverlap(bool is_async) {
        if (is_async) {
            flags = flags | CallTreeNodeFlags::OVERLAP;
        } else {
            flags = flags & ~CallTreeNodeFlags::OVERLAP;
        }
    }

    inline bool is_root() { return (flags & CallTreeNodeFlags::ROOT) == CallTreeNodeFlags::ROOT; }

    inline Metrics& my_metrics(bool override_check = false) {
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
        assert(owner_thread == pthread_self() || override_check);
#endif
        return metrics;
    }

    inline void update_metrics(pthread_t thread, std::function<void(Metrics&)> update_function) {
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
        assert(owner_thread == pthread_self());
#endif
        if (thread == this->owner_thread) {
            update_function(metrics);
        } else {
            update_function(per_thread_metrics[thread]);
        }
    }

    bool validateChildren() {
        for (auto& children : _children) {
            for (auto [node_name, node] : children) {
                if (this->metrics.duration < node->metrics.duration) {
                    return false;
                }
            }
        }
        return true;
    }

    void collectData() {
        // assert(GLOBALS.main_thread == pthread_self() && GLOBALS.main_thread == thread);
        collectData(this);
    }

    void print(std::ostream& stream = std::cout) const {
        std::vector<char const*> callpath;
        print_internal(callpath, stream);
    }

    void print(containers::string filename) const {
        std::ofstream stream(filename);
        std::vector<char const*> callpath;
        print_internal(callpath, stream);
    }

    size_t calculate_size() const {
        size_t size = sizeof(*this);
        for (auto& children : _children) {
            for (auto [name, child] : children) {
                size += child->calculate_size() + sizeof(CallTreeNodeList::value_type);
            }
        }
        peers.visit_all([&](auto& item) { size += item.second->calculate_size() + sizeof(item); });
        return size;
    }
    template <typename Packer>
    void msgpack_pack(Packer& msgpack_pk) const {
        char const* name = this->name();
        if (name == nullptr) {
            name = "";
        }

        MetricAdapter duration(per_thread_metrics, [](const auto& metrics) { return metrics.duration; });
        MetricAdapter visits(per_thread_metrics, [](const auto& metrics) { return metrics.visits; });
        MetricAdapter bytes(per_thread_metrics, [](const auto& metrics) { return metrics.bytes; });

#ifdef EXTRA_PROF_ENERGY
        auto energy_tuple = std::make_tuple(std::make_tuple(energy_cpu.load(std::memory_order_acquire)),
                                            std::make_tuple(energy_gpu.load(std::memory_order_acquire)));
#endif

        msgpack::type::make_define_array(name, _children, type, flags, duration, visits, bytes, gpu_metrics
#ifdef EXTRA_PROF_ENERGY
                                         ,
                                         energy_tuple
#endif
                                         )
            .msgpack_pack(msgpack_pk);
    }

private:
    void collectData(CallTreeNode* main_peer) {
        if (!metrics.is_zero() || main_peer->owner_thread == owner_thread) {
            main_peer->per_thread_metrics[owner_thread] += metrics;
        }
        for (auto&& [thread, metrics] : per_thread_metrics) {
            if (thread != owner_thread && !metrics.is_zero()) {
                main_peer->per_thread_metrics[thread] += metrics;
            }
        }

        for (size_t rt = 0; rt < _children.size(); rt++) {
            auto& children = _children[rt];
            if (children.empty()) {
                continue;
            }
            for (auto& [id, child] : children) {
                child->collectData(main_peer->findOrAddChild(RegionType(rt), id, child->type, child->flags));
            }
        }
        peers.visit_all([&](auto& item) { item.second->collectData(main_peer); });
    }
};

template <typename Packer>
void CallTreeNodeList::msgpack_pack(Packer& msgpack_pk) const {
    size_t size = 0;
    for (const auto& sublist : *this) {
        size += sublist.size();
    }

    msgpack_pk.pack_array(size);
    for (const auto& sublist : *this) {
        for (const auto& item : sublist) {
            item.second->msgpack_pack(msgpack_pk);
        }
    }
}

} // namespace extra_prof
MSGPACK_ADD_ENUM(extra_prof::CallTreeNodeFlags);
MSGPACK_ADD_ENUM(extra_prof::CallTreeNodeType);