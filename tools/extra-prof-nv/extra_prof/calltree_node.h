#pragma once
#include "common_types.h"

#include "memory_pool.h"
#include "msgpack_adaptors.h"
#include <array>
#include <atomic>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <msgpack.hpp>
#include <mutex>
#include <ostream>
#include <vector>

namespace extra_prof {
enum class CallTreeNodeFlags : uint16_t { NONE = 0, ASYNC = 1 << 0, OVERLAP = 1 << 1 };
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
    NONE = 0,
    KERNEL_LAUNCH = 1,
    KERNEL = 2,
    MEMCPY = 3,
    MEMSET = 4,
    SYNCHRONIZE = 5,
    OVERHEAD = 6,
    OVERLAP = 7,
    MEMMGMT = 8
};

class CallTreeNode;
class CallTreeNodeList : public std::vector<std::pair<char const *, CallTreeNode *>> {
public:
    template <typename Packer>
    void msgpack_pack(Packer &msgpack_pk) const {
        msgpack_pk.pack_array(this->size());
        for (auto &&[name, node] : *this) {
            node->msgpack_pack(msgpack_pk);
        }
    }
};

class CallTreeNode {
    CallTreeNodeList _children;
    CallTreeNode *_parent = nullptr;
    char const *_name = nullptr;
    mutable std::mutex mutex;
    // thread_local time_point _temp_start_point = 0;

    void print_internal(std::vector<char const *> &callpath, std::ostream &stream) const {
        std::lock_guard lg(mutex);
        callpath.push_back(_name);
        for (auto const &fptr : callpath) {
            stream << fptr << " ";
        }
        stream << ": " << duration << '\n';
        for (auto [name, child] : _children) {
            child->print_internal(callpath, stream);
        }
        callpath.pop_back();
    }

    CallTreeNode(CallTreeNode &node) = delete;
    CallTreeNode(const CallTreeNode &node) = delete;

public:
    std::atomic<uint64_t> duration = 0;
    std::atomic<uint64_t> bytes = 0;
    std::atomic<uint64_t> visits = 0;
    CallTreeNodeFlags flags = CallTreeNodeFlags::NONE;
    CallTreeNodeType type = CallTreeNodeType::NONE;

    CallTreeNode(){};
    CallTreeNode(char const *name, CallTreeNode *parent, CallTreeNodeType type_ = CallTreeNodeType::NONE,
                 CallTreeNodeFlags flags_ = CallTreeNodeFlags::NONE)
        : _parent(parent), _name(name), type(type_), flags(flags_) {}
    ~CallTreeNode(){};

    CallTreeNode *findOrAddChild(char const *name, CallTreeNodeType type = CallTreeNodeType::NONE,
                                 CallTreeNodeFlags flags = CallTreeNodeFlags::NONE);
    inline CallTreeNode *parent() const { return _parent; }
    inline char const *name() const { return _name; }

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

    // time_point temp_start_point() { return _temp_start_point; }
    // void temp_start_point(time_point point) { _temp_start_point = point; }

    void print(std::ostream &stream = std::cout) const {
        std::vector<char const *> callpath;
        print_internal(callpath, stream);
    }

    void print(std::string filename) const {
        std::ofstream stream(filename);
        std::vector<char const *> callpath;
        print_internal(callpath, stream);
    }

    size_t calculate_size() const {
        std::lock_guard lg(mutex);
        size_t size = sizeof(*this);
        for (auto [node_name, node] : _children) {
            size += node->calculate_size() + sizeof(const char *) + sizeof(CallTreeNode *);
        }
        return size;
    }
    template <typename Packer>
    void msgpack_pack(Packer &msgpack_pk) const {
        char const *name = _name;
        if (name == nullptr) {
            name = "";
        }
        msgpack::type::make_define_array(name, _children, type, flags, duration, visits, bytes)
            .msgpack_pack(msgpack_pk);
    }
};
}
MSGPACK_ADD_ENUM(extra_prof::CallTreeNodeFlags);
MSGPACK_ADD_ENUM(extra_prof::CallTreeNodeType);