#pragma once
#include "calltree_node.h"
#include <filesystem>
#include <initializer_list>

namespace extra_prof {
typedef uint64_t time_point;

enum class EventType : uint8_t {
    START = 0,
    END = 1,
    TYPE_MASK = 0xfe,
    NONE = static_cast<uint8_t>(CallTreeNodeType::NONE) << 1,
    MEMCPY = static_cast<uint8_t>(CallTreeNodeType::MEMCPY) << 1,
    MEMSET = static_cast<uint8_t>(CallTreeNodeType::MEMSET) << 1,
    KERNEL = static_cast<uint8_t>(CallTreeNodeType::KERNEL) << 1,
    SYNCHRONIZE = static_cast<uint8_t>(CallTreeNodeType::SYNCHRONIZE) << 1,
    OVERHEAD = static_cast<uint8_t>(CallTreeNodeType::OVERHEAD) << 1,
    MEMMGMT = static_cast<uint8_t>(CallTreeNodeType::MEMMGMT) << 1
};

inline EventType operator|(EventType lhs, EventType rhs) {
    return static_cast<EventType>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}
inline EventType operator&(EventType lhs, EventType rhs) {
    return static_cast<EventType>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}
template <typename T, typename S>
inline T enum_cast(S lhs);
template <>
inline CallTreeNodeType enum_cast(EventType lhs) {
    return static_cast<CallTreeNodeType>(static_cast<uint8_t>(lhs) >> 1);
}

template <typename S>
inline S &operator<<(S &lhs, EventType rhs) {
    switch (rhs & EventType::TYPE_MASK) {
    case EventType::NONE:
        lhs << "NONE";
        break;
    case EventType::MEMCPY:
        lhs << "MEMCPY";
        break;
    case EventType::MEMSET:
        lhs << "MEMSET";
        break;
    case EventType::KERNEL:
        lhs << "KERNEL";
        break;
    case EventType::SYNCHRONIZE:
        lhs << "SYNCHRONIZE";
        break;
    case EventType::OVERHEAD:
        lhs << "OVERHEAD";
        break;
    case EventType::MEMMGMT:
        lhs << "MEMMGMT";
        break;
    default:
        lhs << "UNKNOWN";
        break;
    }
    return lhs;
}

struct EventStart {
public:
    CallTreeNode *node = nullptr;
};
struct Event;
struct EventEnd {
public:
    Event *start = nullptr;
};

struct Event {
public:
    time_point timestamp;
    union {
        EventStart start_event;
        EventEnd end_event;
    };
    EventType type = EventType::NONE;
    uint8_t correlationId = 0;
    uint16_t streamId = 0;
    float resourceUsage = 0;

    Event(){};
    Event(Event &&evt) = default;
    Event(time_point timestamp_, EventType type_, EventStart start, uint16_t streamId_ = 0, uint8_t correlationId_ = 0,
          float resourceUsage_ = 0)
        : timestamp(timestamp_), type(type_), streamId(streamId_), correlationId(correlationId_),
          resourceUsage(resourceUsage_), start_event(start) {}
    Event(time_point timestamp_, EventType type_, EventEnd end, uint16_t streamId_ = 0, uint8_t correlationId_ = 0)
        : timestamp(timestamp_), type(type_), streamId(streamId_), correlationId(correlationId_), end_event(end) {}

    inline bool is_start() const { return (type & EventType::END) == EventType::START; }
    inline bool is_end() const { return (type & EventType::END) == EventType::END; }
    inline EventType get_type() const { return type & EventType::TYPE_MASK; }

    inline bool operator<(const Event &other) const { return timestamp < other.timestamp; }

    Event &operator=(Event &&other) = default;

    // MSGPACK_DEFINE(type, threads);
};

inline void addEventPair(std::deque<Event> &event_stream, EventType type, time_point start, time_point stop,
                         CallTreeNode *node, float resourceUsage, uint32_t correlation_id, uint32_t stream_id) {

    auto &ref =
        event_stream.emplace_back(start, type | EventType::START, EventStart{node}, static_cast<uint16_t>(stream_id),
                                  static_cast<uint8_t>(correlation_id), resourceUsage);
    event_stream.emplace_back(stop, type | EventType::END, EventEnd{&ref}, static_cast<uint16_t>(stream_id),
                              static_cast<uint8_t>(correlation_id));
}

void write_event(std::ofstream &stream, const Event &event, const int pid = 0) {
    stream << "{\"name\": \"";

    if (event.is_start()) {
        stream << event.start_event.node->name();
    } else {
        stream << event.end_event.start->start_event.node->name();
    }

    stream << "\", \"cat\": \"" << event.get_type() << "\", \"ts\": " << event.timestamp / 1000 << ",";
    if (event.is_start()) {
        stream << "\"ph\": \"B\"";
    } else {
        stream << "\"ph\": \"E\"";
    }
    // stream << ", \"id\":" << event.correlation_id;
    // stream << ", \"args\": {\"correlation_id\": " << event.correlation_id << '}';
    stream << ", \"tid\": " << static_cast<int>(event.streamId);
    stream << ", \"pid\": \"" << pid << "\" },\n";
}

void write_event_stream(std::filesystem::path filename,
                        const std::initializer_list<std::reference_wrapper<std::deque<Event>>> &event_streams) {
    std::ofstream stream(filename);
    stream << '[';
    int pid_ctr = 0;
    for (auto event_stream : event_streams) {
        for (auto &event : event_stream.get()) {
            write_event(stream, event, pid_ctr);
        }
        pid_ctr++;
    }
    stream << ']';
}
}

// MSGPACK_ADD_ENUM(extra_prof::cupti::EventType);