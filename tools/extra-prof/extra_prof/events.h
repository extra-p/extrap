#pragma once
#include "common_types.h"

#include "calltree_node.h"
#include "concurrent_array_list.h"
#include "containers/string.h"
#include <initializer_list>

namespace extra_prof {
typedef uint64_t time_point;

enum class EventType : uint8_t {
    START = 0,
    END = 1,

    HAPPENS_ON_GPU_FLAG = 2,

    NUM_FLAGS = 2,
    TYPE_MASK = 0xfe,

    NONE = static_cast<uint8_t>(CallTreeNodeType::NONE) << NUM_FLAGS,
    MEMCPY = static_cast<uint8_t>(CallTreeNodeType::MEMCPY) << NUM_FLAGS | HAPPENS_ON_GPU_FLAG,
    MEMSET = static_cast<uint8_t>(CallTreeNodeType::MEMSET) << NUM_FLAGS | HAPPENS_ON_GPU_FLAG,
    KERNEL = static_cast<uint8_t>(CallTreeNodeType::KERNEL) << NUM_FLAGS | HAPPENS_ON_GPU_FLAG,
    SYNCHRONIZE = static_cast<uint8_t>(CallTreeNodeType::SYNCHRONIZE) << NUM_FLAGS,
    OVERHEAD = static_cast<uint8_t>(CallTreeNodeType::OVERHEAD) << NUM_FLAGS,
    MEMMGMT = static_cast<uint8_t>(CallTreeNodeType::MEMMGMT) << NUM_FLAGS
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
    return static_cast<CallTreeNodeType>(static_cast<uint8_t>(lhs) >> static_cast<uint8_t>(EventType::NUM_FLAGS));
}

template <typename S>
inline S& operator<<(S& lhs, EventType rhs) {
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
    CallTreeNode* node = nullptr;
    pthread_t thread;
};
struct Event;
struct EventEnd {
public:
    Event* start = nullptr;
};

struct Event {
public:
    typedef uint16_t StreamIdType;
    time_point timestamp = 0;
    union {
        EventStart start_event;
        EventEnd end_event;
    };
    EventType type = EventType::NONE;
    uint8_t correlationId = 0;
    StreamIdType streamId = 0;
    float resourceUsage = 0;

    Event(){};
    Event(Event&& evt) = default;
    Event(time_point timestamp_, EventType type_, EventStart start, StreamIdType streamId_ = 0,
          uint8_t correlationId_ = 0, float resourceUsage_ = 0)
        : timestamp(timestamp_), type(type_), streamId(streamId_), correlationId(correlationId_),
          resourceUsage(resourceUsage_), start_event(start) {}
    Event(time_point timestamp_, EventType type_, EventEnd end, StreamIdType streamId_ = 0, uint8_t correlationId_ = 0)
        : timestamp(timestamp_), type(type_), streamId(streamId_), correlationId(correlationId_), end_event(end) {}

    inline bool is_start() const { return (type & EventType::END) == EventType::START; }
    inline bool is_end() const { return (type & EventType::END) == EventType::END; }
    inline bool happens_on_gpu() const {
        return (type & EventType::HAPPENS_ON_GPU_FLAG) == EventType::HAPPENS_ON_GPU_FLAG;
    }
    inline EventType get_type() const { return type & EventType::TYPE_MASK; }

    inline bool operator<(const Event& other) const { return timestamp < other.timestamp; }

    Event& operator=(Event&& other) = default;

    // MSGPACK_DEFINE(type, threads);
};

inline void addEventPair(ConcurrentArrayList<Event>& event_stream, EventType type, time_point start, time_point stop,
                         CallTreeNode* node, pthread_t thread, float resourceUsage, uint32_t correlation_id,
                         uint32_t stream_id) {

    auto& ref = event_stream.emplace(start, type | EventType::START, EventStart{node, thread},
                                     static_cast<Event::StreamIdType>(stream_id), static_cast<uint8_t>(correlation_id),
                                     resourceUsage);
    event_stream.emplace(stop, type | EventType::END, EventEnd{&ref}, static_cast<Event::StreamIdType>(stream_id),
                         static_cast<uint8_t>(correlation_id));
}

inline void write_event(std::ofstream& stream, const Event& event, const int pid = 0) {
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
    if (!event.happens_on_gpu()) {
        if (event.is_start()) {
            stream << ", \"tid\": " << static_cast<unsigned int>(event.start_event.thread);
        } else {
            stream << ", \"tid\": " << static_cast<unsigned int>(event.end_event.start->start_event.thread);
        }
    } else {
        stream << ", \"tid\": " << static_cast<unsigned int>(event.streamId);
    }
    stream << ", \"pid\": \"" << pid << "\" },\n";
}

inline void write_event_stream(
    containers::string filename,
    const std::initializer_list<std::pair<const char*, std::reference_wrapper<ConcurrentArrayList<Event>>>>&
        event_streams) {

    std::ofstream stream(filename.c_str());
    stream << '[';

    int pid_ctr = 0;
    for (auto [event_stream_name, event_stream] : event_streams) {
        std::set<unsigned int> gpu_streams;
        stream << R"BLOCK({"name": "process_name", "ph": "M", "pid":)BLOCK" << pid_ctr
               << R"BLOCK(, "args": {"name" : ")BLOCK" << event_stream_name << R"BLOCK("}},)BLOCK"
               << "\n";

        auto& event_stream_list = event_stream.get();
        for (auto event_it = event_stream_list.cbegin(); event_it != event_stream_list.cend(); ++event_it) {
            const Event& event = *event_it;
            write_event(stream, event, pid_ctr);
            if (event.is_start() && event.happens_on_gpu()) {
                gpu_streams.emplace(event.streamId);
            }
        }

        for (auto streamId : gpu_streams) {
            stream << R"BLOCK({"name": "thread_name", "ph": "M", "pid": ")BLOCK" << pid_ctr << R"BLOCK(", "tid":")BLOCK"
                   << streamId << R"BLOCK(", "args": {"name" : "Stream"}},)BLOCK"
                   << "\n";
        }

        pid_ctr++;
    }
    stream << ']';
}
} // namespace extra_prof

// MSGPACK_ADD_ENUM(extra_prof::cupti::EventType);