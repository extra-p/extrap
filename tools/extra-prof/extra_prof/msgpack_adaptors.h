#pragma once
#include "common_types.h"

#include <atomic>
#include <msgpack.hpp>

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
    namespace adaptor {

        template <>
        struct convert<std::atomic<uint64_t>> {
            msgpack::object const& operator()(msgpack::object const& o, std::atomic<uint64_t>& v) const {
                if (o.type != msgpack::type::POSITIVE_INTEGER)
                    throw msgpack::type_error();
                v = o.as<uint64_t>();
                return o;
            }
        };

        template <>
        struct pack<std::atomic<uint64_t>> {
            template <typename Stream>
            packer<Stream>& operator()(msgpack::packer<Stream>& o, std::atomic<uint64_t> const& v) const {
                // packing member variables as an array.
                o.pack_uint64(v);
                return o;
            }
        };

        // template <>
        // struct object_with_zone<std::atomic<uint64_t>> {
        //     void operator()(msgpack::object::with_zone &o, my_class const &v) const {
        //         o.type = type::POSITIVE_INTEGER;
        //         o.via.u64=;
        //         o.via.array.ptr = static_cast<msgpack::object *>(o.zone.allocate_align(
        //             sizeof(msgpack::object) * o.via.array.size, MSGPACK_ZONE_ALIGNOF(msgpack::object)));
        //         o.via.array.ptr[0] = msgpack::object(v.get_name(), o.zone);
        //         o.via.array.ptr[1] = msgpack::object(v.get_age(), o.zone);
        //     }
        // };

    } // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack
