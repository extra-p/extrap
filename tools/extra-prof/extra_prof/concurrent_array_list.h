#pragma once
#include "common_types.h"
#include <atomic>
#include <mutex>
#include <vector>
template <typename T>
class ConcurrentArrayList {

private:
    struct Chunk {
        int chunk_size = 128;

        std::mutex data_mutex;
        T* data = new T[chunk_size];
        size_t fill_size = 0;

        std::mutex next_mutex;
        Chunk* _next_node = nullptr;

        ~Chunk() {
            delete[] data;
            if (_next_node != nullptr) {
                delete _next_node;
            }
        }

        Chunk* get_next_chunk() {
            std::lock_guard lk(next_mutex);
            return _next_node;
        }

        Chunk* get_or_create_next_chunk() {
            Chunk* next_chunk = get_next_chunk();
            if (next_chunk != nullptr) {
                return next_chunk;
            } else {
                Chunk* new_node = new Chunk();
                std::lock_guard lk(next_mutex);
                if (_next_node == nullptr) {
                    _next_node = new_node;
                } else {
                    delete new_node;
                }
                return _next_node;
            }
        }

        std::pair<T*, bool> find_next_empty_spot() {
            if (data_mutex.try_lock()) {
                std::lock_guard lk(data_mutex, std::adopt_lock); //

                if (fill_size < chunk_size) {
                    auto* ptr = data + fill_size;
                    fill_size++;
                    return {ptr, fill_size < chunk_size};
                } else {
                    return {nullptr, false};
                }
            } else {
                return {nullptr, true};
            }
        }
    };

    Chunk* first_chunk;
    std::atomic<Chunk*> first_free_chunk;

public:
    ConcurrentArrayList() {
        first_chunk = new Chunk();
        first_free_chunk = first_chunk;
    };

    template <class... Args>
    T& emplace(Args&&... args) {
        Chunk* free_chunk = first_free_chunk;
        auto spot = free_chunk->find_next_empty_spot();
        int ctr = 0;
        if (spot.first != nullptr && !spot.second) {
            Chunk* expected = free_chunk;
            free_chunk = free_chunk->get_or_create_next_chunk();
            first_free_chunk.compare_exchange_strong(expected, free_chunk);
        }
        while (spot.first == nullptr) {

            Chunk* expected = free_chunk;
            free_chunk = free_chunk->get_or_create_next_chunk();
            if (!spot.second) {
                first_free_chunk.compare_exchange_strong(expected, free_chunk);
            }

            spot = free_chunk->find_next_empty_spot();
            // if (ctr >= 3) {
            //     std::cout << "Search for spot 3<" << ctr++ << std::endl;
            // }
        }
        // std::lock_guard lk(free_chunk->data_mutex);
        return *new (spot.first) T(std::forward<Args>(args)...);
    }

    size_t estimate_size() {
        size_t estimated_size = 0;
        for (Chunk* current_chunk = first_chunk; current_chunk != nullptr;
             current_chunk = current_chunk->get_next_chunk()) {
            std::lock_guard lk(current_chunk->data_mutex);
            estimated_size += current_chunk->fill_size;
        }
        return estimated_size;
    }

    class const_iterator {
        Chunk* chunk;
        T* ptr;
        std::unique_lock<std::mutex> chunk_lock;

    public:
        const_iterator() : chunk(nullptr), ptr(nullptr) {}
        const_iterator(Chunk* chunk_) : chunk(chunk_), chunk_lock(chunk->data_mutex) {
            if (chunk->fill_size == 0) {
                ptr = nullptr;
            } else {
                ptr = chunk->data;
            }
        }

        void advance_to_next_chunk() {
            Chunk* next_chunk = chunk->get_next_chunk();
            if (next_chunk == nullptr) {
                chunk = nullptr;
                ptr = nullptr;
                chunk_lock = std::unique_lock<std::mutex>();
            } else {
                chunk_lock = std::unique_lock(next_chunk->data_mutex);
                chunk = next_chunk;
                if (chunk->fill_size == 0) {
                    ptr = nullptr;
                } else {
                    ptr = chunk->data;
                }
            }
        }

        const_iterator& operator++() {
            if (chunk == nullptr) {
                return *this;
            }
            while (ptr == nullptr) {
                advance_to_next_chunk();
                if (chunk == nullptr) {
                    return *this;
                }
            }
            if (ptr + 1 < chunk->data + chunk->fill_size) {
                ptr++;
            } else {
                do {
                    advance_to_next_chunk();
                    if (chunk == nullptr) {
                        return *this;
                    }
                } while (ptr == nullptr);
            }
            return *this;
        }
        bool operator==(const const_iterator& other) const { return chunk == other.chunk && ptr == other.ptr; }
        bool operator!=(const const_iterator& other) const { return !(*this == other); }
        const T& operator*() const { return *ptr; }
    };

    const_iterator cbegin() { return const_iterator(first_chunk); }
    const_iterator cend() { return const_iterator(); }
    const ConcurrentArrayList& crange() const noexcept { return *this; }

    ~ConcurrentArrayList() {
        first_free_chunk = nullptr;
        delete first_chunk;
    };
};
