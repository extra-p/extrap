
#pragma once
#include "common_types.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace extra_prof {
template <typename T, size_t alignment_ = 0>
class MemoryReusePool {
    std::vector<T *> available_ressources;
    size_t _size;
    size_t _num_buffers = 0;

    T *alloc_aligned(size_t size) {
        uint8_t *memory = reinterpret_cast<uint8_t *>(malloc(size * sizeof(T) + alignment_));

        size_t offset = alignment_ - reinterpret_cast<size_t>(memory) % alignment_;
        uint8_t *ret = memory + static_cast<uint8_t>(offset);

        // store the number of extra bytes in the byte before the returned pointer
        memory[offset - 1] = offset;

        return reinterpret_cast<T *>(ret);
    }

    void free_aligned(T *aligned_ptr) {
        uint8_t *raw_ptr = static_cast<uint8_t *>(aligned_ptr);
        int offset = *(raw_ptr - 1);
        free(raw_ptr - offset);
    }

public:
    explicit MemoryReusePool(size_t size_) : _size(size_) {}

    [[nodiscard]] T *get_mem() {
        if (available_ressources.empty()) {
            _num_buffers++;
            if (alignment_ == 0) {
                if (_size == 0) {
                    return new T;
                } else {
                    return new T[_size];
                }

            } else {
                size_t size = _size;
                if (size == 0) {
                    size = 1;
                }
                T *ptr = alloc_aligned(size);
                for (size_t i = 0; i < size; i++) {
                    new (ptr + i) T;
                }
                return ptr;
            }
        } else {
            T *memory = available_ressources.back();
            available_ressources.pop_back();
            return memory;
        }
    }

    void return_mem(T *&mem) {
        available_ressources.push_back(mem);
        mem = nullptr;
    }

    inline size_t size() { return _size; }
    inline size_t num_buffers() { return _num_buffers; }

    void initial_buffer_resize(size_t new_size) {
        if (_num_buffers > 0) {
            throw std::logic_error("EXTRA PROF: Resize not possible once first buffer has been requested.");
        }
        _size = new_size;
    }

    ~MemoryReusePool() {
        for (auto *mem : available_ressources) {
            if (alignment_ == 0) {
                if (_size == 0) {
                    delete mem;
                } else {
                    delete[] mem;
                }
            } else {
                auto size = _size;
                if (size == 0) {
                    size = 1;
                }
                for (size_t j = 0; j < size; j++) {
                    mem[j].~T();
                }
                free_aligned(mem);
            }
        }
        available_ressources.clear();
    }
};
template <typename T, size_t block_size>
class NonReusableBlockPool {

    std::vector<T *> blocks;
    size_t next_element = 0;
    std::mutex mutex;

public:
    NonReusableBlockPool() {
        blocks.reserve(16);
        blocks.push_back(reinterpret_cast<T *>(::operator new(block_size * sizeof(T))));
    }

    T *get_mem() {
        std::lock_guard lg(mutex);
        if (block_size <= next_element) {
            blocks.push_back(reinterpret_cast<T *>(::operator new(block_size * sizeof(T))));
            next_element = 0;
        }
        T *ptr = blocks.back() + next_element;
        next_element++;
        return ptr;
    }
    template <typename... _Args>
    T *construct(_Args &&...__args) {
        return new (get_mem()) T(std::forward<_Args>(__args)...);
    }

    size_t unused_space() {
        std::lock_guard lg(mutex);
        return sizeof(T) * (block_size - next_element);
    }

    ~NonReusableBlockPool() {
        for (auto block = blocks.begin(); block != blocks.end() - 1; ++block) {
            for (size_t i = 0; i < block_size; i++) {
                (*block)[i].~T();
            }
            ::operator delete(*block);
        }
        auto last_block = blocks.end() - 1;
        for (size_t i = 0; i < next_element; i++) {
            (*last_block)[i].~T();
        }
        ::operator delete(*last_block);
    }
};
}