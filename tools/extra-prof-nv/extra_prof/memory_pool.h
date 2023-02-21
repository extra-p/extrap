
#pragma once
#include <memory>
#include <vector>
namespace extra_prof {
template <typename T, size_t alignment_ = 0>
class MemoryReusePool {
    std::vector<T *> available_ressources;
    size_t _size;
    size_t _num_buffers = 0;

public:
    explicit MemoryReusePool(size_t size_) : _size(size_) {}

    [[nodiscard]] T *get_mem() {
        if (available_ressources.empty()) {
            _num_buffers++;
            if (_size == 0) {
                if (alignment_ == 0) {
                    return new T;
                } else {
                    return new (std::align_val_t(alignment_)) T;
                }
            } else {
                if (alignment_ == 0) {
                    return new T[_size];
                } else {
                    return new (std::align_val_t(alignment_)) T[_size];
                }
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
        for (auto &&i : available_ressources) {
            if (_size == 0) {
                delete i;
            } else {
                delete[] i;
            }
        }
        available_ressources.clear();
    }
};
template <typename T, size_t block_size>
class NonReusableBlockPool {

    std::vector<T *> blocks;
    size_t next_element = 0;

public:
    NonReusableBlockPool() {
        blocks.reserve(16);
        blocks.push_back(reinterpret_cast<T *>(::operator new(block_size * sizeof(T))));
    }

    T *get_mem() {
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

    size_t unused_space() { return sizeof(T) * (block_size - next_element); }

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