#pragma once
namespace extra_prof::containers {
template <typename T>
class vector {
private:
    T* _data = nullptr;
    size_t _capacity = 0;
    size_t _size = 0;

public:
    vector(){};
    ~vector() {
        if (_data == nullptr) {
            return;
        }
        for (size_t i = 0; i < _size; i++) {
            _data[i].~T();
        }
        free(_data);
    };

    size_t size() { return _size; }
    size_t capacity() { return _capacity; }

    void reserve(size_t new_size) {
        if (new_size <= size()) {
            return;
        }
        T* temp = malloc(new_size * sizeof(T));
        if (_data != nullptr) {
            memcpy(temp, _data, _size * sizeof(T));
            free(_data);
        }
        _data = temp;
        _capacity = new_size;
        _size = new_size;
    }
    void resize(size_t new_size) {
        if (new_size <= size()) {
            return;
        }
        T* temp = malloc(new_size * sizeof(T));
        if (_data != nullptr) {
            memcpy(temp, _data, _size * sizeof(T));
            free(_data);
        }
        _data = temp;
        for (size_t i = _size; i < new_size; i++) {
            new (_data + i) T;
        }
        _capacity = new_size;
        _size = new_size;
    }
};
} // namespace extra_prof::containers