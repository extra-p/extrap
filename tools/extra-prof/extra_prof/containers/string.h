#pragma once
#include "../common_types.h"

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace extra_prof::containers {

class string {
    friend string operator+(const char *&lhs, const string &rhs);

private:
    char *_string = nullptr;
    size_t _size = 0;

    char *allocate(const size_t len) {
        char *ptr = reinterpret_cast<char *>(malloc((len + 1) * sizeof(char)));
        ptr[len] = 0;
        return ptr;
    }
    char *allocate_and_copy(const size_t len, const char *src) {
        char *ptr = allocate(len);
        if (src != nullptr) {
            strncpy(ptr, src, len);
        }
        return ptr;
    }
    char *allocate_and_append(const size_t len, const char *src) {
        char *ptr = allocate(_size + len);
        if (_string != nullptr) {
            strcpy(ptr, _string);
        }
        if (src != nullptr) {
            strcpy(ptr + _size, src);
        }
        return ptr;
    }
    void destruct(char *ptr) {
        if (ptr == nullptr) {
            return;
        }
        free(ptr);
    }

public:
    string(const string &str) {
        _size = str._size;
        _string = allocate_and_copy(_size, str._string);
    };

    string(string &&str) {
        _size = str._size;
        _string = str._string;
        str._string = nullptr;
        str._size = 0;
    };

    string &operator=(const string &str) {
        destruct(_string);
        _size = str._size;
        _string = allocate_and_copy(_size, str._string);
        return *this;
    }

    string &operator=(string &&str) {
        destruct(_string);
        _size = str._size;
        _string = str._string;
        str._string = nullptr;
        str._size = 0;
        return *this;
    }

    string(){};

    string(const char *str) {
        if (str != nullptr) {
            _size = strlen(str);
            _string = allocate_and_copy(_size, str);
        }
    };
    explicit string(size_t len) {
        _size = len;
        _string = allocate(_size);
    };
    string(const char *str, const size_t len) {
        _size = len;
        _string = allocate_and_copy(_size, str);
    };
    ~string() {
        if (_string == nullptr) {
            return;
        }
        free(_string);
    };

    static constexpr size_t npos = -1;

    const char *c_str() const { return _string; }
    char *data() { return _string; }
    size_t &internal_size() { return _size; }

    operator const char *() const { return _string; }

    size_t size() const { return _size; }

    size_t find(const char *needle, size_t pos = 0) const {
        if (needle == nullptr) {
            return 0;
        }
        if (pos >= _size) {
            return npos;
        }
        char *result = strstr(_string + pos, needle);
        if (result == nullptr) {
            return npos;
        }
        return result - _string;
    }

    size_t find(char needle, size_t pos = 0) const {
        if (pos >= _size) {
            return npos;
        }
        char *result = strchr(_string + pos, needle);
        if (result == nullptr) {
            return npos;
        }
        return result - _string;
    }

    string substr(size_t pos = 0, size_t count = npos) const {
        if (pos > _size) {
            throw std::out_of_range("Substr position outside string.");
        }
        if (count > _size - pos) {
            count = _size - pos;
        }
        return string(_string + pos, count);
    }

    string &operator+=(const string &str) {
        char *new_string = allocate_and_append(str._size, str._string);
        destruct(_string);
        _string = new_string;
        _size += str._size;
        return *this;
    }

    string &operator+=(const char *str) {
        if (str == nullptr) {
            return *this;
        }
        size_t str_size = strlen(str);
        char *new_string = allocate_and_append(str_size, str);
        destruct(_string);
        _string = new_string;
        _size += str_size;
        return *this;
    }

    string &operator+=(const char chr) {
        char *new_string = allocate_and_copy(_size + 1, _string);
        new_string[_size] = chr;
        new_string[_size + 1] = 0;
        destruct(_string);
        _string = new_string;
        _size += 1;
        return *this;
    }

    string operator+(const string &rhs) const {
        string new_string(_size + rhs._size);
        if (_string != nullptr) {
            strcpy(new_string._string, _string);
        }
        if (rhs._string != nullptr) {
            strcpy(new_string._string + _size, rhs._string);
        }
        return new_string;
    }
    string operator+(const char *&rhs) const {
        if (rhs == nullptr) {
            return string(*this);
        }
        string new_string(_size + strlen(rhs));
        if (_string != nullptr) {
            strcpy(new_string._string, _string);
        }
        strcpy(new_string._string + _size, rhs);
        return new_string;
    }

    bool operator==(const string &rhs) const {
        if (_size != rhs._size) {
            return false;
        } else if (_string == rhs._string) {
            return true;
        } else if (_string == nullptr || rhs._string == nullptr) {
            return false;
        }
        return strcmp(_string, rhs._string) == 0;
    }

    static string format(const char *format, ...) {
        va_list arglist;
        va_start(arglist, format);
        int res = vsnprintf(nullptr, 0, format, arglist);
        if (res < 0) {
            throw std::runtime_error("Format error in string.");
        }
        string new_string(static_cast<size_t>(res));
        res = vsnprintf(new_string._string, new_string._size + 1, format, arglist);
        if (res < 0) {
            throw std::runtime_error("Format error in string.");
        }
        va_end(arglist);
        return new_string;
    }

    template <typename Packer>
    void msgpack_pack(Packer &msgpack_pk) const {
        msgpack_pk.pack_str(_size).pack_str_body(_string, _size);
    }
};
inline string operator+(const char *&lhs, const string &rhs) {
    size_t lhs_len = strlen(lhs);
    string new_string(lhs_len + rhs._size);
    strcpy(new_string._string, lhs);
    strcpy(new_string._string + lhs_len, rhs._string);
    return new_string;
}

}

template <>
struct std::hash<extra_prof::containers::string> {
    std::size_t operator()(extra_prof::containers::string const &s) const noexcept {
        size_t h = 5381;
        int c;
        const char *cstr = s.c_str();
        while ((c = *cstr++))
            h = ((h << 5) + h) + c;
        return h;
    }
};