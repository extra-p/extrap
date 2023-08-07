#pragma once
#include "common_types.h"
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
template <typename K, typename V>
class ConcurrentMap {
private:
    mutable std::shared_mutex mutex;
    std::unordered_map<K, V> map;

public:
    class iterator {
        std::shared_lock<std::shared_mutex> lk;
        typename std::unordered_map<K, V>::iterator iter;

    public:
        iterator(std::shared_mutex& mutex, typename std::unordered_map<K, V>::iterator iter_)
            : lk(mutex), iter(iter_) {}

        iterator& operator++() {
            ++iter;
            return *this;
        }
        bool operator==(iterator& other) const { return iter == other.iter; }
        bool operator!=(iterator& other) const { return iter != other.iter; }
        typename std::unordered_map<K, V>::reference operator*() const { return *iter; }
        typename std::unordered_map<K, V>::value_type* operator->() { return iter.operator->(); }
    };

    class const_iterator {
        std::shared_lock<std::shared_mutex> lk;
        typename std::unordered_map<K, V>::const_iterator iter;

    public:
        const_iterator(std::shared_mutex& mutex, typename std::unordered_map<K, V>::const_iterator iter_)
            : lk(mutex), iter(iter_) {}

        const_iterator& operator++() {
            ++iter;
            return *this;
        }
        bool operator==(const_iterator& other) const { return iter == other.iter; }
        bool operator!=(const_iterator& other) const { return iter != other.iter; }
        const typename std::unordered_map<K, V>::reference operator*() const { return *iter; }
        const typename std::unordered_map<K, V>::value_type* operator->() const { return iter.operator->(); }
    };

    EP_INLINE V* try_get(const K& key) {
        std::shared_lock lk(mutex);
        auto iterator = map.find(key);
        if (iterator != map.end()) {
            return &iterator->second;
        }
        return nullptr;
    }

    EP_INLINE iterator begin() {
        std::shared_lock lk(mutex);
        return iterator(mutex, map.begin());
    }
    EP_INLINE iterator end() {
        std::shared_lock lk(mutex);
        return iterator(mutex, map.end());
    }

    EP_INLINE const_iterator cbegin() const {
        std::shared_lock lk(mutex);
        return const_iterator(mutex, map.cbegin());
    }
    EP_INLINE const_iterator cend() const {
        std::shared_lock lk(mutex);
        return const_iterator(mutex, map.cend());
    }

    EP_INLINE V& operator[](const K& key) {
        {
            std::shared_lock lk(mutex);
            auto iterator = map.find(key);
            if (iterator != map.end()) {
                return iterator->second;
            }
        }
        {
            std::unique_lock lk(mutex);
            return map[key];
        }
    }

    template <typename... _Args>
    EP_INLINE std::pair<std::reference_wrapper<V>, bool> try_emplace(const K& key, _Args&&... __args) {
        {
            std::shared_lock lk(mutex);
            auto iterator = map.find(key);
            if (iterator != map.end()) {
                return std::pair(std::ref(iterator->second), false);
            }
        }
        {
            std::unique_lock lk(mutex);
            auto res = map.try_emplace(key, std::forward<_Args>(__args)...);
            return std::pair(std::ref(res.first->second), res.second);
        }
    }

    template <typename... _Args>
    EP_INLINE bool emplace(_Args&&... __args) {
        std::unique_lock lk(mutex);
        return map.emplace(std::forward<_Args>(__args)...).second;
    }

    EP_INLINE size_t size() const {
        std::shared_lock lk(mutex);
        return map.size();
    }
};
