#pragma once

template <typename K, typename V>
class concurrent_pair_vector {
private:
    std::vector<std::pair<K, V>> data;
    mutable std::mutex mutex;

public:
    concurrent_pair_vector(){};
    ~concurrent_pair_vector(){};

    bool visit(const K& key, std::function<void(std::pair<K, V>&)> func) {
        std::lock_guard lk(mutex);
        if (!data.empty()) {
            for (auto& item : data) {
                if (key == item.first) {
                    func(item);
                    return true;
                }
            }
        }
        return false;
    }

    bool visit(const K& key, std::function<void(const std::pair<K, V>&)> func) const {
        std::lock_guard lk(mutex);
        if (!data.empty()) {
            for (const auto& item : data) {
                if (key == item.first) {
                    func(item);
                    return true;
                }
            }
        }
        return false;
    }

    void visit_or_add(const K& key, std::function<void(std::pair<K, V>&)> func, std::function<V()> add_func) {
        std::lock_guard lk(mutex);
        if (!data.empty()) {
            for (auto& item : data) {
                if (key == item.first) {
                    func(item);
                    return;
                }
            }
        }
        data.emplace_back(key, add_func());
    }

    void visit_or_add(const K& key, std::function<void(std::pair<K, V>&)> func, std::function<V()> add_func,
                      std::function<size_t()> reservation_size) {
        std::lock_guard lk(mutex);
        if (data.empty()) {
            data.reserve(reservation_size());
        } else {
            for (auto& item : data) {
                if (key == item.first) {
                    func(item);
                    return;
                }
            }
        }
        data.emplace_back(key, add_func());
    }

    void visit_all(std::function<void(std::pair<K, V>&)> func) {
        std::lock_guard lk(mutex);
        for (auto& item : data) {
            func(item);
        }
    }
    void visit_all(std::function<void(const std::pair<K, V>&)> func) const {
        std::lock_guard lk(mutex);
        for (const auto& item : data) {
            func(item);
        }
    }

    bool empty() {
        std::lock_guard lk(mutex);
        return data.empty();
    }

    void reserve(size_t size) {
        std::lock_guard lk(mutex);
        return data.reserve(size);
    }
};
