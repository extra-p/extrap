#pragma once
namespace extra_prof::containers {

template <typename _T1, typename _T2>
struct pair {
    _T1 first;
    _T2 second;

    constexpr pair() : first(), second() {}

    template <typename U1, typename U2>
    constexpr pair(const U1& __a, const U2& __b) : first(__a), second(__b) {}

    template <typename U1, typename U2>
    constexpr pair(const pair<U1, U2>& __p) : first(__p.first), second(__p.second) {}

    constexpr pair(const pair&) = default;
    constexpr pair(pair&&) = default;
};
} // namespace extra_prof::containers