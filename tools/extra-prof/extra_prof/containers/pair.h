#pragma once
namespace extra_prof::containers {

template <typename _T1, typename _T2>
struct pair {
    // typedef _T1 first_type;  ///< The type of the `first` member
    // typedef _T2 second_type; ///< The type of the `second` member

    _T1 first;  ///< The first member
    _T2 second; ///< The second member

    constexpr pair() : first(), second() {}

    // explicit constexpr pair() : first(), second() {}
    // constexpr pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) {}
    template <typename U1, typename U2>
    constexpr pair(const U1 &__a, const U2 &__b) : first(__a), second(__b) {}

    // explicit constexpr pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) {}

    template <typename U1, typename U2>
    constexpr pair(const pair<U1, U2> &__p) : first(__p.first), second(__p.second) {}

    // explicit constexpr pair(const pair<_U1, _U2> &__p) : first(__p.first), second(__p.second) {}

    constexpr pair(const pair &) = default; ///< Copy constructor
    constexpr pair(pair &&) = default;      ///< Move constructor

    // // DR 811.
    // template <typename _U1>
    // constexpr pair(_U1 &&__x, const _T2 &__y) : first(std::forward<_U1>(__x)), second(__y) {}

    // template <typename _U1>
    // explicit constexpr pair(_U1 &&__x, const _T2 &__y) : first(std::forward<_U1>(__x)), second(__y) {}

    // template <typename _U2>
    // constexpr pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward<_U2>(__y)) {}

    // template <typename _U2>
    // explicit pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward<_U2>(__y)) {}

    // template <typename _U1, typename _U2>
    // constexpr pair(_U1 &&__x, _U2 &&__y) : first(std::forward<_U1>(__x)), second(std::forward<_U2>(__y)) {}

    // template <typename _U1, typename _U2>
    // explicit constexpr pair(_U1 &&__x, _U2 &&__y) : first(std::forward<_U1>(__x)), second(std::forward<_U2>(__y)) {}

    // template <typename _U1, typename _U2>
    // constexpr pair(pair<_U1, _U2> &&__p) : first(std::forward<_U1>(__p.first)), second(std::forward<_U2>(__p.second))
    // {}

    // template <typename _U1, typename _U2>
    // explicit constexpr pair(pair<_U1, _U2> &&__p)
    //     : first(std::forward<_U1>(__p.first)), second(std::forward<_U2>(__p.second)) {}

    // template <typename... _Args1, typename... _Args2>
    // constexpr pair(piecewise_construct_t, tuple<_Args1...>, tuple<_Args2...>);

    // constexpr pair &operator=(const pair &__p) {
    //     first = __p.first;
    //     second = __p.second;
    //     return *this;
    // }

    // constexpr pair &operator=(pair &&__p) {
    //     first = std::forward<first_type>(__p.first);
    //     second = std::forward<second_type>(__p.second);
    //     return *this;
    // }

    // template <typename _U1, typename _U2>
    // constexpr pair &operator=(const pair<_U1, _U2> &__p) {
    //     first = __p.first;
    //     second = __p.second;
    //     return *this;
    // }

    // template <typename _U1, typename _U2>
    // constexpr pair operator=(pair<_U1, _U2> &&__p) {
    //     first = std::forward<_U1>(__p.first);
    //     second = std::forward<_U2>(__p.second);
    //     return *this;
    // }

    // /// Swap the first members and then the second members.
    // constexpr void swap(pair &__p) {
    //     using std::swap;
    //     swap(first, __p.first);
    //     swap(second, __p.second);
    // }
};
}