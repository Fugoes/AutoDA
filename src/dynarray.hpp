#ifndef FSTD_DYNARRAY_HPP
#define FSTD_DYNARRAY_HPP

#include <cstddef>
#include <iterator>
#include <vector>
#include <memory>

namespace fstd {

template<typename T>
class dynarray {
 public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *iterator;
  typedef const value_type *const_iterator;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  inline dynarray();
  explicit inline dynarray(size_type n);
  inline dynarray(size_type n, const value_type &v);
  inline dynarray(std::initializer_list<value_type> vs);
  inline dynarray(dynarray &&that) noexcept;
  // inline dynarray(const dynarray &that);
  inline dynarray(const std::vector<T> &that);
  dynarray(const dynarray &that) = delete;
  ~dynarray();

  inline iterator begin() noexcept { return iterator(__base__); }
  inline const_iterator begin() const noexcept { return const_iterator(__base__); }
  inline const_iterator cbegin() const noexcept { return const_iterator(__base__); }
  inline iterator end() noexcept { return iterator(__base__ + __size__); }
  inline const_iterator end() const noexcept { return const_iterator(__base__ + __size__); }
  inline const_iterator cend() const noexcept { return const_iterator(__base__ + __size__); }

  inline reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  inline const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  inline const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  inline reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  inline const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  inline const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  inline size_type size() const noexcept { return __size__; }
  inline size_type max_size() const noexcept { return __size__; }
  inline bool empty() const noexcept { return __size__ == 0; }

  inline reference operator[](size_type __n) { return __base__[__n]; }
  inline const_reference operator[](size_type __n) const { return __base__[__n]; }
  inline reference front() { return __base__[0]; }
  inline const_reference front() const { return __base__[0]; }
  inline reference back() { return __base__[__size__ - 1]; }
  inline const_reference back() const { return __base__[__size__ - 1]; }
  inline reference at(size_type n);
  inline const_reference at(size_type n) const;

  inline pointer data() noexcept { return __base__; }
  inline const_pointer data() const noexcept { return __base__; }

  inline void swap(dynarray<T> &that) noexcept;

 private:
  size_type __size__;
  value_type *__base__;

  inline void alloc();
  inline void dealloc();
};

template<typename T>
void dynarray<T>::alloc() {
  std::allocator<T> allocator;
  if (__size__ > 0) { __base__ = allocator.allocate(__size__); }
  else { __base__ = nullptr; }
}

template<typename T>
void dynarray<T>::dealloc() {
  std::allocator<T> allocator;
  if (__size__ > 0) { allocator.deallocate(__base__, __size__); }
}

template<typename T>
dynarray<T>::dynarray() : __size__(0), __base__(nullptr) {}

template<typename T>
dynarray<T>::dynarray(dynarray::size_type n) {
  __size__ = n;
  alloc();
  for (size_type i = 0; i < __size__; i++) { ::new(__base__ + i) T; }
}

template<typename T>
dynarray<T>::dynarray(dynarray::size_type n, const value_type &v) {
  __size__ = n;
  alloc();
  for (size_type i = 0; i < __size__; i++) { ::new(__base__ + i) T(v); }
}

template<typename T>
dynarray<T>::dynarray(std::initializer_list<value_type> vs) {
  __size__ = vs.size();
  alloc();
  auto iter = vs.begin();
  for (size_type i = 0; i < __size__; i++, ++iter) { ::new(__base__ + i) T(*iter); }
}

template<typename T>
dynarray<T>::dynarray(dynarray &&that) noexcept {
  __size__ = that.__size__;
  __base__ = that.__base__;
  that.__size__ = 0;
  that.__base__ = nullptr;
}

template<typename T>
dynarray<T>::dynarray(const std::vector<T> &that) {
  __size__ = that.size();
  alloc();
  for (size_type i = 0; i < __size__; i++) { ::new(__base__ + i) T(that[i]); }
}

template<typename T>
dynarray<T>::~dynarray() {
  for (size_type i = 0; i < __size__; i++) { (__base__ + i)->value_type::~value_type(); }
  dealloc();
}

template<typename T>
typename dynarray<T>::reference dynarray<T>::at(dynarray::size_type n) {
  if (n >= __size__) {
    throw std::out_of_range("fstd::dynarray::at");
  }
  return __base__[n];
}

template<typename T>
typename dynarray<T>::const_reference dynarray<T>::at(dynarray::size_type n) const {
  if (n >= __size__) {
    throw std::out_of_range("fstd::dynarray::at");
  }
  return __base__[n];
}

template<typename T>
void dynarray<T>::swap(dynarray<T> &that) noexcept {
  dynarray<T>::size_type that_size = that.__size__;
  dynarray<T>::pointer that_base = that.__base__;
  that.__size__ = __size__;
  that.__base__ = __base__;
  __size__ = that_size;
  __base__ = that_base;
}

}

#endif //FSTD_DYNARRAY_HPP
