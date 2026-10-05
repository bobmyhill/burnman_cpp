/* GPL v3 or later. Checked conversions to the native APIs' integer indices. */
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace burnman::utils {
template <typename Integer> int checked_int(Integer value) {
  static_assert(std::is_integral_v<Integer>, "An index must be integral.");
  if constexpr (std::is_signed_v<Integer>) {
    const auto wide = static_cast<std::intmax_t>(value);
    if (wide < std::numeric_limits<int>::min() ||
        wide > std::numeric_limits<int>::max())
      throw std::overflow_error("Index exceeds the native integer range.");
  } else if (static_cast<std::uintmax_t>(value) >
             static_cast<std::uintmax_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Index exceeds the native integer range.");
  }
  return static_cast<int>(value);
}
} // namespace burnman::utils
