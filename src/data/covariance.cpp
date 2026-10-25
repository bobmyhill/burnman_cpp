// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#include "HGP_2018_ds633_covariance_data.hpp"
#include "HP_2011_ds62_covariance_data.hpp"
#include "HPx_ds636_covariance_data.hpp"
#include "burnman/minerals/datasets.hpp"
#include <hdf5.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace burnman::minerals {
namespace {
void require(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(message);
}

struct Hdf5Handle {
  hid_t id;
  herr_t (*close)(hid_t);
  Hdf5Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) {
    require(id >= 0, "Cannot open embedded HDF5 covariance data.");
  }
  ~Hdf5Handle() { close(id); }
  Hdf5Handle(const Hdf5Handle &) = delete;
  Hdf5Handle &operator=(const Hdf5Handle &) = delete;
};

CovarianceData read_covariance(const char *filename, const unsigned char *image,
                               std::size_t size) {
  // Serialise reads even when the installed HDF5 library was built
  // without thread safety. The returned data is independent of HDF5 handles.
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  unsigned int filter_flags = 0;
  require(H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 &&
              H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_flags) >= 0 &&
              (filter_flags & H5Z_FILTER_CONFIG_DECODE_ENABLED),
          "HDF5 must support gzip/deflate decoding for covariance data.");
  const Hdf5Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  require(H5Pset_fapl_core(access.id, 65536, false) >= 0 &&
              H5Pset_file_image(access.id, const_cast<unsigned char *>(image),
                                size) >= 0,
          "Cannot initialise the HDF5 covariance file image.");
  const Hdf5Handle file(H5Fopen(filename, H5F_ACC_RDONLY, access.id), H5Fclose);
  const Hdf5Handle names(H5Dopen2(file.id, "endmember_names", H5P_DEFAULT),
                         H5Dclose);
  const Hdf5Handle names_space(H5Dget_space(names.id), H5Sclose);
  const Hdf5Handle names_type(H5Dget_type(names.id), H5Tclose);
  hsize_t n = 0;
  require(
      H5Sget_simple_extent_ndims(names_space.id) == 1 &&
          H5Sget_simple_extent_dims(names_space.id, &n, nullptr) >= 0 &&
          n > 0 &&
          n <= static_cast<hsize_t>(std::numeric_limits<Eigen::Index>::max()) &&
          H5Tget_class(names_type.id) == H5T_STRING &&
          H5Tis_variable_str(names_type.id) == 0,
      "Invalid endmember names in HDF5 covariance data.");
  const std::size_t width = H5Tget_size(names_type.id);
  require(width > 0, "Invalid covariance endmember name width.");
  std::vector<char> name_data(static_cast<std::size_t>(n) * width);
  require(H5Dread(names.id, names_type.id, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  name_data.data()) >= 0,
          "Cannot read covariance endmember names.");
  CovarianceData result;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    const std::string name(name_data.data() + i * width, width);
    result.endmember_names.push_back(name.substr(0, name.find('\0')));
  }
  const Hdf5Handle matrix(H5Dopen2(file.id, "covariance_matrix", H5P_DEFAULT),
                          H5Dclose);
  const Hdf5Handle matrix_space(H5Dget_space(matrix.id), H5Sclose);
  hsize_t shape[2]{};
  require(H5Sget_simple_extent_ndims(matrix_space.id) == 2 &&
              H5Sget_simple_extent_dims(matrix_space.id, shape, nullptr) >= 0 &&
              shape[0] == n && shape[1] == n,
          "Covariance matrix dimensions do not match the endmember names.");
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values(
      static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  require(H5Dread(matrix.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  values.data()) >= 0,
          "Cannot read the HDF5 covariance matrix.");
  result.covariance_matrix = values;
  return result;
}
} // namespace

namespace HP_2011_ds62 {
CovarianceData cov() {
  static const auto data = read_covariance(
      "HP_2011_ds62-covariance.h5", detail::HP_2011_ds62_covariance_data,
      sizeof(detail::HP_2011_ds62_covariance_data));
  return data;
}
} // namespace HP_2011_ds62
namespace HGP_2018_ds633 {
CovarianceData cov() {
  static const auto data = read_covariance(
      "HGP_2018_ds633-covariance.h5", detail::HGP_2018_ds633_covariance_data,
      sizeof(detail::HGP_2018_ds633_covariance_data));
  return data;
}
} // namespace HGP_2018_ds633
namespace HPx_ds636 {
CovarianceData cov() {
  static const auto data = read_covariance(
      "HPx_ds636-covariance.h5", detail::HPx_ds636_covariance_data,
      sizeof(detail::HPx_ds636_covariance_data));
  return data;
}
} // namespace HPx_ds636
} // namespace burnman::minerals
