/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    types.h
 * Brief:   Defines the Real type as an alias for double and provides
 *          type aliases for Eigen matrix and vector types using Real.
 *          This allows easy switching between different floating-point
 *          precisions throughout the codebase.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Dense>

namespace tlfea {

// Define Real as the primary floating-point type for the project
typedef double Real;

// Forward declaration for Dynamic size and storage options
static constexpr int Dynamic = Eigen::Dynamic;
static constexpr int RowMajor = Eigen::RowMajor;
static constexpr int ColMajor = Eigen::ColMajor;

// Wrap Eigen::Matrix under tlfea namespace
// This allows future flexibility to change the underlying implementation
template<typename Scalar, int Rows, int Cols, int Options = 0>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options>;

// Wrap Eigen::Map under tlfea namespace
// This allows future flexibility to change the underlying implementation
template<typename PlainObjectType, int MapOptions = Eigen::Unaligned, typename StrideType = Eigen::Stride<0, 0>>
using Map = Eigen::Map<PlainObjectType, MapOptions, StrideType>;

// Type aliases using Real and our Matrix template
typedef Matrix<Real, Dynamic, Dynamic> MatrixXR;
typedef Matrix<Real, Dynamic, 1> VectorXR;
typedef Matrix<Real, 3, 3> Matrix3R;
typedef Matrix<Real, 3, 1> Vector3R;
typedef Matrix<int, Dynamic, Dynamic> MatrixXi;
typedef Matrix<int, Dynamic, 1> VectorXi;

}  // namespace tlfea
