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

// Eigen type aliases using Real
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixXR;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> VectorXR;
typedef Eigen::Matrix<Real, 3, 3> Matrix3R;
typedef Eigen::Matrix<Real, 3, 1> Vector3R;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatrixXi;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;

}  // namespace tlfea
