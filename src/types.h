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

// Define Real as the primary floating-point type for the project
typedef double Real;

// Eigen type aliases using Real
namespace Eigen {
    typedef Matrix<Real, Dynamic, Dynamic> MatrixXR;
    typedef Matrix<Real, Dynamic, 1> VectorXR;
    typedef Matrix<Real, 3, 3> Matrix3R;
    typedef Matrix<Real, 3, 1> Vector3R;
}
