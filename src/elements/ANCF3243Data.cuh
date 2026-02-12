/*==============================================================
 * Stub header for ANCF3243 - not implemented in this minimal version
 * Only FEAT10 element is supported
 *==============================================================*/
#pragma once

#include "ElementBase.h"

// Forward declaration only - not implemented
class GPU_ANCF3243_Data : public ElementBase {
public:
    void CalcDsDuPre() {}
    __host__ __device__ int get_n_beam() const override { return 0; }
    __host__ __device__ int get_n_coef() const override { return 0; }
    void CalcMassMatrix() override {}
    void CalcInternalForce() override {}
    void CalcConstraintData() override {}
    void CalcP() override {}
    void RetrieveInternalForceToCPU(Eigen::VectorXd &) override {}
    void RetrieveConstraintDataToCPU(Eigen::VectorXd &) override {}
    void RetrieveConstraintJacobianToCPU(Eigen::MatrixXd &) override {}
    void RetrievePositionToCPU(Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &) override {}
    void RetrieveDeformationGradientToCPU(std::vector<std::vector<Eigen::MatrixXd>> &) override {}
    void RetrievePFromFToCPU(std::vector<std::vector<Eigen::MatrixXd>> &) override {}
};
