#pragma once
/*==============================================================
 *==============================================================
 * Project: TLFEA
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedAdamW.cuh
 * Brief:   Declares the SyncedAdamWSolver class and its GPU-facing state for
 *          a fully synchronized first-order AdamW method. Manages device
 *          buffers for velocities, dual variables, convergence flags, and
 *          time-stepping parameters, and provides device accessors used by the
 *          AdamW kernel implementations.
 *==============================================================
 *==============================================================*/

#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT4Data.cuh"
#include "SolverBase.h"
#include <MoPhiEssentials.h>
#include "../types.h"

namespace tlfea {

// this is a first order AdamW method
// fully synced, and each inner iteration will compute the full gradient

struct SyncedAdamWParams {
    Real lr, beta1, beta2, eps, weight_decay, lr_decay;
    Real inner_tol, outer_tol, rho;
    int max_outer, max_inner;
    Real time_step;
    int convergence_check_interval;
    Real inner_rtol;
};

class SyncedAdamWSolver : public SolverBase {
  public:
    SyncedAdamWSolver(ElementBase* data, int n_constraints)
        : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()), n_constraints_(n_constraints) {
        // Type-based casting to get the correct d_data from derived class
        if (data->type == TYPE_3243) {
            type_ = TYPE_3243;
            auto* typed_data = static_cast<GPU_ANCF3243_Data*>(data);
            d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
            n_total_qp_ = Quadrature::N_TOTAL_QP_3_2_2;
            n_shape_ = Quadrature::N_SHAPE_3243;
            typed_data->CalcDsDuPre();
        } else if (data->type == TYPE_3443) {
            type_ = TYPE_3443;
            auto* typed_data = static_cast<GPU_ANCF3443_Data*>(data);
            d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
            n_total_qp_ = Quadrature::N_TOTAL_QP_4_4_3;
            n_shape_ = Quadrature::N_SHAPE_3443;
            typed_data->CalcDsDuPre();
        } else if (data->type == TYPE_T10) {
            type_ = TYPE_T10;
            auto* typed_data = static_cast<GPU_FEAT10_Data*>(data);
            d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
            n_total_qp_ = Quadrature::N_QP_T10_5;
            n_shape_ = Quadrature::N_NODE_T10_10;
        } else if (data->type == TYPE_T4) {
            type_ = TYPE_T4;
            auto* typed_data = static_cast<GPU_FEAT4_Data*>(data);
            d_data_ = typed_data->d_data;
            n_total_qp_ = Quadrature::N_QP_T4_1;
            n_shape_ = Quadrature::N_NODE_T4_4;
        } else {
            d_data_ = nullptr;
            MOPHI_ERROR("Unknown element type!");
        }

        if (d_data_ == nullptr) {
            MOPHI_ERROR("d_data_ is null in SyncedAdamWSolver constructor");
        }

        cudaMalloc(&d_norm_g_, sizeof(Real));
        cudaMalloc(&d_inner_flag_, sizeof(int));
        cudaMalloc(&d_outer_flag_, sizeof(int));
        cudaMalloc(&d_alpha_, sizeof(Real));
        cudaMalloc(&d_inner_tol_, sizeof(Real));
        cudaMalloc(&d_inner_rtol_, sizeof(Real));
        cudaMalloc(&d_outer_tol_, sizeof(Real));
        cudaMalloc(&d_max_outer_, sizeof(int));
        cudaMalloc(&d_max_inner_, sizeof(int));
        cudaMalloc(&d_time_step_, sizeof(Real));
        cudaMalloc(&d_solver_rho_, sizeof(Real));
        cudaMalloc(&d_convergence_check_interval_, sizeof(int));

        cudaMalloc(&d_adamw_solver_, sizeof(SyncedAdamWSolver));

        cudaMalloc(&d_lr_, sizeof(Real));
        cudaMalloc(&d_beta1_, sizeof(Real));
        cudaMalloc(&d_beta2_, sizeof(Real));
        cudaMalloc(&d_eps_, sizeof(Real));
        cudaMalloc(&d_weight_decay_, sizeof(Real));
        cudaMalloc(&d_lr_decay_, sizeof(Real));

        // Long arrays: use DualArray (manages both pinned host and device memory).
        da_v_guess_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_guess_.BindDevicePointer(&d_v_guess_);
        da_v_prev_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_prev_.BindDevicePointer(&d_v_prev_);
        da_v_k_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_k_.BindDevicePointer(&d_v_k_);
        da_v_next_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_next_.BindDevicePointer(&d_v_next_);
        da_lambda_guess_.resize(static_cast<size_t>(n_constraints_));
        da_lambda_guess_.BindDevicePointer(&d_lambda_guess_);
        da_g_.resize(static_cast<size_t>(n_coef_) * 3);
        da_g_.BindDevicePointer(&d_g_);
        da_x12_prev_.resize(static_cast<size_t>(n_coef_));
        da_x12_prev_.BindDevicePointer(&d_x12_prev);
        da_y12_prev_.resize(static_cast<size_t>(n_coef_));
        da_y12_prev_.BindDevicePointer(&d_y12_prev);
        da_z12_prev_.resize(static_cast<size_t>(n_coef_));
        da_z12_prev_.BindDevicePointer(&d_z12_prev);
    }

    ~SyncedAdamWSolver() {
        // Long arrays managed by DualArrays
        da_v_guess_.free();
        da_v_prev_.free();
        da_v_k_.free();
        da_v_next_.free();
        da_lambda_guess_.free();
        da_g_.free();
        da_x12_prev_.free();
        da_y12_prev_.free();
        da_z12_prev_.free();

        cudaFree(d_norm_g_);
        cudaFree(d_inner_flag_);
        cudaFree(d_outer_flag_);
        cudaFree(d_alpha_);
        cudaFree(d_inner_tol_);
        cudaFree(d_inner_rtol_);
        cudaFree(d_outer_tol_);
        cudaFree(d_max_outer_);
        cudaFree(d_max_inner_);
        cudaFree(d_time_step_);
        cudaFree(d_solver_rho_);
        cudaFree(d_convergence_check_interval_);

        cudaFree(d_lr_);
        cudaFree(d_beta1_);
        cudaFree(d_beta2_);
        cudaFree(d_eps_);
        cudaFree(d_weight_decay_);
        cudaFree(d_lr_decay_);

        cudaFree(d_adamw_solver_);
    }

    void SetParameters(void* params) override {
        SyncedAdamWParams* p = static_cast<SyncedAdamWParams*>(params);

        cudaMemcpy(d_lr_, &p->lr, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta1_, &p->beta1, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta2_, &p->beta2, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps_, &p->eps, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight_decay_, &p->weight_decay, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lr_decay_, &p->lr_decay, sizeof(Real), cudaMemcpyHostToDevice);

        cudaMemcpy(d_inner_tol_, &p->inner_tol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inner_rtol_, &p->inner_rtol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outer_tol_, &p->outer_tol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_outer_, &p->max_outer, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_inner_, &p->max_inner, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_time_step_, &p->time_step, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_solver_rho_, &p->rho, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_convergence_check_interval_, &p->convergence_check_interval, sizeof(int), cudaMemcpyHostToDevice);

        da_v_guess_.SetVal(Real(0));
        da_v_guess_.MakeReadyDevice();
        da_v_prev_.SetVal(Real(0));
        da_v_prev_.MakeReadyDevice();
        da_lambda_guess_.SetVal(Real(0));
        da_lambda_guess_.MakeReadyDevice();
    }

    void Setup() {
        da_x12_prev_.SetVal(Real(0));
        da_x12_prev_.MakeReadyDevice();
        da_y12_prev_.SetVal(Real(0));
        da_y12_prev_.MakeReadyDevice();
        da_z12_prev_.SetVal(Real(0));
        da_z12_prev_.MakeReadyDevice();

        da_v_guess_.SetVal(Real(0));
        da_v_guess_.MakeReadyDevice();
        da_v_prev_.SetVal(Real(0));
        da_v_prev_.MakeReadyDevice();
        da_v_k_.SetVal(Real(0));
        da_v_k_.MakeReadyDevice();
        da_v_next_.SetVal(Real(0));
        da_v_next_.MakeReadyDevice();
        da_lambda_guess_.SetVal(Real(0));
        da_lambda_guess_.MakeReadyDevice();
        da_g_.SetVal(Real(0));
        da_g_.MakeReadyDevice();

        MOPHI_GPU_CALL(cudaMemcpy(d_adamw_solver_, this, sizeof(SyncedAdamWSolver), cudaMemcpyHostToDevice));
    }

#if defined(__CUDACC__)
    // Device accessors (define as __device__ in .cuh or .cu as needed)
    __device__ Map<VectorXR> v_guess() {
        return Map<VectorXR>(d_v_guess_, n_coef_ * 3);
    }
    __device__ Map<VectorXR> v_prev() {
        return Map<VectorXR>(d_v_prev_, n_coef_ * 3);
    }
    __device__ Map<VectorXR> v_k() {
        return Map<VectorXR>(d_v_k_, n_coef_ * 3);
    }
    __device__ Map<VectorXR> v_next() {
        return Map<VectorXR>(d_v_next_, n_coef_ * 3);
    }
    __device__ Map<VectorXR> lambda_guess() {
        return Map<VectorXR>(d_lambda_guess_, n_constraints_);
    }
    __device__ Map<VectorXR> g() {
        return Map<VectorXR>(d_g_, 3 * n_coef_);
    }

    __device__ int gpu_n_constraints() {
        return n_constraints_;
    }
    __device__ int gpu_n_total_qp() {
        return n_total_qp_;
    }
    __device__ int gpu_n_shape() {
        return n_shape_;
    }

    __device__ Real* norm_g() {
        return d_norm_g_;
    }
    __device__ int* inner_flag() {
        return d_inner_flag_;
    }
    __device__ int* outer_flag() {
        return d_outer_flag_;
    }
    __device__ Real* solver_rho() {
        return d_solver_rho_;
    }
    __device__ Real solver_alpha() const {
        return *d_alpha_;
    }
    __device__ Real solver_inner_tol() const {
        return *d_inner_tol_;
    }
    __device__ Real solver_inner_rtol() const {
        return *d_inner_rtol_;
    }
    __device__ Real solver_outer_tol() const {
        return *d_outer_tol_;
    }
    __device__ int solver_max_outer() const {
        return *d_max_outer_;
    }
    __device__ int solver_max_inner() const {
        return *d_max_inner_;
    }
    __device__ Real solver_time_step() const {
        return *d_time_step_;
    }
    __device__ Real solver_lr() const {
        return *d_lr_;
    }
    __device__ Real solver_beta1() const {
        return *d_beta1_;
    }
    __device__ Real solver_beta2() const {
        return *d_beta2_;
    }
    __device__ Real solver_eps() const {
        return *d_eps_;
    }
    __device__ Real solver_weight_decay() const {
        return *d_weight_decay_;
    }
    __device__ Real solver_lr_decay() const {
        return *d_lr_decay_;
    }

    __device__ int solver_convergence_check_interval() const {
        return *d_convergence_check_interval_;
    }

    __device__ Map<VectorXR> x12_prev() {
        return Map<VectorXR>(d_x12_prev, n_coef_);
    }
    __device__ Map<VectorXR> y12_prev() {
        return Map<VectorXR>(d_y12_prev, n_coef_);
    }
    __device__ Map<VectorXR> z12_prev() {
        return Map<VectorXR>(d_z12_prev, n_coef_);
    }
#endif

    __host__ __device__ int get_n_coef() const {
        return n_coef_;
    }
    __host__ __device__ int get_n_beam() const {
        return n_beam_;
    }

    void OneStepAdamW();

    void Solve() override {
        OneStepAdamW();
    }

  private:
    ElementType type_;
    ElementBase* d_data_;
    SyncedAdamWSolver* d_adamw_solver_;
    int n_total_qp_, n_shape_;
    int n_coef_, n_beam_, n_constraints_;

    // DualArrays for long arrays (manage both pinned host and device memory).
    mophi::DualArray<Real> da_v_guess_, da_v_prev_, da_v_k_, da_v_next_;
    mophi::DualArray<Real> da_lambda_guess_, da_g_;
    mophi::DualArray<Real> da_x12_prev_, da_y12_prev_, da_z12_prev_;

    // Raw device pointers for GPU kernel access (bound to DualArrays above).
    Real *d_x12_prev, *d_y12_prev, *d_z12_prev;
    Real *d_v_guess_, *d_v_prev_, *d_v_k_, *d_v_next_;
    Real *d_lambda_guess_, *d_g_;
    Real* d_norm_g_;
    int *d_inner_flag_, *d_outer_flag_;

    Real *d_lr_, *d_beta1_, *d_beta2_, *d_eps_, *d_weight_decay_, *d_lr_decay_;

    Real *d_alpha_, *d_inner_tol_, *d_inner_rtol_, *d_outer_tol_, *d_time_step_, *d_solver_rho_;

    int* d_convergence_check_interval_;

    int *d_max_inner_, *d_max_outer_;
};

}  // namespace tlfea
