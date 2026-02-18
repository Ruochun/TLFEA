/*==============================================================
 *==============================================================
 * Project: TLFEA
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedAdamWNocoop.cuh
 * Brief:   Declares the non-cooperative GPU-synchronized AdamW solver.
 *          Provides host/device storage and kernels to evaluate residuals and
 *          apply AdamW updates for ANCF3243, ANCF3443, and FEAT10 element data.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cublas_v2.h>

#include <iostream>

#include "SyncedAdamW.cuh"

namespace tlfea {

using SyncedAdamWNocoopParams = SyncedAdamWParams;

class SyncedAdamWNocoopSolver : public SolverBase {
 public:
  SyncedAdamWNocoopSolver(ElementBase *data, int n_constraints)
      : n_coef_(data->get_n_coef()),
        n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints) {
    if (data->type == TYPE_3243) {
      type_             = TYPE_3243;
      auto *typed_data  = static_cast<GPU_ANCF3243_Data *>(data);
      d_data_           = typed_data->d_data;
      d_constraint_ptr_ = typed_data->Get_Is_Constraint_Setup()
                              ? typed_data->Get_Constraint_Ptr()
                              : nullptr;
      n_total_qp_       = Quadrature::N_TOTAL_QP_3_2_2;
      n_shape_          = Quadrature::N_SHAPE_3243;
      typed_data->CalcDsDuPre();
    } else if (data->type == TYPE_3443) {
      type_             = TYPE_3443;
      auto *typed_data  = static_cast<GPU_ANCF3443_Data *>(data);
      d_data_           = typed_data->d_data;
      d_constraint_ptr_ = typed_data->Get_Is_Constraint_Setup()
                              ? typed_data->Get_Constraint_Ptr()
                              : nullptr;
      n_total_qp_       = Quadrature::N_TOTAL_QP_4_4_3;
      n_shape_          = Quadrature::N_SHAPE_3443;
      typed_data->CalcDsDuPre();
    } else if (data->type == TYPE_T10) {
      type_             = TYPE_T10;
      auto *typed_data  = static_cast<GPU_FEAT10_Data *>(data);
      d_data_           = typed_data->d_data;
      d_constraint_ptr_ = typed_data->Get_Is_Constraint_Setup()
                              ? typed_data->Get_Constraint_Ptr()
                              : nullptr;
      n_total_qp_       = Quadrature::N_QP_T10_5;
      n_shape_          = Quadrature::N_NODE_T10_10;
    } else {
      d_data_           = nullptr;
      d_constraint_ptr_ = nullptr;
      MOPHI_ERROR("Unknown element type!");
    }

    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_v_k_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_v_next_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(Real));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_norm_g_, sizeof(Real));
    cudaMalloc(&d_norm_v_, sizeof(Real));
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

    cudaMalloc(&d_adamw_solver_, sizeof(SyncedAdamWNocoopSolver));

    cudaMalloc(&d_lr_, sizeof(Real));
    cudaMalloc(&d_beta1_, sizeof(Real));
    cudaMalloc(&d_beta2_, sizeof(Real));
    cudaMalloc(&d_eps_, sizeof(Real));
    cudaMalloc(&d_weight_decay_, sizeof(Real));
    cudaMalloc(&d_lr_decay_, sizeof(Real));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(Real));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(Real));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(Real));

    cudaMalloc(&d_m_, n_coef_ * 3 * sizeof(Real));
    cudaMalloc(&d_v_adam_, n_coef_ * 3 * sizeof(Real));

    cublasCreate(&cublas_handle_);
    cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
    cudaMalloc(&d_norm_temp_cublas_, sizeof(Real));
  }

  ~SyncedAdamWNocoopSolver() {
    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_v_k_);
    cudaFree(d_v_next_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
    cudaFree(d_norm_g_);
    cudaFree(d_norm_v_);
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

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);

    cudaFree(d_m_);
    cudaFree(d_v_adam_);

    if (cublas_handle_)
      cublasDestroy(cublas_handle_);
    cudaFree(d_norm_temp_cublas_);
  }

  void SetParameters(void *params) override {
    SyncedAdamWNocoopParams *p = static_cast<SyncedAdamWNocoopParams *>(params);

    cudaMemcpy(d_lr_, &p->lr, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta1_, &p->beta1, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta2_, &p->beta2, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps_, &p->eps, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_decay_, &p->weight_decay, sizeof(Real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lr_decay_, &p->lr_decay, sizeof(Real),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_inner_tol_, &p->inner_tol, sizeof(Real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inner_rtol_, &p->inner_rtol, sizeof(Real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_outer_tol_, &p->outer_tol, sizeof(Real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_outer_, &p->max_outer, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_inner_, &p->max_inner, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_time_step_, &p->time_step, sizeof(Real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_solver_rho_, &p->rho, sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convergence_check_interval_, &p->convergence_check_interval,
               sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_v_guess_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(Real));
  }

  void Setup() {
    cudaMemset(d_x12_prev, 0, n_coef_ * sizeof(Real));
    cudaMemset(d_y12_prev, 0, n_coef_ * sizeof(Real));
    cudaMemset(d_z12_prev, 0, n_coef_ * sizeof(Real));

    cudaMemset(d_v_guess_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_v_k_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_v_next_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(Real));
    cudaMemset(d_g_, 0, n_coef_ * 3 * sizeof(Real));

    cudaMemset(d_m_, 0, n_coef_ * 3 * sizeof(Real));
    cudaMemset(d_v_adam_, 0, n_coef_ * 3 * sizeof(Real));

    MOPHI_GPU_CALL(cudaMemcpy(d_adamw_solver_, this,
                            sizeof(SyncedAdamWNocoopSolver),
                            cudaMemcpyHostToDevice));
  }

#if defined(__CUDACC__)
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
  __device__ Map<VectorXR> m() {
    return Map<VectorXR>(d_m_, 3 * n_coef_);
  }
  __device__ Map<VectorXR> v_adam() {
    return Map<VectorXR>(d_v_adam_, 3 * n_coef_);
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

  __device__ Real *norm_g() {
    return d_norm_g_;
  }
  __device__ Real *norm_v() {
    return d_norm_v_;
  }
  __device__ int *inner_flag() {
    return d_inner_flag_;
  }
  __device__ int *outer_flag() {
    return d_outer_flag_;
  }
  __device__ Real *solver_rho() {
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

  void OneStepAdamWNocoop();

  void Solve() override {
    OneStepAdamWNocoop();
  }

 private:
  Real compute_l2_norm_cublas(Real *d_vec, int n_dofs);

  ElementType type_;
  ElementBase *d_data_;
  SyncedAdamWNocoopSolver *d_adamw_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  Real *d_constraint_ptr_;

  Real *d_x12_prev, *d_y12_prev, *d_z12_prev;

  Real *d_v_guess_, *d_v_prev_, *d_v_k_, *d_v_next_;
  Real *d_lambda_guess_, *d_g_;
  Real *d_norm_g_, *d_norm_v_;
  int *d_inner_flag_, *d_outer_flag_;

  Real *d_lr_, *d_beta1_, *d_beta2_, *d_eps_, *d_weight_decay_, *d_lr_decay_;

  Real *d_alpha_, *d_inner_tol_, *d_inner_rtol_, *d_outer_tol_, *d_time_step_,
      *d_solver_rho_;

  int *d_convergence_check_interval_;

  int *d_max_inner_, *d_max_outer_;

  Real *d_m_, *d_v_adam_;

  cublasHandle_t cublas_handle_;
  Real *d_norm_temp_cublas_;
};

}  // namespace tlfea
