#include <cuda_runtime.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <cstring>
#include <iostream>
#include <vector>
#include <MoPhiEssentials.h>

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3243Data.cuh
 * Brief:   Declares the GPU_ANCF3243_Data structure and host/GPU interfaces
 *          for ANCF 3243 beam elements. Encapsulates mesh connectivity,
 *          quadrature configuration, CSR mass matrices, internal/external
 *          force vectors, and constraint storage shared with solvers.
 *==============================================================
 *==============================================================*/

#include "../utils/cpu_utils.h"
#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "../materials/MaterialModel.cuh"
#include "ElementBase.h"
#include "types.h"

// Definition of GPU_ANCF3243 and data access device functions
#pragma once

namespace tlfea {

//
// define a SAP data strucutre
struct GPU_ANCF3243_Data : public ElementBase {
    // Constraint modes:
    // - kConstraintFixedCoefficients: legacy fixed coefficient constraints via
    //   d_fixed_nodes[] and x12_jac/y12_jac/z12_jac targets.
    // - kConstraintLinearCSR: general linear constraints with explicit J in CSR
    //   and per-row rhs targets.
    static constexpr int kConstraintNone = 0;
    static constexpr int kConstraintFixedCoefficients = 1;
    static constexpr int kConstraintLinearCSR = 2;

#if defined(__CUDACC__)

    // Const get functions
    __device__ const Map<MatrixXR> B_inv(int elem_idx) const {
        const int row_size = Quadrature::N_SHAPE_3243;
        const int col_size = Quadrature::N_SHAPE_3243;
        return Map<MatrixXR>(d_B_inv + elem_idx * row_size * col_size, row_size, col_size);
    }

    __device__ Map<MatrixXR> grad_N_ref(int elem_idx, int qp_idx) const {
        const int row_size = Quadrature::N_SHAPE_3243;
        const int col_size = 3;
        Real* qp_data = d_grad_N_ref + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * row_size * col_size;
        return Map<MatrixXR>(qp_data, row_size, col_size);
    }

    __device__ Real& detJ_ref(int elem_idx, int qp_idx) {
        return d_detJ_ref[elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx];
    }

    __device__ Real detJ_ref(int elem_idx, int qp_idx) const {
        return d_detJ_ref[elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx];
    }

    __device__ const Map<VectorXR> gauss_xi_m() const {
        return Map<VectorXR>(d_gauss_xi_m, Quadrature::N_QP_6);
    }

    __device__ const Map<VectorXR> gauss_xi() const {
        return Map<VectorXR>(d_gauss_xi, Quadrature::N_QP_3);
    }

    __device__ const Map<VectorXR> gauss_eta() const {
        return Map<VectorXR>(d_gauss_eta, Quadrature::N_QP_2);
    }

    __device__ const Map<VectorXR> gauss_zeta() const {
        return Map<VectorXR>(d_gauss_zeta, Quadrature::N_QP_2);
    }

    __device__ const Map<VectorXR> weight_xi_m() const {
        return Map<VectorXR>(d_weight_xi_m, Quadrature::N_QP_6);
    }

    __device__ const Map<VectorXR> weight_xi() const {
        return Map<VectorXR>(d_weight_xi, Quadrature::N_QP_3);
    }

    __device__ const Map<VectorXR> weight_eta() const {
        return Map<VectorXR>(d_weight_eta, Quadrature::N_QP_2);
    }

    __device__ const Map<VectorXR> weight_zeta() const {
        return Map<VectorXR>(d_weight_zeta, Quadrature::N_QP_2);
    }

    __device__ void gather_element_dofs(const Real* global, int elem, Real* local) const {
        const int node0 = element_node(elem, 0);
        const int node1 = element_node(elem, 1);
    #pragma unroll
        for (int d = 0; d < 4; ++d) {
            local[0 * 4 + d] = global[node0 * 4 + d];
            local[1 * 4 + d] = global[node1 * 4 + d];
        }
    }

    __device__ void x12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_x12_jac, elem, buffer);
    }
    __device__ void y12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_y12_jac, elem, buffer);
    }
    __device__ void z12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_z12_jac, elem, buffer);
    }

    __device__ Map<VectorXR> x12_jac() {
        return Map<VectorXR>(d_x12_jac, n_coef);
    }

    __device__ Map<VectorXR> const x12_jac() const {
        return Map<VectorXR>(d_x12_jac, n_coef);
    }

    __device__ Map<VectorXR> y12_jac() {
        return Map<VectorXR>(d_y12_jac, n_coef);
    }

    __device__ Map<VectorXR> const y12_jac() const {
        return Map<VectorXR>(d_y12_jac, n_coef);
    }

    __device__ Map<VectorXR> z12_jac() {
        return Map<VectorXR>(d_z12_jac, n_coef);
    }

    __device__ Map<VectorXR> const z12_jac() const {
        return Map<VectorXR>(d_z12_jac, n_coef);
    }

    __device__ Map<VectorXR> x12() {
        return Map<VectorXR>(d_x12, n_coef);
    }

    __device__ Map<VectorXR> const x12() const {
        return Map<VectorXR>(d_x12, n_coef);
    }

    __device__ Map<VectorXR> y12() {
        return Map<VectorXR>(d_y12, n_coef);
    }

    __device__ Map<VectorXR> const y12() const {
        return Map<VectorXR>(d_y12, n_coef);
    }

    __device__ Map<VectorXR> z12() {
        return Map<VectorXR>(d_z12, n_coef);
    }

    __device__ Map<VectorXR> const z12() const {
        return Map<VectorXR>(d_z12, n_coef);
    }

    __device__ Map<VectorXR> x12(int elem) {
        return Map<VectorXR>(d_x12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<VectorXR> const x12(int elem) const {
        return Map<VectorXR>(d_x12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<VectorXR> y12(int elem) {
        return Map<VectorXR>(d_y12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<VectorXR> const y12(int elem) const {
        return Map<VectorXR>(d_y12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<VectorXR> z12(int elem) {
        return Map<VectorXR>(d_z12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<VectorXR> const z12(int elem) const {
        return Map<VectorXR>(d_z12 + elem * (Quadrature::N_SHAPE_3243 / 2), Quadrature::N_SHAPE_3243);
    }

    __device__ Map<MatrixXR> F(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> F(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<MatrixXR> P(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    // Time-derivative of deformation gradient (viscous computation)
    __device__ Map<MatrixXR> Fdot(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> Fdot(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    // Viscous Piola stress storage
    __device__ Map<MatrixXR> P_vis(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P_vis(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<VectorXR> f_int(int global_node_idx) {
        return Map<VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ const Map<VectorXR> f_int(int global_node_idx) const {
        return Map<VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ Map<VectorXR> f_int() {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ const Map<VectorXR> f_int() const {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ Map<VectorXR> f_ext(int global_node_idx) {
        return Map<VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ const Map<VectorXR> f_ext(int global_node_idx) const {
        return Map<VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ Map<VectorXR> f_ext() {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ const Map<VectorXR> f_ext() const {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ Map<VectorXR> constraint() {
        return Map<VectorXR>(d_constraint, n_constraint);
    }

    __device__ const Map<VectorXR> constraint() const {
        return Map<VectorXR>(d_constraint, n_constraint);
    }

    __device__ const Real* constraint_rhs() const {
        return d_constraint_rhs;
    }

    __device__ int constraint_mode_device() const {
        return constraint_mode;
    }

    __device__ Map<VectorXi> fixed_nodes() {
        return Map<VectorXi>(d_fixed_nodes, n_constraint / 3);
    }

    // ================================

    __device__ int element_node(int elem, int local_node_idx) const {
        return d_element_connectivity[elem * 2 + local_node_idx];
    }

    __device__ Real L(int elem_idx) const {
        return d_L[elem_idx];
    }

    __device__ Real W(int elem_idx) const {
        return d_W[elem_idx];
    }

    __device__ Real H(int elem_idx) const {
        return d_H[elem_idx];
    }

    __device__ Real rho0() const {
        return *d_rho0;
    }

    __device__ Real nu() const {
        return *d_nu;
    }

    __device__ Real E() const {
        return *d_E;
    }

    __device__ Real lambda() const {
        return *d_lambda;
    }

    __device__ Real mu() const {
        return *d_mu;
    }

    __device__ int material_model() const {
        return *d_material_model;
    }

    __device__ Real mu10() const {
        return *d_mu10;
    }

    __device__ Real mu01() const {
        return *d_mu01;
    }

    __device__ Real kappa() const {
        return *d_kappa;
    }

    __device__ Real eta_damp() const {
        return *d_eta_damp;
    }

    __device__ Real lambda_damp() const {
        return *d_lambda_damp;
    }
    __device__ int gpu_n_beam() const {
        return n_beam;
    }

    __device__ int gpu_n_coef() const {
        return n_coef;
    }

    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }
    //===========================================

    __device__ int* csr_offsets() {
        return d_csr_offsets;
    }

    __device__ int* csr_columns() {
        return d_csr_columns;
    }

    __device__ Real* csr_values() {
        return d_csr_values;
    }

    __device__ int* cj_csr_offsets() {
        return d_cj_csr_offsets;
    }

    __device__ int* cj_csr_columns() {
        return d_cj_csr_columns;
    }

    __device__ Real* cj_csr_values() {
        return d_cj_csr_values;
    }

    __device__ int* j_csr_offsets() {
        return d_j_csr_offsets;
    }

    __device__ int* j_csr_columns() {
        return d_j_csr_columns;
    }

    __device__ Real* j_csr_values() {
        return d_j_csr_values;
    }

    __device__ int nnz() {
        return *d_nnz;
    }

#endif
    __host__ __device__ int get_n_beam() const {
        return n_beam;
    }
    __host__ __device__ int get_n_coef() const {
        return n_coef;
    }
    __host__ __device__ int get_n_constraint() const {
        return n_constraint;
    }

    // Constructor
    GPU_ANCF3243_Data(int num_nodes, int num_elements) : n_nodes(num_nodes), n_elements(num_elements) {
        n_beam = num_elements;  // Initialize n_beam with n_elements
        n_coef = 4 * n_nodes;   // Non-overlapping DOFs: 4 DOFs per node
        type = TYPE_3243;
        constraint_mode = kConstraintNone;
    }

    void Initialize() {
        MOPHI_GPU_CALL(
            cudaMalloc(&d_B_inv, n_beam * Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_grad_N_ref,
                                  n_beam * Quadrature::N_TOTAL_QP_3_2_2 * Quadrature::N_SHAPE_3243 * 3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_detJ_ref, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_gauss_xi_m, Quadrature::N_QP_6 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_gauss_xi, Quadrature::N_QP_3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_gauss_eta, Quadrature::N_QP_2 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_gauss_zeta, Quadrature::N_QP_2 * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_weight_xi_m, Quadrature::N_QP_6 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_weight_xi, Quadrature::N_QP_3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_weight_eta, Quadrature::N_QP_2 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_weight_zeta, Quadrature::N_QP_2 * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_x12_jac, n_coef * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_y12_jac, n_coef * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_z12_jac, n_coef * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_x12, n_coef * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_y12, n_coef * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_z12, n_coef * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_element_connectivity, n_elements * 2 * sizeof(int)));

        MOPHI_GPU_CALL(cudaMalloc(&d_F, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_P, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real)));
        // Kelvin-Voigt damping related buffers
        MOPHI_GPU_CALL(cudaMalloc(&d_Fdot, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_P_vis, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real)));
        // damping parameters (single scalar copied to device)
        MOPHI_GPU_CALL(cudaMalloc(&d_eta_damp, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda_damp, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_f_int, n_coef * 3 * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_f_ext, n_coef * 3 * sizeof(Real)));

        // copy struct to device
        MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_ANCF3243_Data)));

        // beam data
        MOPHI_GPU_CALL(cudaMalloc(&d_H, n_beam * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_W, n_beam * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_L, n_beam * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_rho0, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_nu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_E, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_material_model, sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu10, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu01, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_kappa, sizeof(Real)));
    }

    void Setup(const VectorXR& length,
               const VectorXR& width,
               const VectorXR& height,
               const VectorXR& gauss_xi_m,
               const VectorXR& gauss_xi,
               const VectorXR& gauss_eta,
               const VectorXR& gauss_zeta,
               const VectorXR& weight_xi_m,
               const VectorXR& weight_xi,
               const VectorXR& weight_eta,
               const VectorXR& weight_zeta,
               const VectorXR& h_x12,
               const VectorXR& h_y12,
               const VectorXR& h_z12,
               const Matrix<int, Dynamic, 2, RowMajor>& h_element_connectivity) {
        if (is_setup) {
            MOPHI_ERROR(std::string("GPU_ANCF3243_Data is already set up."));
            return;
        }

        if (length.size() != n_beam || width.size() != n_beam || height.size() != n_beam) {
            MOPHI_ERROR(std::string("GPU_ANCF3243_Data::Setup: length/width/height must have size n_beam."));
            return;
        }

        VectorXR h_B_inv_flat;
        try {
            ANCFCPUUtils::ANCF3243_B12_matrix_flat_per_element(length, width, height, h_B_inv_flat,
                                                               Quadrature::N_SHAPE_3243);
        } catch (const std::exception& e) {
            MOPHI_ERROR("GPU_ANCF3243_Data::Setup: failed to build per-element B_inv: %s", e.what());
            return;
        }
        const int n_binv = static_cast<int>(h_B_inv_flat.size());

        MOPHI_GPU_CALL(cudaMemcpy(d_B_inv, h_B_inv_flat.data(), n_binv * sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(
            cudaMemcpy(d_gauss_xi_m, gauss_xi_m.data(), Quadrature::N_QP_6 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_gauss_xi, gauss_xi.data(), Quadrature::N_QP_3 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_gauss_eta, gauss_eta.data(), Quadrature::N_QP_2 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_gauss_zeta, gauss_zeta.data(), Quadrature::N_QP_2 * sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(
            cudaMemcpy(d_weight_xi_m, weight_xi_m.data(), Quadrature::N_QP_6 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_weight_xi, weight_xi.data(), Quadrature::N_QP_3 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_weight_eta, weight_eta.data(), Quadrature::N_QP_2 * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_weight_zeta, weight_zeta.data(), Quadrature::N_QP_2 * sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_x12_jac, h_x12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_y12_jac, h_y12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_z12_jac, h_z12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_x12, h_x12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_y12, h_y12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_z12, h_z12.data(), n_coef * sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_element_connectivity, h_element_connectivity.data(), n_elements * 2 * sizeof(int),
                                  cudaMemcpyHostToDevice));

        cudaMemset(d_f_int, 0, n_coef * 3 * sizeof(Real));

        cudaMemset(d_F, 0, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real));
        cudaMemset(d_P, 0, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real));
        // initialize damping buffers to zero
        cudaMemset(d_Fdot, 0, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real));
        cudaMemset(d_P_vis, 0, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(Real));

        MOPHI_GPU_CALL(cudaMemcpy(d_H, height.data(), n_beam * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_W, width.data(), n_beam * sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_L, length.data(), n_beam * sizeof(Real), cudaMemcpyHostToDevice));

        Real rho0 = 0.0;
        Real nu = 0.0;
        Real E = 0.0;
        Real mu = E / (2 * (1 + nu));                        // Shear modulus μ
        Real lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé’s first parameter λ
        Real eta_damp = 0.0;
        Real lambda_damp = 0.0;
        int material_model = MATERIAL_MODEL_SVK;
        Real mu10 = 0.0;
        Real mu01 = 0.0;
        Real kappa = 0.0;

        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(Real), cudaMemcpyHostToDevice));
        // copy damping scalars to device (single Real each)
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));

        is_setup = true;
        is_reference_precomputed = false;
    }

    void Setup(Real length,
               Real width,
               Real height,
               const VectorXR& gauss_xi_m,
               const VectorXR& gauss_xi,
               const VectorXR& gauss_eta,
               const VectorXR& gauss_zeta,
               const VectorXR& weight_xi_m,
               const VectorXR& weight_xi,
               const VectorXR& weight_eta,
               const VectorXR& weight_zeta,
               const VectorXR& h_x12,
               const VectorXR& h_y12,
               const VectorXR& h_z12,
               const Matrix<int, Dynamic, 2, RowMajor>& h_element_connectivity) {
        VectorXR lengths = VectorXR::Constant(n_beam, length);
        VectorXR widths = VectorXR::Constant(n_beam, width);
        VectorXR heights = VectorXR::Constant(n_beam, height);
        Setup(lengths, widths, heights, gauss_xi_m, gauss_xi, gauss_eta, gauss_zeta, weight_xi_m, weight_xi, weight_eta,
              weight_zeta, h_x12, h_y12, h_z12, h_element_connectivity);
    }

    /**
     * Set reference density (used for mass/inertial terms).
     */
    void SetDensity(Real rho0) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data must be set up before setting density.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice));
    }

    /**
     * Set Kelvin-Voigt damping parameters.
     * eta_damp: shear-like damping coefficient
     * lambda_damp: volumetric-like damping coefficient
     */
    void SetDamping(Real eta_damp, Real lambda_damp) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data must be set up before setting damping.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
    }

    /**
     * Select Saint Venant-Kirchhoff (SVK) material model using current E/nu.
     */
    void SetSVK() {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_SVK;
        Real mu10 = 0.0;
        Real mu01 = 0.0;
        Real kappa = 0.0;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetSVK(Real E, Real nu) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data must be set up before setting material.");
            return;
        }

        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(Real), cudaMemcpyHostToDevice));

        Real mu = E / (2 * (1 + nu));
        Real lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(Real), cudaMemcpyHostToDevice));

        SetSVK();
    }

    /**
     * Set compressible Mooney-Rivlin parameters.
     * mu10, mu01: isochoric Mooney-Rivlin coefficients
     * kappa: volumetric penalty (bulk-modulus-like) coefficient
     */
    void SetMooneyRivlin(Real mu10, Real mu01, Real kappa) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetExternalForce(const VectorXR& f_ext) {
        if (f_ext.size() != n_coef * 3) {
            MOPHI_ERROR("External force vector size mismatch.");
            return;
        }
        cudaMemset(d_f_ext, 0, n_coef * 3 * sizeof(Real));
        MOPHI_GPU_CALL(cudaMemcpy(d_f_ext, f_ext.data(), n_coef * 3 * sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetNodalFixed(const VectorXi& fixed_nodes) {
        if (is_constraints_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data CONSTRAINT is already set up.");
            return;
        }

        n_constraint = fixed_nodes.size() * 3;
        constraint_mode = kConstraintFixedCoefficients;

        MOPHI_GPU_CALL(cudaMalloc(&d_constraint, n_constraint * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_fixed_nodes, fixed_nodes.size() * sizeof(int)));

        MOPHI_GPU_CALL(cudaMemset(d_constraint, 0, n_constraint * sizeof(Real)));
        MOPHI_GPU_CALL(
            cudaMemcpy(d_fixed_nodes, fixed_nodes.data(), fixed_nodes.size() * sizeof(int), cudaMemcpyHostToDevice));

        is_constraints_setup = true;
        if (d_data) {
            MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));
        }
    }

    // General linear constraints:
    //   c = J * x - rhs
    // where x is the flattened coefficient vector in solver ordering
    // (coef_index major, then xyz component), J is CSR (rows = constraints,
    // cols = n_coef*3), and rhs is per-row target.
    //
    // This uploads both J (CSR) and J^T (CSR).
    void SetLinearConstraintsCSR(const std::vector<int>& j_offsets,
                                 const std::vector<int>& j_columns,
                                 const std::vector<Real>& j_values,
                                 const VectorXR& rhs) {
        if (is_constraints_setup) {
            MOPHI_ERROR("GPU_ANCF3243_Data CONSTRAINT is already set up.");
            return;
        }
        if (j_offsets.empty() || j_offsets.front() != 0) {
            MOPHI_ERROR("SetLinearConstraintsCSR: invalid offsets.");
            return;
        }
        if (static_cast<int>(rhs.size()) + 1 != static_cast<int>(j_offsets.size())) {
            MOPHI_ERROR("SetLinearConstraintsCSR: offsets/rhs size mismatch.");
            return;
        }
        if (j_columns.size() != j_values.size()) {
            MOPHI_ERROR("SetLinearConstraintsCSR: columns/values size mismatch.");
            return;
        }

        const int nnz = static_cast<int>(j_columns.size());
        if (j_offsets.back() != nnz) {
            MOPHI_ERROR("SetLinearConstraintsCSR: offsets.back != nnz.");
            return;
        }

        const int n_dofs = n_coef * 3;
        for (int c : j_columns) {
            if (c < 0 || c >= n_dofs) {
                MOPHI_ERROR("SetLinearConstraintsCSR: column out of range.");
                return;
            }
        }

        d_fixed_nodes = nullptr;
        n_constraint = static_cast<int>(rhs.size());
        constraint_mode = kConstraintLinearCSR;

        MOPHI_GPU_CALL(cudaMalloc(&d_constraint, n_constraint * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMemset(d_constraint, 0, n_constraint * sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_constraint_rhs, n_constraint * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMemcpy(d_constraint_rhs, rhs.data(), n_constraint * sizeof(Real), cudaMemcpyHostToDevice));

        // J (CSR): rows=constraints, cols=dofs.
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_offsets, static_cast<size_t>(n_constraint + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_nnz, sizeof(int)));

        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_offsets, j_offsets.data(),
                                  static_cast<size_t>(n_constraint + 1) * sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_columns, j_columns.data(), static_cast<size_t>(nnz) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_values, j_values.data(), static_cast<size_t>(nnz) * sizeof(Real),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
        is_j_csr_setup = true;

        // Build J^T (CSR) on host: rows=dofs, cols=constraints.
        std::vector<int> jt_offsets(static_cast<size_t>(n_dofs + 1), 0);
        std::vector<int> jt_columns(static_cast<size_t>(nnz), 0);
        std::vector<Real> jt_values(static_cast<size_t>(nnz), 0.0);

        std::vector<int> counts(static_cast<size_t>(n_dofs), 0);
        for (int idx = 0; idx < nnz; ++idx) {
            counts[static_cast<size_t>(j_columns[static_cast<size_t>(idx)])] += 1;
        }
        int running = 0;
        for (int i = 0; i < n_dofs; ++i) {
            jt_offsets[static_cast<size_t>(i)] = running;
            running += counts[static_cast<size_t>(i)];
        }
        jt_offsets[static_cast<size_t>(n_dofs)] = nnz;

        std::vector<int> positions = jt_offsets;
        for (int row = 0; row < n_constraint; ++row) {
            const int start = j_offsets[static_cast<size_t>(row)];
            const int end = j_offsets[static_cast<size_t>(row + 1)];
            for (int idx = start; idx < end; ++idx) {
                const int col = j_columns[static_cast<size_t>(idx)];
                const int out = positions[static_cast<size_t>(col)]++;
                jt_columns[static_cast<size_t>(out)] = row;
                jt_values[static_cast<size_t>(out)] = j_values[static_cast<size_t>(idx)];
            }
        }

        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_offsets, static_cast<size_t>(n_dofs + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_nnz, sizeof(int)));

        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_offsets, jt_offsets.data(), static_cast<size_t>(n_dofs + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_columns, jt_columns.data(), static_cast<size_t>(nnz) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_values, jt_values.data(), static_cast<size_t>(nnz) * sizeof(Real),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
        is_cj_csr_setup = true;

        is_constraints_setup = true;
        if (d_data) {
            MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));
        }
    }

    // Free memory
    void Destroy() {
        MOPHI_GPU_CALL(cudaFree(d_B_inv));
        MOPHI_GPU_CALL(cudaFree(d_grad_N_ref));
        MOPHI_GPU_CALL(cudaFree(d_detJ_ref));

        MOPHI_GPU_CALL(cudaFree(d_gauss_xi_m));
        MOPHI_GPU_CALL(cudaFree(d_gauss_xi));
        MOPHI_GPU_CALL(cudaFree(d_gauss_eta));
        MOPHI_GPU_CALL(cudaFree(d_gauss_zeta));
        MOPHI_GPU_CALL(cudaFree(d_weight_xi_m));
        MOPHI_GPU_CALL(cudaFree(d_weight_xi));
        MOPHI_GPU_CALL(cudaFree(d_weight_eta));
        MOPHI_GPU_CALL(cudaFree(d_weight_zeta));

        MOPHI_GPU_CALL(cudaFree(d_x12_jac));
        MOPHI_GPU_CALL(cudaFree(d_y12_jac));
        MOPHI_GPU_CALL(cudaFree(d_z12_jac));
        MOPHI_GPU_CALL(cudaFree(d_x12));
        MOPHI_GPU_CALL(cudaFree(d_y12));
        MOPHI_GPU_CALL(cudaFree(d_z12));

        MOPHI_GPU_CALL(cudaFree(d_element_connectivity));

        if (is_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_nnz));
        }

        if (is_cj_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_cj_nnz));
            is_cj_csr_setup = false;
        }

        if (is_j_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_j_nnz));
            is_j_csr_setup = false;
        }

        MOPHI_GPU_CALL(cudaFree(d_F));
        MOPHI_GPU_CALL(cudaFree(d_P));
        MOPHI_GPU_CALL(cudaFree(d_Fdot));
        MOPHI_GPU_CALL(cudaFree(d_P_vis));
        MOPHI_GPU_CALL(cudaFree(d_f_int));
        MOPHI_GPU_CALL(cudaFree(d_f_ext));
        MOPHI_GPU_CALL(cudaFree(d_eta_damp));
        MOPHI_GPU_CALL(cudaFree(d_lambda_damp));

        MOPHI_GPU_CALL(cudaFree(d_H));
        MOPHI_GPU_CALL(cudaFree(d_W));
        MOPHI_GPU_CALL(cudaFree(d_L));

        MOPHI_GPU_CALL(cudaFree(d_rho0));
        MOPHI_GPU_CALL(cudaFree(d_nu));
        MOPHI_GPU_CALL(cudaFree(d_E));
        MOPHI_GPU_CALL(cudaFree(d_lambda));
        MOPHI_GPU_CALL(cudaFree(d_mu));
        MOPHI_GPU_CALL(cudaFree(d_material_model));
        MOPHI_GPU_CALL(cudaFree(d_mu10));
        MOPHI_GPU_CALL(cudaFree(d_mu01));
        MOPHI_GPU_CALL(cudaFree(d_kappa));

        if (is_constraints_setup) {
            MOPHI_GPU_CALL(cudaFree(d_constraint));
            if (d_fixed_nodes) {
                MOPHI_GPU_CALL(cudaFree(d_fixed_nodes));
                d_fixed_nodes = nullptr;
            }
            if (d_constraint_rhs) {
                MOPHI_GPU_CALL(cudaFree(d_constraint_rhs));
                d_constraint_rhs = nullptr;
            }
        }

        MOPHI_GPU_CALL(cudaFree(d_data));
    }

    void CalcDsDuPre();

    void CalcMassMatrix();

    void BuildMassCSRPattern();

    void ConvertToCSR_ConstraintJacT();

    void BuildConstraintJacobianTransposeCSR() {
        ConvertToCSR_ConstraintJacT();
    }

    void ConvertToCSR_ConstraintJac();

    void BuildConstraintJacobianCSR() {
        ConvertToCSR_ConstraintJac();
    }

    void CalcP();

    void CalcInternalForce();

    void CalcConstraintData() override;

    void PrintDsDuPre();

    void RetrieveConnectivityToCPU(MatrixXi& connectivity);

    void RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ);

    void RetrieveMassCSRToCPU(std::vector<int>& offsets, std::vector<int>& columns, std::vector<Real>& values);

    void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>& deformation_gradient);

    void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F);

    void RetrieveInternalForceToCPU(VectorXR& internal_force);

    void RetrieveConstraintDataToCPU(VectorXR& constraint);

    void RetrieveConstraintJacobianToCPU(MatrixXR& constraint_jac);

    void RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12);

    Real* Get_Constraint_Ptr() {
        return d_constraint;
    }

    bool Get_Is_Constraint_Setup() {
        return is_constraints_setup;
    }

    int GetConstraintMode() const {
        return constraint_mode;
    }

    void RetrieveConstraintJacobianCSRToCPU(std::vector<int>& offsets,
                                            std::vector<int>& columns,
                                            std::vector<Real>& values);

    GPU_ANCF3243_Data* d_data;  // Storing GPU copy of SAPGPUData

    int n_nodes;
    int n_elements;
    int n_beam;
    int n_coef;
    int n_constraint;

  private:
    Real* d_B_inv;
    Real* d_grad_N_ref;  // (n_beam, N_QP, N_SHAPE, 3)
    Real* d_detJ_ref;    // (n_beam, N_QP)
    Real *d_gauss_xi_m, *d_gauss_xi, *d_gauss_eta, *d_gauss_zeta;
    Real *d_weight_xi_m, *d_weight_xi, *d_weight_eta, *d_weight_zeta;

    Real *d_x12_jac, *d_y12_jac, *d_z12_jac;
    Real *d_x12, *d_y12, *d_z12;

    int* d_element_connectivity;  // n_elements × 2 array of node IDs
    int *d_csr_offsets, *d_csr_columns;
    Real* d_csr_values;
    int* d_nnz;

    Real *d_F, *d_P;
    // Kelvin-Voigt damping related device buffers
    Real* d_Fdot;         // time derivative of F (per element, per qp, 3x3)
    Real* d_P_vis;        // viscous Piola (per element, per qp, 3x3)
    Real* d_eta_damp;     // per-element (or global) viscous coefficient
    Real* d_lambda_damp;  // per-element (or global) second viscous coeff

    Real *d_H, *d_W, *d_L;

    Real *d_rho0, *d_nu, *d_E, *d_lambda, *d_mu;
    int* d_material_model;
    Real *d_mu10, *d_mu01, *d_kappa;

    Real* d_constraint;
    int* d_fixed_nodes;
    Real* d_constraint_rhs = nullptr;

    // Constraint Jacobian J^T in CSR format
    int* d_cj_csr_offsets;
    int* d_cj_csr_columns;
    Real* d_cj_csr_values;
    int* d_cj_nnz;

    // Constraint Jacobian J in CSR format
    int* d_j_csr_offsets;
    int* d_j_csr_columns;
    Real* d_j_csr_values;
    int* d_j_nnz;

    // force related parameters
    Real *d_f_int, *d_f_ext;

    bool is_setup = false;
    bool is_constraints_setup = false;
    bool is_csr_setup = false;
    bool is_cj_csr_setup = false;
    bool is_j_csr_setup = false;
    bool is_reference_precomputed = false;
    int constraint_mode = kConstraintNone;
};

}  // namespace tlfea
