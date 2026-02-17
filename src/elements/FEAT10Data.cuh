#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "../materials/MaterialModel.cuh"
#include "ElementBase.h"
#include <MoPhiEssentials.h>

// Definition of GPU_ANCF3443 and data access device functions
#pragma once

//
// define a SAP data strucutre
struct GPU_FEAT10_Data : public ElementBase {
#if defined(__CUDACC__)

    // Helper: gather 16 DOFs for an element using connectivity
    __device__ void gather_element_dofs(const double* global,
                                        Eigen::Map<Eigen::MatrixXi> connectivity,
                                        int elem,
                                        double* local) const {
        // Each element has 4 nodes, each node has 4 DOFs
        for (int n = 0; n < 4; ++n) {
            int node = connectivity(elem, n);
    #pragma unroll
            for (int d = 0; d < 4; ++d) {
                local[n * 4 + d] = global[node * 4 + d];
            }
        }
    }

    __device__ Eigen::Map<Eigen::MatrixXi> element_connectivity() const {
        return Eigen::Map<Eigen::MatrixXi>(d_element_connectivity, n_elem, Quadrature::N_NODE_T10_10);
    }

    __device__ Eigen::Map<Eigen::MatrixXR> grad_N_ref(int elem_idx, int qp_idx) {
        return Eigen::Map<Eigen::MatrixXR>(d_grad_N_ref + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3, 10, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXR> grad_N_ref(int elem_idx, int qp_idx) const {
        return Eigen::Map<Eigen::MatrixXR>(d_grad_N_ref + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3, 10, 3);
    }

    __device__ double& detJ_ref(int elem_idx, int qp_idx) {
        return d_detJ_ref[elem_idx * Quadrature::N_QP_T10_5 + qp_idx];
    }

    __device__ double detJ_ref(int elem_idx, int qp_idx) const {
        return d_detJ_ref[elem_idx * Quadrature::N_QP_T10_5 + qp_idx];
    }

    __device__ double tet5pt_x(int qp_idx) {
        return d_tet5pt_x[qp_idx];
    }

    __device__ double tet5pt_y(int qp_idx) {
        return d_tet5pt_y[qp_idx];
    }

    __device__ double tet5pt_z(int qp_idx) {
        return d_tet5pt_z[qp_idx];
    }

    __device__ double tet5pt_weights(int qp_idx) {
        return d_tet5pt_weights[qp_idx];
    }

    __device__ Eigen::Map<Eigen::VectorXR> x12() {
        return Eigen::Map<Eigen::VectorXR>(d_h_x12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const x12() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_x12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> y12() {
        return Eigen::Map<Eigen::VectorXR>(d_h_y12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const y12() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_y12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> z12() {
        return Eigen::Map<Eigen::VectorXR>(d_h_z12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const z12() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_z12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const x12_jac() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_x12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const y12_jac() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_y12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXR> const z12_jac() const {
        return Eigen::Map<Eigen::VectorXR>(d_h_z12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::MatrixXR> F(int elem_idx, int qp_idx) {
        return Eigen::Map<Eigen::MatrixXR>(d_F + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXR> F(int elem_idx, int qp_idx) const {
        return Eigen::Map<Eigen::MatrixXR>(d_F + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ Eigen::Map<Eigen::MatrixXR> P(int elem_idx, int qp_idx) {
        return Eigen::Map<Eigen::MatrixXR>(d_P + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXR> P(int elem_idx, int qp_idx) const {
        return Eigen::Map<Eigen::MatrixXR>(d_P + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    // Time-derivative of deformation gradient (viscous computation)
    __device__ Eigen::Map<Eigen::MatrixXR> Fdot(int elem_idx, int qp_idx) {
        return Eigen::Map<Eigen::MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXR> Fdot(int elem_idx, int qp_idx) const {
        return Eigen::Map<Eigen::MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    // Viscous Piola stress storage
    __device__ Eigen::Map<Eigen::MatrixXR> P_vis(int elem_idx, int qp_idx) {
        return Eigen::Map<Eigen::MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXR> P_vis(int elem_idx, int qp_idx) const {
        return Eigen::Map<Eigen::MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
    }

    __device__ Eigen::Map<Eigen::VectorXR> f_int(int global_node_idx) {
        return Eigen::Map<Eigen::VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXR> f_int(int global_node_idx) const {
        return Eigen::Map<Eigen::VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ Eigen::Map<Eigen::VectorXR> f_int() {
        return Eigen::Map<Eigen::VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXR> f_int() const {
        return Eigen::Map<Eigen::VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ Eigen::Map<Eigen::VectorXR> f_ext(int global_node_idx) {
        return Eigen::Map<Eigen::VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXR> f_ext(int global_node_idx) const {
        return Eigen::Map<Eigen::VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ Eigen::Map<Eigen::VectorXR> f_ext() {
        return Eigen::Map<Eigen::VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXR> f_ext() const {
        return Eigen::Map<Eigen::VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ Eigen::Map<Eigen::VectorXR> constraint() {
        return Eigen::Map<Eigen::VectorXR>(d_constraint, n_constraint);
    }

    __device__ const Eigen::Map<Eigen::VectorXR> constraint() const {
        return Eigen::Map<Eigen::VectorXR>(d_constraint, n_constraint);
    }

    __device__ Eigen::Map<Eigen::VectorXi> fixed_nodes() {
        return Eigen::Map<Eigen::VectorXi>(d_fixed_nodes, n_constraint / 3);
    }

    // ================================
    __device__ double rho0() const {
        return *d_rho0;
    }

    __device__ double nu() const {
        return *d_nu;
    }

    __device__ double E() const {
        return *d_E;
    }

    __device__ double lambda() const {
        return *d_lambda;
    }

    __device__ double eta_damp() const {
        return *d_eta_damp;
    }

    __device__ double lambda_damp() const {
        return *d_lambda_damp;
    }

    __device__ double mu() const {
        return *d_mu;
    }

    __device__ int material_model() const {
        return *d_material_model;
    }

    __device__ double mu10() const {
        return *d_mu10;
    }

    __device__ double mu01() const {
        return *d_mu01;
    }

    __device__ double kappa() const {
        return *d_kappa;
    }

    __device__ int gpu_n_elem() const {
        return n_elem;
    }

    __device__ int gpu_n_coef() const {
        return n_coef;
    }

    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }

    // ======================================================

    __device__ int* csr_offsets() {
        return d_csr_offsets;
    }

    __device__ int* csr_columns() {
        return d_csr_columns;
    }

    __device__ double* csr_values() {
        return d_csr_values;
    }

    __device__ int* cj_csr_offsets() {
        return d_cj_csr_offsets;
    }

    __device__ int* cj_csr_columns() {
        return d_cj_csr_columns;
    }

    __device__ double* cj_csr_values() {
        return d_cj_csr_values;
    }

    __device__ int* j_csr_offsets() {
        return d_j_csr_offsets;
    }

    __device__ int* j_csr_columns() {
        return d_j_csr_columns;
    }

    __device__ double* j_csr_values() {
        return d_j_csr_values;
    }

    __device__ int nnz() {
        return *d_nnz;
    }
#endif

    __host__ __device__ int get_n_elem() const {
        return n_elem;
    }
    __host__ __device__ int get_n_coef() const {
        return n_coef;
    }
    __host__ __device__ int get_n_constraint() const {
        return n_constraint;
    }

    __host__ __device__ int get_n_beam() const override {
        return n_elem;
    }

    void CalcDnDuPre();

    void CalcMassMatrix() override;

    void BuildMassCSRPattern();

    void ConvertToCSR_ConstraintJacT();

    void BuildConstraintJacobianTransposeCSR() {
        ConvertToCSR_ConstraintJacT();
    }

    void ConvertToCSR_ConstraintJac();

    void BuildConstraintJacobianCSR() {
        ConvertToCSR_ConstraintJac();
    }

    void CalcInternalForce() override;

    void CalcConstraintData() override;

    void CalcP() override;

    void RetrieveMassCSRToCPU(std::vector<int>& offsets, std::vector<int>& columns, std::vector<double>& values);

    void RetrieveInternalForceToCPU(Eigen::VectorXR& internal_force) override;

    void RetrieveExternalForceToCPU(Eigen::VectorXR& external_force);

    void RetrieveConstraintDataToCPU(Eigen::VectorXR& constraint) override {}

    void RetrieveConstraintJacobianToCPU(Eigen::MatrixXR& constraint_jac) override {}

    void RetrievePositionToCPU(Eigen::VectorXR& x12, Eigen::VectorXR& y12, Eigen::VectorXR& z12) override;

    void RetrieveDeformationGradientToCPU(std::vector<std::vector<Eigen::MatrixXR>>& deformation_gradient) override {}

    void RetrievePFromFToCPU(std::vector<std::vector<Eigen::MatrixXR>>& p_from_F) override;

    void RetrieveDnDuPreToCPU(std::vector<std::vector<Eigen::MatrixXR>>& dn_du_pre);

    void RetrieveDetJToCPU(std::vector<std::vector<double>>& detJ);

    void RetrieveConnectivityToCPU(Eigen::MatrixXi& connectivity);

    void WriteOutputVTK(const std::string& filename);

    // Constructor
    GPU_FEAT10_Data(int num_elements, int num_nodes) : n_elem(num_elements), n_coef(num_nodes) {
        type = TYPE_T10;
    }

    void Initialize() {
        MOPHI_GPU_CALL(cudaMalloc(&d_h_x12, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_h_y12, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_h_z12, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_h_x12_jac, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_h_y12_jac, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_h_z12_jac, n_coef * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_element_connectivity, n_elem * Quadrature::N_NODE_T10_10 * sizeof(int)));

        MOPHI_GPU_CALL(cudaMalloc(&d_tet5pt_x, Quadrature::N_QP_T10_5 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_tet5pt_y, Quadrature::N_QP_T10_5 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_tet5pt_z, Quadrature::N_QP_T10_5 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_tet5pt_weights, Quadrature::N_QP_T10_5 * sizeof(double)));

        MOPHI_GPU_CALL(cudaMalloc(&d_grad_N_ref, n_elem * Quadrature::N_QP_T10_5 * 10 * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_detJ_ref, n_elem * Quadrature::N_QP_T10_5 * sizeof(double)));

        MOPHI_GPU_CALL(cudaMalloc(&d_F, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_P, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
        // Viscous-related buffers
        MOPHI_GPU_CALL(cudaMalloc(&d_Fdot, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_P_vis, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_f_int, n_coef * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_f_ext, n_coef * 3 * sizeof(double)));

        MOPHI_GPU_CALL(cudaMalloc(&d_rho0, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_nu, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_E, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_material_model, sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu10, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu01, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_kappa, sizeof(double)));
        // damping parameters
        MOPHI_GPU_CALL(cudaMalloc(&d_eta_damp, sizeof(double)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda_damp, sizeof(double)));

        //     // copy struct to device
        MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_FEAT10_Data)));
    }

    void Setup(const Eigen::VectorXR& tet5pt_x_host,
               const Eigen::VectorXR& tet5pt_y_host,
               const Eigen::VectorXR& tet5pt_z_host,
               const Eigen::VectorXR& tet5pt_weights_host,
               const Eigen::VectorXR& h_x12,
               const Eigen::VectorXR& h_y12,
               const Eigen::VectorXR& h_z12,
               const Eigen::MatrixXi& element_connectivity) {
        if (is_setup) {
            MOPHI_ERROR(std::string("GPU_FEAT10_Data is already set up."));
            return;
        }

        MOPHI_GPU_CALL(cudaMemcpy(d_h_x12, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_y12, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_z12, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_x12_jac, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_y12_jac, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_z12_jac, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_element_connectivity, element_connectivity.data(),
                                  n_elem * Quadrature::N_NODE_T10_10 * sizeof(int), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_tet5pt_x, tet5pt_x_host.data(), Quadrature::N_QP_T10_5 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_tet5pt_y, tet5pt_y_host.data(), Quadrature::N_QP_T10_5 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_tet5pt_z, tet5pt_z_host.data(), Quadrature::N_QP_T10_5 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_tet5pt_weights, tet5pt_weights_host.data(), Quadrature::N_QP_T10_5 * sizeof(double),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemset(d_grad_N_ref, 0, n_elem * Quadrature::N_QP_T10_5 * 10 * 3 * sizeof(double)));
        MOPHI_GPU_CALL(cudaMemset(d_detJ_ref, 0, n_elem * Quadrature::N_QP_T10_5 * sizeof(double)));

        cudaMemset(d_f_int, 0, n_coef * 3 * sizeof(double));

        cudaMemset(d_F, 0, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
        cudaMemset(d_P, 0, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
        // initialize viscous buffers to zero
        cudaMemset(d_Fdot, 0, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
        cudaMemset(d_P_vis, 0, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));

        double rho0 = 0.0;
        double nu = 0.0;
        double E = 0.0;
        // Compute material constants
        double mu = E / (2 * (1 + nu));                        // Shear modulus μ
        double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé's first parameter λ
        double eta_damp = 0.0;
        double lambda_damp = 0.0;
        int material_model = MATERIAL_MODEL_SVK;
        double mu10 = 0.0;
        double mu01 = 0.0;
        double kappa = 0.0;

        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));
        // copy damping parameters
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));

        is_setup = true;
    }

    /**
     * Set reference density (used for mass/inertial terms).
     */
    void SetDensity(double rho0) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT10_Data must be set up before setting density.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
    }

    /**
     * Set Kelvin-Voigt damping parameters.
     * eta_damp: shear-like damping coefficient
     * lambda_damp: volumetric-like damping coefficient
     */
    void SetDamping(double eta_damp, double lambda_damp) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT10_Data must be set up before setting damping.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(double), cudaMemcpyHostToDevice));
    }

    /**
     * Select Saint Venant-Kirchhoff (SVK) material model using current E/nu.
     */
    void SetSVK() {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT10_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_SVK;
        double mu10 = 0.0;
        double mu01 = 0.0;
        double kappa = 0.0;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));
    }

    /**
     * Set Saint Venant-Kirchhoff (SVK) parameters.
     * E: Young's modulus
     * nu: Poisson's ratio
     */
    void SetSVK(double E, double nu) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT10_Data must be set up before setting material.");
            return;
        }

        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));

        double mu = E / (2 * (1 + nu));
        double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));

        SetSVK();
    }

    /**
     * Set compressible Mooney-Rivlin parameters.
     * mu10, mu01: isochoric Mooney-Rivlin coefficients
     * kappa: volumetric penalty (bulk-modulus-like) coefficient
     */
    void SetMooneyRivlin(double mu10, double mu01, double kappa) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT10_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));
    }

    void SetExternalForce(const Eigen::VectorXR& h_f_ext) {
        if (h_f_ext.size() != n_coef * 3) {
            MOPHI_ERROR("External force vector size mismatch.");
            return;
        }

        cudaMemset(d_f_ext, 0, n_coef * 3 * sizeof(double));
        MOPHI_GPU_CALL(cudaMemcpy(d_f_ext, h_f_ext.data(), n_coef * 3 * sizeof(double), cudaMemcpyHostToDevice));
    }

    const double* GetX12DevicePtr() const {
        return d_h_x12;
    }

    const double* GetY12DevicePtr() const {
        return d_h_y12;
    }

    const double* GetZ12DevicePtr() const {
        return d_h_z12;
    }

    double* GetExternalForceDevicePtr() {
        return d_f_ext;
    }

    const double* GetExternalForceDevicePtr() const {
        return d_f_ext;
    }

    /**
     * Update node positions on GPU (for prescribed motion of fixed nodes).
     */
    void UpdatePositions(const Eigen::VectorXR& h_x12, const Eigen::VectorXR& h_y12, const Eigen::VectorXR& h_z12) {
        if (h_x12.size() != n_coef || h_y12.size() != n_coef || h_z12.size() != n_coef) {
            MOPHI_ERROR("Position vector size mismatch.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_h_x12, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_y12, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_z12, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
    }

    void UpdateConstraintTargets(const Eigen::VectorXR& h_x12,
                                 const Eigen::VectorXR& h_y12,
                                 const Eigen::VectorXR& h_z12) {
        if (h_x12.size() != n_coef || h_y12.size() != n_coef || h_z12.size() != n_coef) {
            MOPHI_ERROR("Position vector size mismatch.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_h_x12_jac, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_y12_jac, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_h_z12_jac, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
    }

    void SetNodalFixed(const Eigen::VectorXi& fixed_nodes);

    /**
     * Update fixed nodes for dynamic constraint changes (e.g., moving grippers).
     * This reuses existing constraint buffers if the number of fixed nodes
     * matches, otherwise reallocates. After calling this, you must call
     * CalcConstraintData() and rebuild constraint Jacobians (CSR) if needed.
     */
    void UpdateNodalFixed(const Eigen::VectorXi& fixed_nodes);

    // Free memory
    void Destroy() {
        MOPHI_GPU_CALL(cudaFree(d_h_x12));
        MOPHI_GPU_CALL(cudaFree(d_h_y12));
        MOPHI_GPU_CALL(cudaFree(d_h_z12));

        MOPHI_GPU_CALL(cudaFree(d_h_x12_jac));
        MOPHI_GPU_CALL(cudaFree(d_h_y12_jac));
        MOPHI_GPU_CALL(cudaFree(d_h_z12_jac));

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
        }

        if (is_j_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_j_nnz));
        }

        MOPHI_GPU_CALL(cudaFree(d_tet5pt_x));
        MOPHI_GPU_CALL(cudaFree(d_tet5pt_y));
        MOPHI_GPU_CALL(cudaFree(d_tet5pt_z));
        MOPHI_GPU_CALL(cudaFree(d_tet5pt_weights));

        MOPHI_GPU_CALL(cudaFree(d_grad_N_ref));
        MOPHI_GPU_CALL(cudaFree(d_detJ_ref));

        MOPHI_GPU_CALL(cudaFree(d_F));
        MOPHI_GPU_CALL(cudaFree(d_P));
        MOPHI_GPU_CALL(cudaFree(d_Fdot));
        MOPHI_GPU_CALL(cudaFree(d_P_vis));
        MOPHI_GPU_CALL(cudaFree(d_f_int));
        MOPHI_GPU_CALL(cudaFree(d_f_ext));

        MOPHI_GPU_CALL(cudaFree(d_rho0));
        MOPHI_GPU_CALL(cudaFree(d_nu));
        MOPHI_GPU_CALL(cudaFree(d_E));
        MOPHI_GPU_CALL(cudaFree(d_lambda));
        MOPHI_GPU_CALL(cudaFree(d_mu));
        MOPHI_GPU_CALL(cudaFree(d_material_model));
        MOPHI_GPU_CALL(cudaFree(d_mu10));
        MOPHI_GPU_CALL(cudaFree(d_mu01));
        MOPHI_GPU_CALL(cudaFree(d_kappa));
        MOPHI_GPU_CALL(cudaFree(d_eta_damp));
        MOPHI_GPU_CALL(cudaFree(d_lambda_damp));

        MOPHI_GPU_CALL(cudaFree(d_data));

        if (is_constraints_setup) {
            MOPHI_GPU_CALL(cudaFree(d_constraint));
            MOPHI_GPU_CALL(cudaFree(d_fixed_nodes));
        }
    }

    double* Get_Constraint_Ptr() {
        return d_constraint;
    }

    bool Get_Is_Constraint_Setup() {
        return is_constraints_setup;
    }

    GPU_FEAT10_Data* d_data;  // Storing GPU copy of SAPGPUData

    int n_elem;
    int n_coef;
    int n_constraint;

  private:
    // Node positions (global, or per element)
    double *d_h_x12, *d_h_y12, *d_h_z12;  // (n_coef, 1)
    double *d_h_x12_jac, *d_h_y12_jac, *d_h_z12_jac;

    // Element connectivity
    int* d_element_connectivity;  // (n_elem, 10)

    // Mass Matrix
    // Mass Matrix in CSR format
    int *d_csr_offsets, *d_csr_columns;
    double* d_csr_values;
    int* d_nnz;

    // Quadrature points and weights
    double *d_tet5pt_x, *d_tet5pt_y, *d_tet5pt_z;
    double* d_tet5pt_weights;  // (5,)

    // Precomputed reference gradients
    double* d_grad_N_ref;  // (n_elem, 5, 10, 3)
    double* d_detJ_ref;    // (n_elem, 5)

    // Deformation gradient and Piola stress
    double* d_F;  // (n_elem, n_qp, 3, 3)
    double* d_P;  // (n_elem, n_qp, 3, 3)
    // Time-derivative of deformation gradient and viscous Piola
    double* d_Fdot;   // (n_elem, n_qp, 3, 3)
    double* d_P_vis;  // (n_elem, n_qp, 3, 3)

    // Material properties
    double *d_E, *d_nu, *d_rho0, *d_lambda, *d_mu;
    int* d_material_model;
    double *d_mu10, *d_mu01, *d_kappa;
    // Damping parameters
    double *d_eta_damp, *d_lambda_damp;

    // Constraint data
    double* d_constraint;
    int* d_fixed_nodes;
    // Constraint Jacobian J^T in CSR format
    int *d_cj_csr_offsets, *d_cj_csr_columns;
    double* d_cj_csr_values;
    int* d_cj_nnz;

    // Constraint Jacobian J in CSR format
    int *d_j_csr_offsets, *d_j_csr_columns;
    double* d_j_csr_values;
    int* d_j_nnz;

    // Force vectors
    double *d_f_int, *d_f_ext;  // (n_nodes*3)

    bool is_setup = false;
    bool is_constraints_setup = false;
    bool is_csr_setup = false;
    bool is_cj_csr_setup = false;
    bool is_j_csr_setup = false;
};
