/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LinearStaticSolver.cu
 * Brief:   Implements the GPU-side steady-state (linear) FEA solver for
 *          FEAT10 TET10 elements.
 *
 *          Key stages performed inside Solve():
 *            1.  BuildStiffnessCSRPattern  – constructs the 3N×3N sparsity
 *                pattern once using thrust sort + unique on (row,col) keys
 *                generated from element connectivity.
 *            2.  AssembleLinearStiffness   – fills K values via the existing
 *                compute_hessian_assemble_csr device function (SVK tangent at
 *                the reference configuration where F = I gives the classical
 *                linear-elastic stiffness).
 *            3.  ApplyDirichletBCs         – enforces fixed DOFs by row/column
 *                elimination (identity rows, zeroed symmetric columns, zero
 *                RHS entries).
 *            4.  SolveLinearSystemCG       – CG loop using cuSPARSE SpMV and
 *                cuBLAS dot / axpy / scal.
 *            5.  UpdatePositions           – writes u back into the
 *                GPU_FEAT10_Data node-position arrays.
 *==============================================================
 *==============================================================*/

#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdio>

#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../utils/cuda_utils.h"
#include "LinearStaticSolver.cuh"

// cuBLAS error checking (analogous to CHECK_CUSPARSE in cuda_utils.h)
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(func)                                                                           \
    {                                                                                                \
        cublasStatus_t status = (func);                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                       \
            MOPHI_ERROR("cuBLAS API failed with error code %d at %s:%d", status, __FILE__, __LINE__); \
        }                                                                                            \
    }
#endif

namespace tlfea {

// ---------------------------------------------------------------------------
// Helper: set the last entry of an offset array (same helper used in
// FEAT10Data.cu).
// ---------------------------------------------------------------------------
namespace {
__global__ void ls_set_last_offset_kernel(int* d_offsets, int n_rows, int nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_offsets[n_rows] = nnz;
    }
}
}  // namespace

// Number of (node-pair × dof-pair) contributions per TET10 element:
//   10 nodes × 10 nodes × 3 DOFs/node × 3 DOFs/node
static constexpr int STIFFNESS_KEYS_PER_ELEM = 10 * 10 * 3 * 3;

// ---------------------------------------------------------------------------
// Pattern building – step 1: generate raw (row<<32|col) keys.
//
// For each element e, each pair of local nodes (i, j) and each pair of
// DOF indices (d, e_dof) contributes one entry:
//   global_row = 3*global_node_i + d
//   global_col = 3*global_node_j + e_dof
// ---------------------------------------------------------------------------
__global__ void build_stiffness_keys_kernel(GPU_FEAT10_Data* d_data, unsigned long long* d_keys) {
    const int total = d_data->gpu_n_elem() * STIFFNESS_KEYS_PER_ELEM;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int elem      = tid / STIFFNESS_KEYS_PER_ELEM;
    const int rem       = tid % STIFFNESS_KEYS_PER_ELEM;
    const int node_pair = rem / 9;  // 0..99 (10 × 10 node combinations)
    const int dof_pair  = rem % 9;  // 0..8  (3 × 3 DOF combinations)

    const int i_local = node_pair / 10;  // 0..9
    const int j_local = node_pair % 10;  // 0..9
    const int dof_d   = dof_pair / 3;    // 0..2
    const int dof_e   = dof_pair % 3;    // 0..2

    const int global_i = d_data->element_connectivity()(elem, i_local);
    const int global_j = d_data->element_connectivity()(elem, j_local);

    const int row = 3 * global_i + dof_d;
    const int col = 3 * global_j + dof_e;

    d_keys[tid] = (static_cast<unsigned long long>(static_cast<unsigned int>(row)) << 32) |
                  static_cast<unsigned long long>(static_cast<unsigned int>(col));
}

// ---------------------------------------------------------------------------
// Pattern building – step 2: decode unique keys into CSR columns + row counts.
// ---------------------------------------------------------------------------
__global__ void decode_stiffness_keys_kernel(const unsigned long long* d_keys,
                                             int                       nnz,
                                             int*                      d_K_columns,
                                             int*                      d_row_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    const unsigned long long key = d_keys[tid];
    const int                row = static_cast<int>(key >> 32);
    const int                col = static_cast<int>(key & 0xffffffffULL);

    d_K_columns[tid] = col;
    atomicAdd(d_row_counts + row, 1);
}

// ---------------------------------------------------------------------------
// Stiffness assembly – one thread per (element, quadrature-point) pair.
//
// compute_hessian_assemble_csr<GPU_FEAT10_Data> is a __device__ __forceinline__
// function from FEAT10DataFunc.cuh. It uses the SVK tangent and, when called
// at the reference configuration (current positions == reference positions,
// F = I), produces the classical linear-elastic element stiffness matrix.
//
// The SyncedNewtonSolver* parameter is forward-declared but never dereferenced
// inside the FEAT10 specialisation – passing nullptr is safe.
// ---------------------------------------------------------------------------
__global__ void assemble_stiffness_kernel(GPU_FEAT10_Data* d_data,
                                          int*             d_K_offsets,
                                          int*             d_K_columns,
                                          Real*            d_K_values) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int elem = tid / Quadrature::N_QP_T10_5;
    const int qp   = tid % Quadrature::N_QP_T10_5;

    if (elem >= d_data->gpu_n_elem()) return;

    // h = 1.0: no time-step scaling for static assembly.
    compute_hessian_assemble_csr<GPU_FEAT10_Data>(
        d_data, static_cast<SyncedNewtonSolver*>(nullptr), elem, qp,
        d_K_offsets, d_K_columns, d_K_values, 1.0);
}

// ---------------------------------------------------------------------------
// Boundary condition application.
//
// Phase 1: mark which global DOFs are constrained.
// ---------------------------------------------------------------------------
__global__ void mark_fixed_dofs_kernel(GPU_FEAT10_Data* d_data, int* d_is_fixed) {
    const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_fixed   = d_data->gpu_n_constraint() / 3;
    if (tid >= n_fixed) return;

    const int node = d_data->fixed_nodes()(tid);
    d_is_fixed[3 * node + 0] = 1;
    d_is_fixed[3 * node + 1] = 1;
    d_is_fixed[3 * node + 2] = 1;
}

// ---------------------------------------------------------------------------
// BC application – phase 2.
//
// For each row:
//   • If the row belongs to a fixed DOF → replace with identity row, set f=0.
//   • Otherwise                         → zero out any column entry whose
//                                         column index belongs to a fixed DOF
//                                         (keeps the system symmetric; the RHS
//                                         adjustment is trivially zero because
//                                         prescribed displacements are zero).
// ---------------------------------------------------------------------------
__global__ void apply_bcs_kernel(const int* d_is_fixed,
                                 const int* K_offsets,
                                 const int* K_columns,
                                 Real*      K_values,
                                 Real*      f,
                                 int        n_dof) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_dof) return;

    const int row_start = K_offsets[row];
    const int row_end   = K_offsets[row + 1];

    if (d_is_fixed[row]) {
        // Identity row for Dirichlet DOF.
        for (int k = row_start; k < row_end; ++k)
            K_values[k] = (K_columns[k] == row) ? static_cast<Real>(1) : static_cast<Real>(0);
        f[row] = static_cast<Real>(0);
    } else {
        // Free row: zero out columns that correspond to fixed DOFs.
        for (int k = row_start; k < row_end; ++k)
            if (d_is_fixed[K_columns[k]])
                K_values[k] = static_cast<Real>(0);
        // f[row] is already the correct external force – no adjustment needed
        // when all prescribed displacements are zero.
    }
}

// ---------------------------------------------------------------------------
// Position update: x12_i += u_{3i}, y12_i += u_{3i+1}, z12_i += u_{3i+2}.
// ---------------------------------------------------------------------------
__global__ void update_positions_kernel(Real* d_x12, Real* d_y12, Real* d_z12,
                                        const Real* d_u, int n_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    d_x12[i] += d_u[3 * i + 0];
    d_y12[i] += d_u[3 * i + 1];
    d_z12[i] += d_u[3 * i + 2];
}

// ===========================================================================
// LinearStaticSolver – host-side methods
// ===========================================================================

LinearStaticSolver::LinearStaticSolver(GPU_FEAT10_Data* data, Real tol, int max_iter)
    : data_(data),
      n_dof_(3 * data->get_n_coef()),
      cg_tol_(tol),
      cg_max_iter_(max_iter) {
    CHECK_CUSPARSE(cusparseCreate(&cusparse_));
    CHECK_CUBLAS(cublasCreate(&cublas_));

    MOPHI_GPU_CALL(cudaMalloc(&d_u_,  static_cast<size_t>(n_dof_) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_f_,  static_cast<size_t>(n_dof_) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_r_,  static_cast<size_t>(n_dof_) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_p_,  static_cast<size_t>(n_dof_) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_Kp_, static_cast<size_t>(n_dof_) * sizeof(Real)));
}

LinearStaticSolver::~LinearStaticSolver() {
    if (d_K_offsets_) cudaFree(d_K_offsets_);
    if (d_K_columns_) cudaFree(d_K_columns_);
    if (d_K_values_)  cudaFree(d_K_values_);
    if (d_u_)         cudaFree(d_u_);
    if (d_f_)         cudaFree(d_f_);
    if (d_r_)         cudaFree(d_r_);
    if (d_p_)         cudaFree(d_p_);
    if (d_Kp_)        cudaFree(d_Kp_);
    if (cusparse_)    cusparseDestroy(cusparse_);
    if (cublas_)      cublasDestroy(cublas_);
}

// ---------------------------------------------------------------------------
// BuildStiffnessCSRPattern
// ---------------------------------------------------------------------------
void LinearStaticSolver::BuildStiffnessCSRPattern() {
    if (pattern_built_) return;

    const int n_elem = data_->get_n_elem();
    const int total_raw = n_elem * STIFFNESS_KEYS_PER_ELEM;

    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys, static_cast<size_t>(total_raw) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_raw + threads - 1) / threads;
        build_stiffness_keys_kernel<<<blocks, threads>>>(data_->d_data, d_keys);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    // Sort and deduplicate.
    thrust::device_ptr<unsigned long long> keys_begin(d_keys);
    thrust::device_ptr<unsigned long long> keys_end = keys_begin + total_raw;
    thrust::sort(thrust::device, keys_begin, keys_end);
    auto keys_unique_end = thrust::unique(thrust::device, keys_begin, keys_end);
    K_nnz_ = static_cast<int>(keys_unique_end - keys_begin);

    // Allocate CSR arrays.
    MOPHI_GPU_CALL(cudaMalloc(&d_K_offsets_, static_cast<size_t>(n_dof_ + 1) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc(&d_K_columns_, static_cast<size_t>(K_nnz_)     * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc(&d_K_values_,  static_cast<size_t>(K_nnz_)     * sizeof(Real)));

    // Decode unique keys into columns + row counts.
    int* d_row_counts = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_row_counts, static_cast<size_t>(n_dof_) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_row_counts, 0, static_cast<size_t>(n_dof_) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (K_nnz_ + threads - 1) / threads;
        decode_stiffness_keys_kernel<<<blocks, threads>>>(d_keys, K_nnz_, d_K_columns_, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    // Exclusive prefix sum → row offsets.
    thrust::device_ptr<int> row_counts_ptr(d_row_counts);
    thrust::device_ptr<int> offsets_ptr(d_K_offsets_);
    thrust::exclusive_scan(thrust::device, row_counts_ptr, row_counts_ptr + n_dof_, offsets_ptr);
    ls_set_last_offset_kernel<<<1, 1>>>(d_K_offsets_, n_dof_, K_nnz_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    pattern_built_ = true;
}

// ---------------------------------------------------------------------------
// AssembleLinearStiffness
// ---------------------------------------------------------------------------
void LinearStaticSolver::AssembleLinearStiffness() {
    // Zero K values before accumulation.
    MOPHI_GPU_CALL(cudaMemset(d_K_values_, 0, static_cast<size_t>(K_nnz_) * sizeof(Real)));

    const int total_qp = data_->get_n_elem() * Quadrature::N_QP_T10_5;
    constexpr int threads = 128;
    const int blocks = (total_qp + threads - 1) / threads;

    assemble_stiffness_kernel<<<blocks, threads>>>(data_->d_data, d_K_offsets_, d_K_columns_, d_K_values_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// ApplyDirichletBCs
// ---------------------------------------------------------------------------
void LinearStaticSolver::ApplyDirichletBCs() {
    const int n_constraint = data_->get_n_constraint();
    if (n_constraint == 0) return;

    const int n_fixed_nodes = n_constraint / 3;

    // Allocate and zero the is_fixed flag array.
    int* d_is_fixed = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_is_fixed, static_cast<size_t>(n_dof_) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_is_fixed, 0, static_cast<size_t>(n_dof_) * sizeof(int)));

    // Mark fixed DOFs.
    {
        constexpr int threads = 256;
        const int blocks = (n_fixed_nodes + threads - 1) / threads;
        mark_fixed_dofs_kernel<<<blocks, threads>>>(data_->d_data, d_is_fixed);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    // Apply BC row/column modifications to K and the RHS f.
    {
        constexpr int threads = 256;
        const int blocks = (n_dof_ + threads - 1) / threads;
        apply_bcs_kernel<<<blocks, threads>>>(d_is_fixed, d_K_offsets_, d_K_columns_, d_K_values_, d_f_, n_dof_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaFree(d_is_fixed));
}

// ---------------------------------------------------------------------------
// SolveLinearSystemCG – conjugate gradient on GPU.
//
// Uses:
//   cuSPARSE generic SpMV for y = K * x
//   cuBLAS Ddot / Daxpy / Dscal for vector operations (double precision).
// ---------------------------------------------------------------------------
void LinearStaticSolver::SolveLinearSystemCG() {
    const int n = n_dof_;

    // u = 0, r = f, p = r.
    MOPHI_GPU_CALL(cudaMemset(d_u_, 0, static_cast<size_t>(n) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMemcpy(d_r_, d_f_, static_cast<size_t>(n) * sizeof(Real), cudaMemcpyDeviceToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_p_, d_f_, static_cast<size_t>(n) * sizeof(Real), cudaMemcpyDeviceToDevice));

    // Build cuSPARSE descriptors.
    cusparseSpMatDescr_t K_descr  = nullptr;
    cusparseDnVecDescr_t p_descr  = nullptr;
    cusparseDnVecDescr_t Kp_descr = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &K_descr, n, n, K_nnz_,
        d_K_offsets_, d_K_columns_, d_K_values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&p_descr,  n, d_p_,  CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&Kp_descr, n, d_Kp_, CUDA_R_64F));

    const Real one  = 1.0;
    const Real zero = 0.0;

    size_t spMV_bufsize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, K_descr, p_descr, &zero, Kp_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spMV_bufsize));

    void* spMV_buf = nullptr;
    if (spMV_bufsize > 0) {
        MOPHI_GPU_CALL(cudaMalloc(&spMV_buf, spMV_bufsize));
    }

    // Initial residual norm squared.
    Real rr0 = 0.0;
    CHECK_CUBLAS(cublasDdot(cublas_, n, d_r_, 1, d_r_, 1, &rr0));

    // Early exit: zero RHS means solution is u=0.
    if (rr0 == 0.0) {
        last_iter_count_ = 0;
        last_residual_   = 0.0;
        CHECK_CUSPARSE(cusparseDestroySpMat(K_descr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(p_descr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(Kp_descr));
        if (spMV_buf) cudaFree(spMV_buf);
        return;
    }

    Real rr = rr0;

    last_iter_count_ = cg_max_iter_;
    last_residual_   = 1.0;

    for (int iter = 0; iter < cg_max_iter_; ++iter) {
        // Kp = K * p.
        CHECK_CUSPARSE(cusparseSpMV(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, K_descr, p_descr, &zero, Kp_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spMV_buf));

        // alpha = rr / (p · Kp).
        Real pKp = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_p_, 1, d_Kp_, 1, &pKp));

        if (pKp <= 0.0) {
            // Should not happen for a SPD system after proper BC application.
            MOPHI_ERROR("LinearStaticSolver CG: non-positive p·Kp detected; aborting.");
            break;
        }
        Real alpha = rr / pKp;

        // u += alpha * p.
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &alpha, d_p_, 1, d_u_, 1));

        // r -= alpha * Kp.
        const Real neg_alpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &neg_alpha, d_Kp_, 1, d_r_, 1));

        // rr_new = r · r.
        Real rr_new = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_r_, 1, d_r_, 1, &rr_new));

        // Check convergence.
        const Real rel_res = (rr0 > 0.0) ? std::sqrt(rr_new / rr0) : std::sqrt(rr_new);
        if (rel_res < cg_tol_) {
            last_iter_count_ = iter + 1;
            last_residual_   = rel_res;
            break;
        }

        // beta = rr_new / rr.
        const Real beta = rr_new / rr;

        // p = r + beta * p  (cuBLAS: first scale p, then add r).
        CHECK_CUBLAS(cublasDscal(cublas_, n, &beta, d_p_, 1));
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &one,  d_r_, 1, d_p_, 1));

        rr = rr_new;

        if (iter == cg_max_iter_ - 1) {
            last_iter_count_ = cg_max_iter_;
            last_residual_   = rel_res;
        }
    }

    // Clean up descriptors and temporary buffer.
    CHECK_CUSPARSE(cusparseDestroySpMat(K_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(p_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(Kp_descr));
    if (spMV_buf) cudaFree(spMV_buf);
}

// ---------------------------------------------------------------------------
// UpdatePositions – add displacement u to GPU_FEAT10_Data positions.
// ---------------------------------------------------------------------------
void LinearStaticSolver::UpdatePositions() {
    const int n_nodes = data_->get_n_coef();
    constexpr int threads = 256;
    const int blocks = (n_nodes + threads - 1) / threads;

    // Access the raw device pointers through the public getter methods.
    Real* d_x = const_cast<Real*>(data_->GetX12DevicePtr());
    Real* d_y = const_cast<Real*>(data_->GetY12DevicePtr());
    Real* d_z = const_cast<Real*>(data_->GetZ12DevicePtr());

    update_positions_kernel<<<blocks, threads>>>(d_x, d_y, d_z, d_u_, n_nodes);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    // Sync the host-visible struct copy on GPU.
    MOPHI_GPU_CALL(cudaMemcpy(data_->d_data, data_, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Solve – top-level entry point.
// ---------------------------------------------------------------------------
void LinearStaticSolver::Solve() {
    // Build sparsity pattern once.
    BuildStiffnessCSRPattern();

    // Assemble K at current (reference) positions.
    AssembleLinearStiffness();

    // Copy external force vector into the working RHS buffer.
    {
        const Real* d_f_ext = data_->GetExternalForceDevicePtr();
        MOPHI_GPU_CALL(cudaMemcpy(d_f_, d_f_ext,
                                  static_cast<size_t>(n_dof_) * sizeof(Real),
                                  cudaMemcpyDeviceToDevice));
    }

    // Apply Dirichlet boundary conditions.
    ApplyDirichletBCs();

    // Solve K * u = f.
    SolveLinearSystemCG();

    // Write displacement back into GPU_FEAT10_Data.
    UpdatePositions();
}

}  // namespace tlfea
