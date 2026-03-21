/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    NavierStokesSUPGPSPGSolver.cu
 * Brief:   GPU implementation of the transient incompressible
 *          Navier-Stokes solver with SUPG/PSPG stabilisation for TET4.
 *
 *          The solver assembles (per time step) the stabilised Oseen
 *          system and solves it with BiCGSTAB using cuBLAS and cuSPARSE.
 *
 *          Notational conventions used in kernels
 *          ----------------------------------------
 *          n  : node index (0..n_nodes-1)
 *          e  : element index (0..n_elems-1)
 *          a,b: local node indices within an element (0..3)
 *          d,f: velocity direction (0,1,2 = x,y,z)
 *          DOF layout:  DOF = 4*n + d  for velocity direction d
 *                       DOF = 4*n + 3  for pressure
 *==============================================================
 *==============================================================*/

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cmath>
#include <cstdio>

#include "../utils/cuda_utils.h"
#include "NavierStokesSUPGPSPGSolver.cuh"

// --------------------------------------------------------------------------
// cuBLAS error checking macro (mirrors LinearStaticSolver.cu)
// --------------------------------------------------------------------------
#ifndef CHECK_CUBLAS
    #define CHECK_CUBLAS(func)                                                                            \
        {                                                                                                 \
            cublasStatus_t _st = (func);                                                                  \
            if (_st != CUBLAS_STATUS_SUCCESS) {                                                           \
                MOPHI_ERROR("cuBLAS error %d at %s:%d", _st, __FILE__, __LINE__);                        \
            }                                                                                             \
        }
#endif

namespace tlfea {

// ==========================================================================
//  Device utilities
// ==========================================================================
namespace {

// 3×3 matrix inversion using the adjugate formula. Returns determinant.
__device__ __forceinline__ Real mat3_inv(const Real J[3][3], Real Jinv[3][3]) {
    Real c00 = J[1][1]*J[2][2] - J[1][2]*J[2][1];
    Real c01 = J[1][2]*J[2][0] - J[1][0]*J[2][2];
    Real c02 = J[1][0]*J[2][1] - J[1][1]*J[2][0];
    Real c10 = J[0][2]*J[2][1] - J[0][1]*J[2][2];
    Real c11 = J[0][0]*J[2][2] - J[0][2]*J[2][0];
    Real c12 = J[0][1]*J[2][0] - J[0][0]*J[2][1];
    Real c20 = J[0][1]*J[1][2] - J[0][2]*J[1][1];
    Real c21 = J[0][2]*J[1][0] - J[0][0]*J[1][2];
    Real c22 = J[0][0]*J[1][1] - J[0][1]*J[1][0];
    Real det = J[0][0]*c00 + J[0][1]*c01 + J[0][2]*c02;
    Real inv_det = 1.0 / det;
    // Jinv = adj(J)^T / det  (adjugate transpose / det = inverse)
    // cofactor matrix:  Jinv[i][j] = c_ji / det
    Jinv[0][0] = c00 * inv_det;  Jinv[0][1] = c10 * inv_det;  Jinv[0][2] = c20 * inv_det;
    Jinv[1][0] = c01 * inv_det;  Jinv[1][1] = c11 * inv_det;  Jinv[1][2] = c21 * inv_det;
    Jinv[2][0] = c02 * inv_det;  Jinv[2][1] = c12 * inv_det;  Jinv[2][2] = c22 * inv_det;
    return det;
}

// Helper: binary search for value `col` in sorted array [begin, end).
// Returns the index of the match.  Undefined if col is not present.
__device__ __forceinline__ int binary_search_col(const int* begin, int len, int col) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int v   = begin[mid];
        if      (v == col) return mid;
        else if (v <  col) lo = mid + 1;
        else               hi = mid - 1;
    }
    return -1;  // not found (should never happen after pattern build)
}

// Helper: atomically add val to K_values at the CSR position for (row, col).
__device__ __forceinline__ void atomic_add_csr(int*       K_offsets,
                                               int*       K_columns,
                                               Real*      K_values,
                                               int        row,
                                               int        col,
                                               Real       val) {
    if (val == Real(0)) return;
    int row_start = K_offsets[row];
    int row_len   = K_offsets[row + 1] - row_start;
    int pos       = binary_search_col(K_columns + row_start, row_len, col);
    if (pos >= 0)
        atomicAdd(K_values + row_start + pos, val);
}

// Helper: atomically add val to RHS at position dof.
__device__ __forceinline__ void atomic_add_rhs(Real* rhs, int dof, Real val) {
    if (val != Real(0)) atomicAdd(rhs + dof, val);
}

// Set last element of the CSR offsets array.
__global__ void ns_set_last_offset_kernel(int* offsets, int n_rows, int nnz) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        offsets[n_rows] = nnz;
}

}  // anonymous namespace

// ==========================================================================
//  Kernel 1:  Precompute element geometry (grad N, det J, h_elem)
// ==========================================================================
__global__ void ns_precompute_geom_kernel(const Real* __restrict__ nodes,
                                          const int*  __restrict__ connect,
                                          Real*                    gradN,
                                          Real*                    detJ,
                                          Real*                    h_elem,
                                          int                      n_elems) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elems) return;

    // Local node indices
    int n0 = connect[4*e+0];
    int n1 = connect[4*e+1];
    int n2 = connect[4*e+2];
    int n3 = connect[4*e+3];

    // Node coordinates
    Real X0[3] = { nodes[3*n0+0], nodes[3*n0+1], nodes[3*n0+2] };
    Real X1[3] = { nodes[3*n1+0], nodes[3*n1+1], nodes[3*n1+2] };
    Real X2[3] = { nodes[3*n2+0], nodes[3*n2+1], nodes[3*n2+2] };
    Real X3[3] = { nodes[3*n3+0], nodes[3*n3+1], nodes[3*n3+2] };

    // Jacobian: columns are (X1-X0), (X2-X0), (X3-X0)
    Real J[3][3];
    for (int i = 0; i < 3; i++) {
        J[i][0] = X1[i] - X0[i];
        J[i][1] = X2[i] - X0[i];
        J[i][2] = X3[i] - X0[i];
    }

    Real Jinv[3][3];
    Real det = mat3_inv(J, Jinv);
    detJ[e]  = det;   // det J = 6 * V_elem (can be negative depending on orientation)

    // Shape function gradients in reference coords:
    // N0 = 1-ξ-η-ζ  →  ∂N0/∂(ξ,η,ζ) = (-1,-1,-1)
    // N1 = ξ          →  (1,0,0)
    // N2 = η          →  (0,1,0)
    // N3 = ζ          →  (0,0,1)
    Real gref[4][3] = { {-1,-1,-1}, {1,0,0}, {0,1,0}, {0,0,1} };

    // Physical gradients:  ∇N_a = Jinv^T * gref[a]
    //   (Jinv^T)_{ij} = Jinv[j][i]
    for (int a = 0; a < 4; a++) {
        for (int j = 0; j < 3; j++) {
            Real sum = 0;
            for (int k = 0; k < 3; k++)
                sum += Jinv[k][j] * gref[a][k];
            gradN[e*12 + a*3 + j] = sum;
        }
    }

    // Element length scale  h_e = (6 * |V_e|)^{1/3}  =  |det J|^{1/3}
    Real vol = fabs(det) / 6.0;
    h_elem[e] = cbrt(6.0 * vol);   // = cbrt(|det J|)
}

// ==========================================================================
//  Kernel 2:  Build sparsity pattern keys (row<<32|col) for NS system
// ==========================================================================
// Each element contributes 16×16 = 256 (row,col) pairs.
__global__ void ns_build_keys_kernel(const int* __restrict__ connect,
                                     unsigned long long*      keys,
                                     int                      n_elems) {
    constexpr int N_ELEM_DOF   = 16;   // 4 nodes × 4 DOFs
    constexpr int KEYS_PER_ELEM = N_ELEM_DOF * N_ELEM_DOF;  // 256

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_elems * KEYS_PER_ELEM;
    if (tid >= total) return;

    int elem  = tid / KEYS_PER_ELEM;
    int entry = tid % KEYS_PER_ELEM;
    int row_local = entry / N_ELEM_DOF;
    int col_local = entry % N_ELEM_DOF;

    int local_node_r = row_local / 4;
    int dof_r        = row_local % 4;
    int local_node_c = col_local / 4;
    int dof_c        = col_local % 4;

    int global_row = 4 * connect[4*elem + local_node_r] + dof_r;
    int global_col = 4 * connect[4*elem + local_node_c] + dof_c;

    keys[tid] = (static_cast<unsigned long long>(static_cast<unsigned int>(global_row)) << 32)
              |  static_cast<unsigned long long>(static_cast<unsigned int>(global_col));
}

// Decode unique sorted keys into CSR columns + row counts.
__global__ void ns_decode_keys_kernel(const unsigned long long* __restrict__ keys,
                                      int                                    nnz,
                                      int*                                   columns,
                                      int*                                   row_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    unsigned long long key = keys[tid];
    int row = static_cast<int>(key >> 32);
    int col = static_cast<int>(key & 0xffffffffULL);
    columns[tid] = col;
    atomicAdd(row_counts + row, 1);
}

// ==========================================================================
//  Kernel 3:  Assemble NS system matrix + RHS  (one thread per element)
// ==========================================================================
__global__ void ns_assemble_kernel(const int*  __restrict__ connect,
                                   const Real* __restrict__ gradN,
                                   const Real* __restrict__ detJ,
                                   const Real* __restrict__ h_elem,
                                   const Real* __restrict__ vel_prev,  // 3*n_nodes (vx,vy,vz)
                                   int*        K_offsets,
                                   int*        K_columns,
                                   Real*       K_values,
                                   Real*       rhs,
                                   Real        rho,
                                   Real        mu,
                                   Real        inv_dt,
                                   int         n_elems) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elems) return;

    // -------------------------------------------------------------------
    // Local node indices and precomputed geometry
    // -------------------------------------------------------------------
    int nodes[4];
    for (int a = 0; a < 4; a++) nodes[a] = connect[4*e + a];

    Real gN[4][3];  // shape function gradients
    for (int a = 0; a < 4; a++)
        for (int j = 0; j < 3; j++)
            gN[a][j] = gradN[e*12 + a*3 + j];

    Real det = detJ[e];
    Real h   = h_elem[e];

    // Physical quadrature weight: W = (1/6) * |det J|
    Real W = fabs(det) / 6.0;

    // -------------------------------------------------------------------
    // Previous velocity at this element's nodes
    // -------------------------------------------------------------------
    Real u_prev[4][3];
    for (int a = 0; a < 4; a++) {
        int ni = nodes[a];
        u_prev[a][0] = vel_prev[3*ni + 0];
        u_prev[a][1] = vel_prev[3*ni + 1];
        u_prev[a][2] = vel_prev[3*ni + 2];
    }

    // Velocity at element centroid (avg of nodal values, N_a = 0.25)
    Real u_avg[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 4; a++)
        for (int d = 0; d < 3; d++)
            u_avg[d] += 0.25 * u_prev[a][d];

    Real u_mag = sqrt(u_avg[0]*u_avg[0] + u_avg[1]*u_avg[1] + u_avg[2]*u_avg[2]);

    // -------------------------------------------------------------------
    // Stabilisation parameter  τ (Tezduyar formula)
    // -------------------------------------------------------------------
    Real nu   = mu / rho;
    // Guard against degenerate elements (zero or near-zero h)
    static constexpr Real H_EPSILON = 1e-20;
    Real inv_h = 1.0 / (h + H_EPSILON);
    // τ = [ (2/dt)² + (2|u|/h)² + (4ν/h²)² ]^{-1/2}
    Real t1 = 2.0 * inv_dt;
    Real t2 = 2.0 * u_mag * inv_h;
    Real t3 = 4.0 * nu * inv_h * inv_h;
    Real tau = 1.0 / sqrt(t1*t1 + t2*t2 + t3*t3);

    // -------------------------------------------------------------------
    // Precompute per-node streamline convection:  conv[a] = u_avg · ∇N_a
    // -------------------------------------------------------------------
    Real conv[4];
    for (int a = 0; a < 4; a++) {
        conv[a] = 0.0;
        for (int j = 0; j < 3; j++)
            conv[a] += u_avg[j] * gN[a][j];
    }

    // Previous velocity at centroid (same as u_avg; listed separately for clarity)
    // u_avg_prev[d] = Σ_a 0.25 * u_prev[a][d]  — same as u_avg above.

    // -------------------------------------------------------------------
    // Assembly loop over all local node pairs (a, b)
    // -------------------------------------------------------------------
    for (int a = 0; a < 4; a++) {
        int ni = nodes[a];

        for (int b = 0; b < 4; b++) {
            int nj = nodes[b];

            // Precompute shared scalars
            Real Na  = 0.25;   // N_a at centroid
            Real Nb  = 0.25;   // N_b at centroid

            Real gNa_gNb = 0.0;  // ∇N_a · ∇N_b
            for (int j = 0; j < 3; j++)
                gNa_gNb += gN[a][j] * gN[b][j];

            // ----------------------------------------------------------
            // 1.  Velocity–velocity  (same direction d):  DOF (4*ni+d, 4*nj+d)
            // ----------------------------------------------------------
            Real Kvv = W * (
                rho * inv_dt * Na * Nb                               // mass
                + mu  * gNa_gNb                                      // viscous
                + rho * Na * conv[b]                                  // convection
                + tau * rho * conv[a] * (inv_dt * Nb + conv[b])      // SUPG
            );

            for (int d = 0; d < 3; d++) {
                atomic_add_csr(K_offsets, K_columns, K_values,
                               4*ni + d, 4*nj + d, Kvv);
            }

            // ----------------------------------------------------------
            // 2.  Velocity–pressure:  DOF (4*ni+d, 4*nj+3)
            //     Standard gradient:  -Na * ∂N_b/∂x_d  (note: the test fn is N_a)
            //     Wait — see derivation in .cuh: -N_b(centroid) * gradN_a[d]
            //     because the pressure integral is -∫ p ∂N_a/∂x_d dΩ.
            //     After discretising p = Σ_b p_b N_b:
            //        contribution of p_b = -N_b(centroid) * gradN_a[d] * W
            //     SUPG gradient: +τ * conv_a * gradN_b[d] * W
            // ----------------------------------------------------------
            for (int d = 0; d < 3; d++) {
                Real Kvp = W * (
                    -Nb * gN[a][d]                   // gradient  (-N_b * ∂N_a/∂x_d)
                    + tau * conv[a] * gN[b][d]        // SUPG gradient
                );
                atomic_add_csr(K_offsets, K_columns, K_values,
                               4*ni + d, 4*nj + 3, Kvp);
            }

            // ----------------------------------------------------------
            // 3.  Pressure–velocity:  DOF (4*ni+3, 4*nj+d)
            //     Divergence:    +N_a(centroid) * gradN_b[d] * W
            //     PSPG velocity: τ * gradN_a[d] * (N_b/dt + conv_b) * W
            // ----------------------------------------------------------
            for (int d = 0; d < 3; d++) {
                Real Kpv = W * (
                    Na * gN[b][d]                               // divergence
                    + tau * gN[a][d] * (inv_dt * Nb + conv[b]) // PSPG
                );
                atomic_add_csr(K_offsets, K_columns, K_values,
                               4*ni + 3, 4*nj + d, Kpv);
            }

            // ----------------------------------------------------------
            // 4.  Pressure–pressure:  DOF (4*ni+3, 4*nj+3)
            //     PSPG pressure: τ/ρ * (∇N_a · ∇N_b) * W
            // ----------------------------------------------------------
            Real Kpp = W * (tau / rho) * gNa_gNb;
            atomic_add_csr(K_offsets, K_columns, K_values,
                           4*ni + 3, 4*nj + 3, Kpp);
        }  // end loop b

        // ------------------------------------------------------------------
        // RHS contributions from node a.
        // Standard mass:  ρ/dt * N_a(centroid) * u^n_avg[d] * W
        // SUPG mass:      τ * conv_a * (ρ/dt * u^n_avg[d]) * W
        //  combined:  W * ρ/dt * u^n_avg[d] * (0.25 + τ * conv_a)
        // ------------------------------------------------------------------
        for (int d = 0; d < 3; d++) {
            Real rhs_v = W * rho * inv_dt * u_avg[d] * (0.25 + tau * conv[a]);
            atomic_add_rhs(rhs, 4*ni + d, rhs_v);
        }

        // PSPG RHS at node a:  W * τ * (∇N_a · u^n_avg)
        Real pspg_rhs = 0.0;
        for (int d = 0; d < 3; d++)
            pspg_rhs += gN[a][d] * u_avg[d];
        pspg_rhs *= W * tau * inv_dt;   // τ * (1/ρ) * ρ/dt  →  τ * inv_dt
        atomic_add_rhs(rhs, 4*ni + 3, pspg_rhs);

    }  // end loop a
}

// ==========================================================================
//  Kernel 4:  Apply Dirichlet BCs
//   - Rows corresponding to constrained DOFs: set identity row, RHS = value.
//   - Columns corresponding to constrained DOFs in free rows: zero them out
//     (and symmetrically correct RHS).
// ==========================================================================
__global__ void ns_apply_dirichlet_kernel(const int*  __restrict__ is_bc_dof,
                                          const Real* __restrict__ bc_val,
                                          const int*  __restrict__ K_offsets,
                                          const int*  __restrict__ K_columns,
                                          Real*                    K_values,
                                          Real*                    rhs,
                                          int                      n_dof) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_dof) return;

    int row_start = K_offsets[row];
    int row_end   = K_offsets[row + 1];

    if (is_bc_dof[row]) {
        // Identity row, prescribed value in RHS.
        for (int k = row_start; k < row_end; ++k)
            K_values[k] = (K_columns[k] == row) ? Real(1) : Real(0);
        rhs[row] = bc_val[row];
    } else {
        // Free row: zero out columns that correspond to constrained DOFs.
        for (int k = row_start; k < row_end; ++k) {
            int col = K_columns[k];
            if (is_bc_dof[col]) {
                rhs[row] -= K_values[k] * bc_val[col];
                K_values[k] = Real(0);
            }
        }
    }
}

// ==========================================================================
//  Kernel 5:  Extract velocity DOFs from interleaved solution into the
//             previous-velocity array (3*n_nodes, vel_prev[3*i + d]).
// ==========================================================================
__global__ void ns_extract_velocity_kernel(const Real* __restrict__ sol,
                                           Real*                    vel_prev,
                                           int                      n_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    for (int d = 0; d < 3; d++)
        vel_prev[3*i + d] = sol[4*i + d];
}

// ==========================================================================
//  Host-side NavierStokesSUPGPSPGSolver methods
// ==========================================================================

NavierStokesSUPGPSPGSolver::NavierStokesSUPGPSPGSolver(
        const MatrixXR& nodes,
        const MatrixXi& elements,
        const NavierStokesSUPGPSPGParams& params)
    : n_nodes_(nodes.rows())
    , n_elems_(elements.rows())
    , n_dof_(4 * nodes.rows())
    , params_(params)
{
    // ------- cuBLAS / cuSPARSE handles -------
    CHECK_CUSPARSE(cusparseCreate(&cusparse_));
    CHECK_CUBLAS(cublasCreate(&cublas_));

    // ------- Mesh arrays -------
    da_nodes_.resize(static_cast<size_t>(n_nodes_) * 3);
    da_nodes_.BindDevicePointer(&d_nodes_);
    for (int i = 0; i < n_nodes_; i++) {
        da_nodes_.host()[3*i+0] = nodes(i, 0);
        da_nodes_.host()[3*i+1] = nodes(i, 1);
        da_nodes_.host()[3*i+2] = nodes(i, 2);
    }
    da_nodes_.ToDevice();

    da_connect_.resize(static_cast<size_t>(n_elems_) * 4);
    da_connect_.BindDevicePointer(&d_connect_);
    for (int e = 0; e < n_elems_; e++)
        for (int a = 0; a < 4; a++)
            da_connect_.host()[4*e + a] = elements(e, a);
    da_connect_.ToDevice();

    // ------- Geometry (computed by PrecomputeGeometry) -------
    da_gradN_.resize(static_cast<size_t>(n_elems_) * 12);
    da_gradN_.BindDevicePointer(&d_gradN_);
    da_gradN_.SetVal(Real(0));
    da_gradN_.MakeReadyDevice();

    da_detJ_.resize(static_cast<size_t>(n_elems_));
    da_detJ_.BindDevicePointer(&d_detJ_);
    da_detJ_.SetVal(Real(0));
    da_detJ_.MakeReadyDevice();

    da_h_elem_.resize(static_cast<size_t>(n_elems_));
    da_h_elem_.BindDevicePointer(&d_h_elem_);
    da_h_elem_.SetVal(Real(0));
    da_h_elem_.MakeReadyDevice();

    // ------- State vectors -------
    da_sol_.resize(static_cast<size_t>(n_dof_));
    da_sol_.BindDevicePointer(&d_sol_);
    da_sol_.SetVal(Real(0));
    da_sol_.MakeReadyDevice();

    da_sol_prev_.resize(static_cast<size_t>(n_nodes_) * 3);
    da_sol_prev_.BindDevicePointer(&d_sol_prev_);
    da_sol_prev_.SetVal(Real(0));
    da_sol_prev_.MakeReadyDevice();

    // ------- BC arrays (initially unconstrained) -------
    da_is_bc_dof_.resize(static_cast<size_t>(n_dof_));
    da_is_bc_dof_.BindDevicePointer(&d_is_bc_dof_);
    da_is_bc_dof_.SetVal(0);
    da_is_bc_dof_.MakeReadyDevice();

    da_bc_val_.resize(static_cast<size_t>(n_dof_));
    da_bc_val_.BindDevicePointer(&d_bc_val_);
    da_bc_val_.SetVal(Real(0));
    da_bc_val_.MakeReadyDevice();

    // ------- RHS & solver workspace -------
    auto alloc_vec = [this](mophi::DualArray<Real>& da, Real** ptr) {
        da.resize(static_cast<size_t>(n_dof_));
        da.BindDevicePointer(ptr);
        da.SetVal(Real(0));
        da.MakeReadyDevice();
    };
    alloc_vec(da_rhs_,  &d_rhs_);
    alloc_vec(da_x_,    &d_x_);
    alloc_vec(da_r_,    &d_r_);
    alloc_vec(da_rhat_, &d_rhat_);
    alloc_vec(da_p_,    &d_p_);
    alloc_vec(da_v_,    &d_v_);
    alloc_vec(da_s_,    &d_s_);
    alloc_vec(da_t_,    &d_t_);

    // Precompute element geometry (does not change for Eulerian formulation).
    PrecomputeGeometry();
}

NavierStokesSUPGPSPGSolver::~NavierStokesSUPGPSPGSolver() {
    // DualArrays free themselves.
    if (cusparse_) cusparseDestroy(cusparse_);
    if (cublas_)   cublasDestroy(cublas_);
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::SetNoSlipBC(const std::vector<int>& node_ids) {
    // Build uniform zero-velocity entries
    std::vector<Real> zero_vel(node_ids.size() * 3, Real(0));
    // Re-use SetDirichletVelocity mechanism but with zero vector
    // First clear existing BC data from previous calls
    dirichlet_node_ids_.clear();
    dirichlet_vel_.clear();
    for (int n : node_ids) {
        dirichlet_node_ids_.push_back(n);
        dirichlet_vel_.push_back(Real(0));
        dirichlet_vel_.push_back(Real(0));
        dirichlet_vel_.push_back(Real(0));
    }
    bc_needs_rebuild_ = true;
}

void NavierStokesSUPGPSPGSolver::SetDirichletVelocity(
        const std::vector<int>& node_ids, Real ux, Real uy, Real uz) {
    for (int n : node_ids) {
        dirichlet_node_ids_.push_back(n);
        dirichlet_vel_.push_back(ux);
        dirichlet_vel_.push_back(uy);
        dirichlet_vel_.push_back(uz);
    }
    bc_needs_rebuild_ = true;
}

void NavierStokesSUPGPSPGSolver::SetInitialVelocity(const VectorXR& vel) {
    if (vel.size() != n_nodes_ * 3) {
        MOPHI_ERROR("NavierStokesSUPGPSPGSolver::SetInitialVelocity: size mismatch");
        return;
    }
    // Copy vel into sol (interleaved) and sol_prev
    for (int i = 0; i < n_nodes_; i++) {
        da_sol_.host()[4*i+0] = vel(3*i+0);
        da_sol_.host()[4*i+1] = vel(3*i+1);
        da_sol_.host()[4*i+2] = vel(3*i+2);
        da_sol_.host()[4*i+3] = Real(0);
    }
    da_sol_.ToDevice();
    for (int i = 0; i < n_nodes_; i++) {
        da_sol_prev_.host()[3*i+0] = vel(3*i+0);
        da_sol_prev_.host()[3*i+1] = vel(3*i+1);
        da_sol_prev_.host()[3*i+2] = vel(3*i+2);
    }
    da_sol_prev_.ToDevice();
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::Step() {
    // 1. Rebuild BC arrays if needed
    if (bc_needs_rebuild_) {
        da_is_bc_dof_.SetVal(0);
        da_is_bc_dof_.MakeReadyDevice();
        da_bc_val_.SetVal(Real(0));
        da_bc_val_.MakeReadyDevice();
        for (int k = 0; k < static_cast<int>(dirichlet_node_ids_.size()); k++) {
            int n = dirichlet_node_ids_[k];
            for (int d = 0; d < 3; d++) {
                da_is_bc_dof_.host()[4*n + d] = 1;
                da_bc_val_.host()[4*n + d]    = dirichlet_vel_[3*k + d];
            }
        }
        da_is_bc_dof_.ToDevice();
        da_bc_val_.ToDevice();
        bc_needs_rebuild_ = false;
    }

    // 2. Build CSR pattern once
    if (!pattern_built_) BuildCSRPattern();

    // 3. Assemble system
    AssembleSystem();

    // 4. Apply Dirichlet BCs
    ApplyDirichletBCs();

    // 5. Solve A x = b with BiCGSTAB
    SolveBiCGSTAB();

    // 6. Update state: copy x → sol, extract velocity → sol_prev
    UpdateState();

    current_time_ += params_.dt;
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::GetVelocity(VectorXR& vel) const {
    // Use explicit device→host copy; DualArray timestamps are bypassed when
    // d_sol_ is written directly via cudaMemcpy in UpdateState().
    std::vector<Real> host_sol(static_cast<size_t>(n_dof_));
    MOPHI_GPU_CALL(cudaMemcpy(host_sol.data(), d_sol_,
            static_cast<size_t>(n_dof_) * sizeof(Real), cudaMemcpyDeviceToHost));
    vel.resize(n_nodes_ * 3);
    for (int i = 0; i < n_nodes_; i++) {
        vel(3*i+0) = host_sol[static_cast<size_t>(4*i+0)];
        vel(3*i+1) = host_sol[static_cast<size_t>(4*i+1)];
        vel(3*i+2) = host_sol[static_cast<size_t>(4*i+2)];
    }
}

void NavierStokesSUPGPSPGSolver::GetPressure(VectorXR& pres) const {
    std::vector<Real> host_sol(static_cast<size_t>(n_dof_));
    MOPHI_GPU_CALL(cudaMemcpy(host_sol.data(), d_sol_,
            static_cast<size_t>(n_dof_) * sizeof(Real), cudaMemcpyDeviceToHost));
    pres.resize(n_nodes_);
    for (int i = 0; i < n_nodes_; i++)
        pres(i) = host_sol[static_cast<size_t>(4*i+3)];
}

// ==========================================================================
//  Private stage implementations
// ==========================================================================

void NavierStokesSUPGPSPGSolver::PrecomputeGeometry() {
    constexpr int threads = 256;
    const int blocks = (n_elems_ + threads - 1) / threads;
    ns_precompute_geom_kernel<<<blocks, threads>>>(
            d_nodes_, d_connect_, d_gradN_, d_detJ_, d_h_elem_, n_elems_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::BuildCSRPattern() {
    constexpr int KEYS_PER_ELEM = 256;
    const int total_raw = n_elems_ * KEYS_PER_ELEM;

    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys,
            static_cast<size_t>(total_raw) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_raw + threads - 1) / threads;
        ns_build_keys_kernel<<<blocks, threads>>>(d_connect_, d_keys, n_elems_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<unsigned long long> keys_begin(d_keys);
    thrust::sort(thrust::device, keys_begin, keys_begin + total_raw);
    auto keys_end_unique = thrust::unique(thrust::device, keys_begin, keys_begin + total_raw);
    K_nnz_ = static_cast<int>(keys_end_unique - keys_begin);

    da_K_offsets_.resize(static_cast<size_t>(n_dof_ + 1));
    da_K_offsets_.BindDevicePointer(&d_K_offsets_);
    da_K_columns_.resize(static_cast<size_t>(K_nnz_));
    da_K_columns_.BindDevicePointer(&d_K_columns_);
    da_K_values_.resize(static_cast<size_t>(K_nnz_));
    da_K_values_.BindDevicePointer(&d_K_values_);

    int* d_row_counts = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_row_counts, static_cast<size_t>(n_dof_) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_row_counts, 0, static_cast<size_t>(n_dof_) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (K_nnz_ + threads - 1) / threads;
        ns_decode_keys_kernel<<<blocks, threads>>>(d_keys, K_nnz_, d_K_columns_, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> row_ptr(d_row_counts);
    thrust::device_ptr<int> off_ptr(d_K_offsets_);
    thrust::exclusive_scan(thrust::device, row_ptr, row_ptr + n_dof_, off_ptr);
    ns_set_last_offset_kernel<<<1,1>>>(d_K_offsets_, n_dof_, K_nnz_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    MOPHI_INFO("NavierStokesSUPGPSPGSolver: CSR pattern built — n_dof=%d  nnz=%d", n_dof_, K_nnz_);
    pattern_built_ = true;
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::AssembleSystem() {
    // Zero out matrix values and RHS
    da_K_values_.SetVal(Real(0));
    da_K_values_.MakeReadyDevice();
    da_rhs_.SetVal(Real(0));
    da_rhs_.MakeReadyDevice();

    constexpr int threads = 128;
    const int blocks = (n_elems_ + threads - 1) / threads;

    ns_assemble_kernel<<<blocks, threads>>>(
            d_connect_, d_gradN_, d_detJ_, d_h_elem_,
            d_sol_prev_,
            d_K_offsets_, d_K_columns_, d_K_values_,
            d_rhs_,
            params_.rho, params_.mu, Real(1) / params_.dt,
            n_elems_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::ApplyDirichletBCs() {
    if (dirichlet_node_ids_.empty()) return;

    constexpr int threads = 256;
    const int blocks = (n_dof_ + threads - 1) / threads;
    ns_apply_dirichlet_kernel<<<blocks, threads>>>(
            d_is_bc_dof_, d_bc_val_,
            d_K_offsets_, d_K_columns_, d_K_values_,
            d_rhs_, n_dof_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::SolveBiCGSTAB() {
    const int n = n_dof_;
    const Real one  = 1.0;
    const Real zero = 0.0;

    // --- cuSPARSE descriptors ---
    cusparseSpMatDescr_t A_descr = nullptr;
    cusparseDnVecDescr_t p_descr = nullptr;
    cusparseDnVecDescr_t v_descr = nullptr;
    cusparseDnVecDescr_t s_descr = nullptr;
    cusparseDnVecDescr_t t_descr = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(&A_descr, n, n, K_nnz_,
            d_K_offsets_, d_K_columns_, d_K_values_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&p_descr, n, d_p_, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&v_descr, n, d_v_, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&s_descr, n, d_s_, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&t_descr, n, d_t_, CUDA_R_64F));

    // Buffer size query (use p_descr/v_descr for SpMV, s_descr/t_descr for the second SpMV)
    size_t spMV_bufsize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, A_descr, p_descr, &zero, v_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spMV_bufsize));
    void* spMV_buf = nullptr;
    if (spMV_bufsize > 0)
        MOPHI_GPU_CALL(cudaMalloc(&spMV_buf, spMV_bufsize));

    // --- Initialise iterate x = 0, r = b, r_hat = b ---
    da_x_.SetVal(Real(0));
    da_x_.MakeReadyDevice();
    MOPHI_GPU_CALL(cudaMemcpy(d_r_,    d_rhs_, n * sizeof(Real), cudaMemcpyDeviceToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_rhat_, d_rhs_, n * sizeof(Real), cudaMemcpyDeviceToDevice));
    da_p_.SetVal(Real(0));
    da_p_.MakeReadyDevice();
    da_v_.SetVal(Real(0));
    da_v_.MakeReadyDevice();

    Real rho_prev = 1.0;
    Real alpha    = 1.0;
    Real omega    = 1.0;

    // Initial residual norm
    Real b_norm = 0.0;
    CHECK_CUBLAS(cublasDdot(cublas_, n, d_rhs_, 1, d_rhs_, 1, &b_norm));
    b_norm = std::sqrt(b_norm);
    if (b_norm < 1e-30) {
        // RHS is zero — solution is trivially zero.
        last_iter_     = 0;
        last_residual_ = 0.0;
        goto cleanup;
    }

    for (int iter = 0; iter < params_.max_bicgstab; ++iter) {
        // ρ_i = r_hat · r
        Real rho = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_rhat_, 1, d_r_, 1, &rho));

        if (std::abs(rho) < 1e-300) {
            // Breakdown — restart with r_hat = r
            MOPHI_GPU_CALL(cudaMemcpy(d_rhat_, d_r_, n * sizeof(Real), cudaMemcpyDeviceToDevice));
            rho = 0.0;
            CHECK_CUBLAS(cublasDdot(cublas_, n, d_rhat_, 1, d_r_, 1, &rho));
        }

        Real beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        {
            const Real neg_omega = -omega;
            CHECK_CUBLAS(cublasDaxpy(cublas_, n, &neg_omega, d_v_, 1, d_p_, 1));  // p -= omega*v
            CHECK_CUBLAS(cublasDscal(cublas_, n, &beta,      d_p_, 1));             // p *= beta
            CHECK_CUBLAS(cublasDaxpy(cublas_, n, &one,       d_r_, 1, d_p_, 1));  // p += r
        }

        // v = A * p
        // Need to update the descriptor's device pointer (it was bound in creation)
        CHECK_CUSPARSE(cusparseDnVecSetValues(p_descr, d_p_));
        CHECK_CUSPARSE(cusparseDnVecSetValues(v_descr, d_v_));
        CHECK_CUSPARSE(cusparseSpMV(cusparse_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, A_descr, p_descr, &zero, v_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spMV_buf));

        // alpha = rho / (r_hat · v)
        Real rhat_v = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_rhat_, 1, d_v_, 1, &rhat_v));
        if (std::abs(rhat_v) < 1e-300) {
            MOPHI_ERROR("NavierStokesSUPGPSPGSolver BiCGSTAB: breakdown (r_hat.v=0)");
            break;
        }
        alpha = rho / rhat_v;

        // s = r - alpha * v
        MOPHI_GPU_CALL(cudaMemcpy(d_s_, d_r_, n * sizeof(Real), cudaMemcpyDeviceToDevice));
        {
            const Real neg_alpha = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublas_, n, &neg_alpha, d_v_, 1, d_s_, 1));
        }

        // Check ||s||
        Real s_norm = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_s_, 1, d_s_, 1, &s_norm));
        s_norm = std::sqrt(s_norm);
        if (s_norm / b_norm < params_.bicgstab_tol) {
            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublas_, n, &alpha, d_p_, 1, d_x_, 1));
            last_iter_     = iter + 1;
            last_residual_ = s_norm / b_norm;
            goto cleanup;
        }

        // t = A * s
        CHECK_CUSPARSE(cusparseDnVecSetValues(s_descr, d_s_));
        CHECK_CUSPARSE(cusparseDnVecSetValues(t_descr, d_t_));
        CHECK_CUSPARSE(cusparseSpMV(cusparse_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, A_descr, s_descr, &zero, t_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spMV_buf));

        // omega = (t · s) / (t · t)
        Real ts = 0.0, tt = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_t_, 1, d_s_, 1, &ts));
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_t_, 1, d_t_, 1, &tt));
        omega = (tt > 1e-300) ? ts / tt : Real(0);

        // x += alpha*p + omega*s
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &alpha, d_p_, 1, d_x_, 1));
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &omega, d_s_, 1, d_x_, 1));

        // r = s - omega * t
        MOPHI_GPU_CALL(cudaMemcpy(d_r_, d_s_, n * sizeof(Real), cudaMemcpyDeviceToDevice));
        {
            const Real neg_omega = -omega;
            CHECK_CUBLAS(cublasDaxpy(cublas_, n, &neg_omega, d_t_, 1, d_r_, 1));
        }

        // Check convergence
        Real r_norm = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_r_, 1, d_r_, 1, &r_norm));
        r_norm = std::sqrt(r_norm);
        Real rel = r_norm / b_norm;

        if (rel < params_.bicgstab_tol) {
            last_iter_     = iter + 1;
            last_residual_ = rel;
            goto cleanup;
        }

        rho_prev = rho;

        if (iter == params_.max_bicgstab - 1) {
            last_iter_     = params_.max_bicgstab;
            last_residual_ = rel;
        }
    }

cleanup:
    CHECK_CUSPARSE(cusparseDestroySpMat(A_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(p_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(v_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(s_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(t_descr));
    if (spMV_buf) cudaFree(spMV_buf);
}

// --------------------------------------------------------------------------
void NavierStokesSUPGPSPGSolver::UpdateState() {
    // Copy the BiCGSTAB iterate x → sol
    MOPHI_GPU_CALL(cudaMemcpy(d_sol_, d_x_,
            static_cast<size_t>(n_dof_) * sizeof(Real),
            cudaMemcpyDeviceToDevice));

    // Extract velocity DOFs from sol → sol_prev (for next time step)
    constexpr int threads = 256;
    const int blocks = (n_nodes_ + threads - 1) / threads;
    ns_extract_velocity_kernel<<<blocks, threads>>>(d_sol_, d_sol_prev_, n_nodes_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

}  // namespace tlfea
