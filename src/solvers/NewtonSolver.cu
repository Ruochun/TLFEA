/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    NewtonSolver.cu
 * Brief:   Implements the CPU-GPU hybrid Newton-Raphson solver.
 *          GPU: per-element stress (P), internal force, and
 *               tangent-stiffness assembly (via compute_hessian_assemble_csr).
 *          CPU: forms the effective system matrix M/h + h·K_stiff
 *               and solves it with Eigen SparseLU.
 *
 *  Time-integration scheme: BDF1 (backward Euler) in velocity space.
 *    x_{n+1} = x_n + h · v_{n+1}
 *  Residual:
 *    R(v) = M·(v − v_n)/h + f_int(x_n + h·v) − f_ext
 *  Effective stiffness:
 *    K_eff = M/h + h·K_tangent
 *  (K_tangent = df_int/dx, so df_int/dv = h·K_tangent)
 *==============================================================
 *==============================================================*/

#include <set>
#include <algorithm>
#include <iostream>

#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../utils/quadrature_utils.h"
#include "NewtonSolver.cuh"
#include <MoPhiEssentials.h>

namespace tlfea {

// ===========================================================
// GPU kernels
// ===========================================================

/// Update trial positions: x_trial = x_prev + h * v
__global__ void newton_update_positions_kernel(Real* x, Real* y, Real* z,
                                               const Real* x_prev,
                                               const Real* y_prev,
                                               const Real* z_prev,
                                               const Real* v,
                                               Real h, int n_coef) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_coef) return;
    x[i] = x_prev[i] + h * v[3 * i + 0];
    y[i] = y_prev[i] + h * v[3 * i + 1];
    z[i] = z_prev[i] + h * v[3 * i + 2];
}

/// Assemble tangent stiffness into external CSR arrays.
/// One thread per (element, quadrature-point) pair.
/// The parameter h scales stiffness: K_stored = h * K_tangent,
/// so K_eff = M/h + K_stored when h = time_step is passed.
__global__ void newton_assemble_stiffness_t10_kernel(GPU_FEAT10_Data* d_data,
                                                     int n_elem,
                                                     int n_qp,
                                                     int*  d_K_offsets,
                                                     int*  d_K_columns,
                                                     Real* d_K_values,
                                                     Real  h) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_elem * n_qp;
    if (tid >= total) return;
    int elem_idx = tid / n_qp;
    int qp_idx   = tid % n_qp;
    compute_hessian_assemble_csr(d_data,
                                 static_cast<SyncedNewtonSolver*>(nullptr),
                                 elem_idx, qp_idx,
                                 d_K_offsets, d_K_columns, d_K_values, h);
}

/// Update velocity: v += delta_v
__global__ void newton_update_v_kernel(Real* v, const Real* delta_v, int n_dof) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_dof) v[i] += delta_v[i];
}

// ===========================================================
// Helper: thread-block count
// ===========================================================
static inline int blk(int n, int t = 128) { return (n + t - 1) / t; }

// ===========================================================
// NewtonSolver implementation
// ===========================================================

NewtonSolver::NewtonSolver(ElementBase* data,
                           const MatrixXi& element_connectivity,
                           const std::vector<int>& fixed_dofs)
    : fixed_dofs_(fixed_dofs) {
    if (data->type == TYPE_T10) {
        type_        = TYPE_T10;
        d_t10_       = static_cast<GPU_FEAT10_Data*>(data);
        d_data_elem_ = d_t10_->d_data;   // device copy
        n_coef_      = d_t10_->get_n_coef();
        n_elem_      = d_t10_->get_n_elem();
        n_qp_        = Quadrature::N_QP_T10_5;
        n_shape_     = Quadrature::N_NODE_T10_10;
    } else {
        MOPHI_ERROR("NewtonSolver: only TYPE_T10 elements are supported.");
    }

    MOPHI_GPU_CALL(cudaMalloc(&d_v_,      n_coef_ * 3 * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_x_prev_, n_coef_ * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_y_prev_, n_coef_ * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_z_prev_, n_coef_ * sizeof(Real)));

    BuildStiffnessPattern(element_connectivity);

    MOPHI_GPU_CALL(cudaMalloc(&d_K_offsets_, (3 * n_coef_ + 1) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc(&d_K_columns_, K_nnz_ * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc(&d_K_values_,  K_nnz_ * sizeof(Real)));

    MOPHI_GPU_CALL(cudaMemcpy(d_K_offsets_, K_offsets_cpu_.data(),
                              (3 * n_coef_ + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_K_columns_, K_columns_cpu_.data(),
                              K_nnz_ * sizeof(int),
                              cudaMemcpyHostToDevice));
}

NewtonSolver::~NewtonSolver() {
    cudaFree(d_v_);
    cudaFree(d_v_prev_);
    cudaFree(d_x_prev_);
    cudaFree(d_y_prev_);
    cudaFree(d_z_prev_);
    cudaFree(d_K_offsets_);
    cudaFree(d_K_columns_);
    cudaFree(d_K_values_);
}

void NewtonSolver::BuildStiffnessPattern(const MatrixXi& element_connectivity) {
    int n_dof = 3 * n_coef_;
    std::vector<std::set<int>> adj(n_coef_);
    for (int e = 0; e < element_connectivity.rows(); ++e) {
        int npe = element_connectivity.cols();
        for (int a = 0; a < npe; ++a) {
            int na = element_connectivity(e, a);
            for (int b = 0; b < npe; ++b)
                adj[na].insert(element_connectivity(e, b));
        }
    }

    K_offsets_cpu_.resize(n_dof + 1, 0);
    K_columns_cpu_.clear();
    for (int ni = 0; ni < n_coef_; ++ni) {
        for (int di = 0; di < 3; ++di) {
            int row = 3 * ni + di;
            std::vector<int> cols;
            for (int nj : adj[ni]) {
                cols.push_back(3 * nj + 0);
                cols.push_back(3 * nj + 1);
                cols.push_back(3 * nj + 2);
            }
            std::sort(cols.begin(), cols.end());
            K_offsets_cpu_[row + 1] = K_offsets_cpu_[row] + (int)cols.size();
            for (int c : cols) K_columns_cpu_.push_back(c);
        }
    }
    K_nnz_ = (int)K_columns_cpu_.size();
}

void NewtonSolver::SetParameters(void* params) {
    auto* p      = static_cast<NewtonSolverParams*>(params);
    h_           = p->time_step;
    newton_tol_  = p->newton_tol;
    max_newton_  = p->max_newton_iters;
}

void NewtonSolver::Setup() {
    MOPHI_GPU_CALL(cudaMemset(d_v_,      0, n_coef_ * 3 * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMemset(d_x_prev_, 0, n_coef_ * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMemset(d_y_prev_, 0, n_coef_ * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMemset(d_z_prev_, 0, n_coef_ * sizeof(Real)));
}

// -----------------------------------------------------------
// AssembleAndSolveNewtonStep
//
//  Precondition: d_v_ holds the current velocity estimate.
//  1. Update trial positions: x = x_prev + h*v
//  2. Compute P (stress) and f_int (GPU, via existing methods)
//  3. Transfer f_int, f_ext, mass CSR, v, v_prev to CPU
//  4. Build residual: R = M*(v-v_prev)/h + f_int - f_ext
//  5. Zero & assemble K_stiff on GPU; transfer to CPU
//  6. Form K_eff = M/h + K_stiff (K_stiff = h*K_tangent)
//  7. Eliminate fixed DOFs
//  8. Solve K_eff * delta_v = -R (Eigen SparseLU)
// -----------------------------------------------------------
void NewtonSolver::AssembleAndSolveNewtonStep(VectorXR& delta_v_cpu) {
    int n_dof = 3 * n_coef_;

    // ---- 1. Update trial positions ----
    {
        Real* d_x = const_cast<Real*>(d_t10_->GetX12DevicePtr());
        Real* d_y = const_cast<Real*>(d_t10_->GetY12DevicePtr());
        Real* d_z = const_cast<Real*>(d_t10_->GetZ12DevicePtr());
        newton_update_positions_kernel<<<blk(n_coef_), 128>>>(
            d_x, d_y, d_z,
            d_x_prev_, d_y_prev_, d_z_prev_,
            d_v_, h_, n_coef_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    // ---- 2. Compute P and f_int (use existing GPU methods) ----
    // Clear f_int before accumulation
    MOPHI_GPU_CALL(cudaMemset(d_t10_->GetInternalForceDevicePtr(), 0,
                              n_coef_ * 3 * sizeof(Real)));
    d_t10_->CalcP();
    d_t10_->CalcInternalForce();

    // ---- 3. Transfer to CPU ----
    VectorXR f_int_cpu, f_ext_cpu, v_cpu(n_dof), v_prev_cpu(n_dof);
    d_t10_->RetrieveInternalForceToCPU(f_int_cpu);
    d_t10_->RetrieveExternalForceToCPU(f_ext_cpu);
    MOPHI_GPU_CALL(cudaMemcpy(v_cpu.data(),      d_v_,      n_dof * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaMemcpy(v_prev_cpu.data(), d_v_prev_, n_dof * sizeof(Real),
                              cudaMemcpyDeviceToHost));

    std::vector<int>  mass_offsets, mass_cols;
    std::vector<Real> mass_vals;
    d_t10_->RetrieveMassCSRToCPU(mass_offsets, mass_cols, mass_vals);

    // ---- 4. Residual R = M*(v-v_prev)/h + f_int - f_ext ----
    VectorXR R(n_dof);
    R.setZero();
    // Mass term
    for (int ni = 0; ni < n_coef_; ++ni) {
        int rs = mass_offsets[ni], re = mass_offsets[ni + 1];
        for (int idx = rs; idx < re; ++idx) {
            int nj    = mass_cols[idx];
            Real M_ij = mass_vals[idx];
            for (int d = 0; d < 3; ++d)
                R(3*ni+d) += M_ij * (v_cpu(3*nj+d) - v_prev_cpu(3*nj+d)) / h_;
        }
    }
    R += f_int_cpu - f_ext_cpu;
    for (int dof : fixed_dofs_) R(dof) = 0.0;

    // Build is_fixed set for O(1) lookup
    std::vector<bool> is_fixed(n_dof, false);
    for (int dof : fixed_dofs_) is_fixed[dof] = true;

    // ---- 5. Assemble K_stiff on GPU ----
    // compute_hessian_assemble_csr stores h * K_tangent in K_values,
    // so K_eff = M/h + K_stored is correct for the Newton step.
    MOPHI_GPU_CALL(cudaMemset(d_K_values_, 0, K_nnz_ * sizeof(Real)));
    {
        newton_assemble_stiffness_t10_kernel<<<blk(n_elem_ * n_qp_), 128>>>(
            d_data_elem_, n_elem_, n_qp_,
            d_K_offsets_, d_K_columns_, d_K_values_, h_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }
    std::vector<Real> K_vals(K_nnz_);
    MOPHI_GPU_CALL(cudaMemcpy(K_vals.data(), d_K_values_,
                              K_nnz_ * sizeof(Real), cudaMemcpyDeviceToHost));

    // ---- 6. K_eff = M/h + K_stiff with fixed-DOF elimination ----
    // For fixed DOF i: row and column are set to identity (e_i^T, e_i).
    // R(i)=0 guarantees delta_v(i)=0 from the solve.
    typedef Eigen::SparseMatrix<Real> SpMat;
    typedef Eigen::Triplet<Real>      Trip;

    std::vector<Trip> trips;
    trips.reserve(K_nnz_ + (int)mass_cols.size() * 3 + (int)fixed_dofs_.size());

    // K_stiff contributions (skip entries involving any fixed DOF)
    for (int row = 0; row < n_dof; ++row) {
        if (is_fixed[row]) continue;
        for (int idx = K_offsets_cpu_[row]; idx < K_offsets_cpu_[row+1]; ++idx) {
            int col = K_columns_cpu_[idx];
            if (!is_fixed[col])
                trips.emplace_back(row, col, K_vals[idx]);
        }
    }
    // M/h contributions (skip entries involving any fixed DOF)
    for (int ni = 0; ni < n_coef_; ++ni) {
        for (int idx = mass_offsets[ni]; idx < mass_offsets[ni+1]; ++idx) {
            int nj    = mass_cols[idx];
            Real M_ij = mass_vals[idx];
            for (int d = 0; d < 3; ++d) {
                int row = 3*ni+d, col = 3*nj+d;
                if (!is_fixed[row] && !is_fixed[col])
                    trips.emplace_back(row, col, M_ij / h_);
            }
        }
    }
    // Identity rows/columns for fixed DOFs
    for (int dof : fixed_dofs_)
        trips.emplace_back(dof, dof, 1.0);

    SpMat K_eff(n_dof, n_dof);
    K_eff.setFromTriplets(trips.begin(), trips.end());
    K_eff.makeCompressed();

    // ---- 7. Solve K_eff * delta_v = -R ----
    Eigen::SparseLU<SpMat> lu;
    lu.analyzePattern(K_eff);
    lu.factorize(K_eff);
    if (lu.info() != Eigen::Success) {
        MOPHI_ERROR("NewtonSolver: SparseLU factorization failed.");
        delta_v_cpu.setZero();
        return;
    }
    delta_v_cpu = lu.solve(-R);
    for (int dof : fixed_dofs_) delta_v_cpu(dof) = 0.0;
}

// -----------------------------------------------------------
// ComputeEnergies
//   KE = (1/2) v^T M v      (exact, using mass CSR)
//   SE = (1/2) f_int · (h·v)  (linearized increment estimate)
// -----------------------------------------------------------
void NewtonSolver::ComputeEnergies() {
    int n_dof = 3 * n_coef_;

    VectorXR v_cpu(n_dof);
    MOPHI_GPU_CALL(cudaMemcpy(v_cpu.data(), d_v_,
                              n_dof * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<int>  moff, mcols;
    std::vector<Real> mvals;
    d_t10_->RetrieveMassCSRToCPU(moff, mcols, mvals);

    kinetic_energy_ = 0.0;
    for (int ni = 0; ni < n_coef_; ++ni) {
        for (int idx = moff[ni]; idx < moff[ni+1]; ++idx) {
            int nj    = mcols[idx];
            Real M_ij = mvals[idx];
            for (int d = 0; d < 3; ++d)
                kinetic_energy_ += 0.5 * M_ij
                    * v_cpu(3*ni+d) * v_cpu(3*nj+d);
        }
    }

    // SE: use internal force already computed (last Newton sub-step)
    VectorXR f_int_cpu;
    d_t10_->RetrieveInternalForceToCPU(f_int_cpu);
    strain_energy_ = 0.0;
    for (int i = 0; i < n_dof; ++i)
        strain_energy_ += 0.5 * f_int_cpu(i) * h_ * v_cpu(i);
}

// -----------------------------------------------------------
// Solve – one backward-Euler time step
// -----------------------------------------------------------
void NewtonSolver::Solve() {
    // Save current positions as x_prev
    {
        Real* d_x = const_cast<Real*>(d_t10_->GetX12DevicePtr());
        Real* d_y = const_cast<Real*>(d_t10_->GetY12DevicePtr());
        Real* d_z = const_cast<Real*>(d_t10_->GetZ12DevicePtr());
        MOPHI_GPU_CALL(cudaMemcpy(d_x_prev_, d_x, n_coef_ * sizeof(Real),
                                  cudaMemcpyDeviceToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_y_prev_, d_y, n_coef_ * sizeof(Real),
                                  cudaMemcpyDeviceToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_z_prev_, d_z, n_coef_ * sizeof(Real),
                                  cudaMemcpyDeviceToDevice));
    }

    // Warm-start: v_guess = v_prev
    MOPHI_GPU_CALL(cudaMemcpy(d_v_, d_v_prev_, n_coef_ * 3 * sizeof(Real),
                              cudaMemcpyDeviceToDevice));

    // Newton iterations
    int n_dof = 3 * n_coef_;
    VectorXR delta_v_cpu(n_dof);
    Real* d_delta_v = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_delta_v, n_dof * sizeof(Real)));

    for (int iter = 0; iter < max_newton_; ++iter) {
        AssembleAndSolveNewtonStep(delta_v_cpu);

        Real norm_dv = delta_v_cpu.norm();

        // Upload delta_v and update v on GPU
        MOPHI_GPU_CALL(cudaMemcpy(d_delta_v, delta_v_cpu.data(),
                                  n_dof * sizeof(Real), cudaMemcpyHostToDevice));
        newton_update_v_kernel<<<blk(n_dof), 128>>>(d_v_, d_delta_v, n_dof);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());

        // Convergence: relative norm of velocity update
        VectorXR v_cpu(n_dof);
        MOPHI_GPU_CALL(cudaMemcpy(v_cpu.data(), d_v_, n_dof * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        Real rel = norm_dv / (1.0 + v_cpu.norm());
        if (rel < newton_tol_) break;
    }

    cudaFree(d_delta_v);

    // Final position update: x = x_prev + h*v
    {
        Real* d_x = const_cast<Real*>(d_t10_->GetX12DevicePtr());
        Real* d_y = const_cast<Real*>(d_t10_->GetY12DevicePtr());
        Real* d_z = const_cast<Real*>(d_t10_->GetZ12DevicePtr());
        newton_update_positions_kernel<<<blk(n_coef_), 128>>>(
            d_x, d_y, d_z,
            d_x_prev_, d_y_prev_, d_z_prev_,
            d_v_, h_, n_coef_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    // Advance v_prev <- v
    MOPHI_GPU_CALL(cudaMemcpy(d_v_prev_, d_v_, n_coef_ * 3 * sizeof(Real),
                              cudaMemcpyDeviceToDevice));

    // Compute energies (f_int is current from last Newton sub-step)
    ComputeEnergies();
}

}  // namespace tlfea
