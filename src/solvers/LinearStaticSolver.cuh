#pragma once
/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LinearStaticSolver.cuh
 * Brief:   Declares the LinearStaticSolver class, which performs a single-step
 *          steady-state (linear) FEA solve for FEAT10 TET10 elements entirely
 *          on the GPU.
 *
 *          The solver:
 *            1. Builds the linearised (small-strain) tangent stiffness matrix K
 *               in 3N×3N CSR format using the existing SVK tangent block
 *               evaluated at the reference configuration (F = I).
 *            2. Applies Dirichlet boundary conditions by row/column elimination.
 *            3. Solves K·u = f_ext via a conjugate-gradient loop implemented
 *               with cuBLAS (dot, axpy, scal) and cuSPARSE (SpMV).
 *            4. Updates the nodal positions stored in GPU_FEAT10_Data.
 *==============================================================
 *==============================================================*/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <MoPhiEssentials.h>

#include "../elements/FEAT10Data.cuh"
#include "../types.h"
#include "SolverBase.h"

namespace tlfea {

class LinearStaticSolver : public SolverBase {
  public:
    /**
     * Constructor.
     *
     * @param data      Fully initialised GPU_FEAT10_Data.  The caller must have
     *                  already invoked Setup(), SetSVK(), CalcDnDuPre(), and
     *                  (optionally) SetNodalFixed() / SetExternalForce() before
     *                  calling Solve().
     * @param tol       CG relative-residual convergence tolerance.
     * @param max_iter  Maximum number of CG iterations.
     */
    LinearStaticSolver(GPU_FEAT10_Data* data, Real tol = 1e-10, int max_iter = 10000);
    ~LinearStaticSolver();

    /**
     * Assemble K, apply BCs, solve K·u = f_ext on GPU, then write the
     * computed displacement back into the GPU_FEAT10_Data nodal positions.
     */
    void Solve() override;

    /** No-op: parameters are set through the constructor. */
    void SetParameters(void* /*params*/) override {}

    /** Number of CG iterations performed during the last Solve() call. */
    int GetLastIterCount() const { return last_iter_count_; }

    /** Relative residual norm at the end of the last Solve() call. */
    Real GetLastResidual() const { return last_residual_; }

  private:
    /* ---- internal stages ---- */
    void BuildStiffnessCSRPattern();
    void AssembleLinearStiffness();
    void ApplyDirichletBCs();
    void SolveLinearSystemCG();
    void UpdatePositions();

    /* ---- data ---- */
    GPU_FEAT10_Data* data_;
    int n_dof_;  // 3 * n_nodes

    /* Stiffness K in CSR format (n_dof × n_dof) */
    int* d_K_offsets_ = nullptr;
    int* d_K_columns_ = nullptr;
    Real* d_K_values_ = nullptr;
    int K_nnz_ = 0;
    bool pattern_built_ = false;

    /* CG workspace (length n_dof each) */
    Real* d_u_ = nullptr;   // displacement solution
    Real* d_f_ = nullptr;   // RHS (copy of f_ext, modified in-place for BCs)
    Real* d_r_ = nullptr;   // residual
    Real* d_p_ = nullptr;   // search direction
    Real* d_Kp_ = nullptr;  // K * p

    /* cuBLAS / cuSPARSE handles */
    cusparseHandle_t cusparse_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    /* CG parameters */
    Real cg_tol_;
    int cg_max_iter_;

    /* Diagnostics from last Solve() */
    int last_iter_count_ = 0;
    Real last_residual_ = 0.0;
};

}  // namespace tlfea
