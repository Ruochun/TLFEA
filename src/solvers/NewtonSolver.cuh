/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    NewtonSolver.cuh
 * Brief:   Declares the NewtonSolver class: a CPU-GPU hybrid
 *          Newton-Raphson integrator for implicit BDF1 (backward
 *          Euler) time stepping.  GPU kernels assemble the tangent
 *          stiffness matrix and residual; the CPU assembles the
 *          effective system matrix and solves it with Eigen
 *          SparseLU.  Works with FEAT10, ANCF3243, and ANCF3443
 *          element data.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Sparse>
#include <vector>

#include "../elements/FEAT10Data.cuh"
#include "../elements/ElementBase.h"
#include "../types.h"
#include "../utils/cuda_utils.h"
#include "SolverBase.h"
#include <MoPhiEssentials.h>

namespace tlfea {

// ============================================================
// Forward-declared solver type required by *DataFunc.cuh files.
// The pointer is passed to compute_hessian_assemble_csr but is
// never dereferenced there; the struct only needs to exist.
// ============================================================
struct SyncedNewtonSolver {};

// ============================================================
// Parameter struct passed to NewtonSolver::SetParameters()
// ============================================================
struct NewtonSolverParams {
    Real time_step;        ///< Δt for backward-Euler integration
    Real newton_tol;       ///< Absolute tolerance on residual norm
    int max_newton_iters;  ///< Max Newton iterations per time step
};

// ============================================================
// NewtonSolver – implicit BDF1 Newton-Raphson solver
// ============================================================
class NewtonSolver : public SolverBase {
  public:
    /**
     * @param data               Element data (host-side handle).
     * @param element_connectivity  n_elem × nodes_per_elem matrix
     *                              used to build the stiffness-matrix
     *                              sparsity pattern.
     * @param fixed_dofs         Sorted list of fixed scalar DOF
     *                           indices (0 … 3*n_coef−1).
     */
    NewtonSolver(ElementBase* data, const MatrixXi& element_connectivity, const std::vector<int>& fixed_dofs);

    ~NewtonSolver();

    /**
     * SetParameters – accepts a pointer to NewtonSolverParams.
     */
    void SetParameters(void* params) override;

    /**
     * Setup – uploads the solver state to the device.  Must be
     * called after SetParameters and before Solve.
     */
    void Setup();

    /**
     * Solve – advances the simulation by one time step using
     * backward-Euler + Newton-Raphson.
     */
    void Solve() override;

    // ----------------------------------------------------------
    // Energy accessors (computed at the end of each time step)
    // ----------------------------------------------------------
    Real GetKineticEnergy() const { return kinetic_energy_; }
    Real GetStrainEnergy() const { return strain_energy_; }
    Real GetTotalEnergy() const { return kinetic_energy_ + strain_energy_; }

  private:
    // ----------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------
    void BuildStiffnessPattern(const MatrixXi& element_connectivity);
    void AssembleAndSolveNewtonStep(VectorXR& delta_v_cpu);
    void ComputeEnergies();

    // ----------------------------------------------------------
    // Data
    // ----------------------------------------------------------
    ElementType type_;
    GPU_FEAT10_Data* d_t10_;        ///< host pointer to element data (TYPE_T10)
    GPU_FEAT10_Data* d_data_elem_;  ///< device copy pointer for GPU kernels

    int n_coef_;      ///< number of nodes (= coefficients for T10)
    int n_elem_;      ///< number of elements
    int n_qp_;        ///< quadrature points per element
    int n_shape_;     ///< shape functions per element (= nodes/elem)

    Real h_;           ///< time step
    Real newton_tol_;  ///< Newton convergence tolerance
    int max_newton_;   ///< max Newton iterations

    // -- velocity state on GPU --
    Real *d_v_;       ///< current velocity (3·n_coef)
    Real *d_v_prev_;  ///< velocity from previous step

    // -- previous positions on GPU --
    Real *d_x_prev_, *d_y_prev_, *d_z_prev_;

    // -- stiffness matrix CSR on GPU --
    int* d_K_offsets_;   ///< row offsets (3·n_coef + 1)
    int* d_K_columns_;   ///< column indices
    Real* d_K_values_;   ///< non-zero values (zeroed before each step)
    int K_nnz_;          ///< number of non-zeros

    // -- stiffness sparsity kept on CPU for matrix assembly --
    std::vector<int> K_offsets_cpu_;
    std::vector<int> K_columns_cpu_;

    // -- fixed scalar DOFs (boundary conditions) --
    std::vector<int> fixed_dofs_;

    // -- energy accumulators --
    Real kinetic_energy_ = 0.0;
    Real strain_energy_  = 0.0;
};

}  // namespace tlfea
