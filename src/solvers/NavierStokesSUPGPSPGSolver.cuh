#pragma once
/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    NavierStokesSUPGPSPGSolver.cuh
 * Brief:   Declares NavierStokesSUPGPSPGSolver — a self-contained,
 *          GPU-accelerated transient incompressible Navier-Stokes solver
 *          for unstructured TET4 meshes.
 *
 *          Formulation
 *          -----------
 *          Solves the semi-discrete (backward-Euler in time) linearised
 *          incompressible Navier-Stokes equations (Oseen problem):
 *
 *            ρ/dt (u^{n+1} - u^n) + ρ (u^n · ∇) u^{n+1}
 *                - μ Δ u^{n+1} + ∇ p^{n+1} = f
 *            ∇ · u^{n+1}  =  0
 *
 *          Equal-order P1/P1 velocity–pressure interpolation is stabilised
 *          with the classical SUPG (Streamline-Upwind Petrov-Galerkin) and
 *          PSPG (Pressure-Stabilised Petrov-Galerkin) terms; see e.g.
 *          Tezduyar et al. (1992).  For TET4 the second spatial derivatives
 *          vanish (constant-strain elements), so only the time and convection
 *          contributions appear in the stabilisation residual.
 *
 *          DOF layout
 *          ----------
 *          Interleaved per node:  4*i+0 = u_x,  4*i+1 = u_y,
 *                                 4*i+2 = u_z,  4*i+3 = p
 *          Total system size: n_dof = 4 * n_nodes
 *
 *          Linear solver
 *          -------------
 *          The assembled system matrix is non-symmetric (convection term),
 *          so a BiCGSTAB iteration is used with cuBLAS (dot/axpy/scal) and
 *          cuSPARSE (SpMV).
 *
 *          Design
 *          ------
 *          The solver owns all its GPU data (mesh arrays, state vectors,
 *          system matrix).  It does NOT inherit from any element class;
 *          the TET4 geometry is handled entirely internally.  This follows
 *          the project principle of algorithm-specific, self-contained
 *          solver classes that prefer composition over inheritance.
 *==============================================================
 *==============================================================*/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <MoPhiEssentials.h>
#include <string>
#include <vector>

#include "../types.h"

namespace tlfea {

// ---------------------------------------------------------------------------
// Parameter struct for the NS solver.
// ---------------------------------------------------------------------------
struct NavierStokesSUPGPSPGParams {
    Real rho           = 1.0;      ///< Fluid density [kg/m³]
    Real mu            = 1e-3;     ///< Dynamic viscosity [Pa·s]
    Real dt            = 1e-3;     ///< Time step [s]
    Real bicgstab_tol  = 1e-6;     ///< BiCGSTAB relative residual tolerance
    int  max_bicgstab  = 5000;     ///< Maximum BiCGSTAB iterations per step
};

// ---------------------------------------------------------------------------
// NavierStokesSUPGPSPGSolver
// ---------------------------------------------------------------------------
class NavierStokesSUPGPSPGSolver {
  public:
    /**
     * Construct the solver from mesh data already held on the host.
     *
     * @param nodes       Node coordinates, shape [n_nodes × 3], row-major.
     * @param elements    TET4 element connectivity, shape [n_elems × 4], row-major.
     * @param params      Fluid properties and solver settings.
     */
    NavierStokesSUPGPSPGSolver(const MatrixXR&                     nodes,
                                const MatrixXi&                     elements,
                                const NavierStokesSUPGPSPGParams&   params);
    ~NavierStokesSUPGPSPGSolver();

    // -------------------------------------------------------------------
    // Boundary conditions
    // -------------------------------------------------------------------

    /**
     * Prescribe a no-slip (zero-velocity) Dirichlet condition at the listed nodes.
     * May be called repeatedly before the first Step(); later calls replace
     * the entire BC list.
     */
    void SetNoSlipBC(const std::vector<int>& node_ids);

    /**
     * Prescribe a uniform Dirichlet velocity at the listed nodes.
     * This call APPENDS to the Dirichlet set (does not clear previous BCs).
     * If a node already has a BC set via SetNoSlipBC, adding it here
     * creates a duplicate entry; the last value for each node wins when
     * the BC arrays are rebuilt (last writer wins per DOF in the rebuild loop).
     * Nodes should be disjoint from the SetNoSlipBC set to avoid conflicts.
     */
    void SetDirichletVelocity(const std::vector<int>& node_ids, Real ux, Real uy, Real uz);

    /**
     * Set the initial velocity field (host-side, size 3*n_nodes, order x,y,z).
     * If not called, the initial state is zero velocity and zero pressure.
     */
    void SetInitialVelocity(const VectorXR& vel);

    // -------------------------------------------------------------------
    // Time integration
    // -------------------------------------------------------------------

    /**
     * Advance the solution by one time step dt (set in params).
     * This method:
     *   1. Assembles the stabilised system matrix and RHS on the GPU.
     *   2. Applies Dirichlet BCs.
     *   3. Solves the linear system with BiCGSTAB.
     *   4. Updates the internal state vectors.
     */
    void Step();

    // -------------------------------------------------------------------
    // Result access
    // -------------------------------------------------------------------

    /** Copy the current velocity field to the host (size 3*n_nodes, [x,y,z] interleaved). */
    void GetVelocity(VectorXR& vel) const;

    /** Copy the current pressure field to the host (size n_nodes). */
    void GetPressure(VectorXR& pres) const;

    /** Elapsed simulated time [s]. */
    Real GetTime() const { return current_time_; }

    /** Number of BiCGSTAB iterations in the most recent Step(). */
    int  GetLastIterCount() const { return last_iter_; }

    /** Relative residual at the end of the most recent Step(). */
    Real GetLastResidual() const { return last_residual_; }

    /** Mesh node count. */
    int GetNumNodes() const { return n_nodes_; }

    /** Mesh element count. */
    int GetNumElems() const { return n_elems_; }

  private:
    // ----------------------------------------------------------------
    // Internal helper stages
    // ----------------------------------------------------------------
    void PrecomputeGeometry();
    void BuildCSRPattern();
    void AssembleSystem();
    void ApplyDirichletBCs();
    void SolveBiCGSTAB();
    void UpdateState();

    // ----------------------------------------------------------------
    // Mesh and solver dimensions
    // ----------------------------------------------------------------
    int n_nodes_;
    int n_elems_;
    int n_dof_;    ///< 4 * n_nodes (interleaved vx,vy,vz,p)

    // ----------------------------------------------------------------
    // Parameters
    // ----------------------------------------------------------------
    NavierStokesSUPGPSPGParams params_;

    // ----------------------------------------------------------------
    // GPU mesh geometry (allocated once, never changed)
    // ----------------------------------------------------------------
    mophi::DualArray<Real>  da_nodes_;       ///< node coords (n_nodes * 3)
    mophi::DualArray<int>   da_connect_;     ///< connectivity (n_elems * 4)
    mophi::DualArray<Real>  da_gradN_;       ///< shape gradients (n_elems * 4 * 3)
    mophi::DualArray<Real>  da_detJ_;        ///< Jacobian determinants (n_elems)
    mophi::DualArray<Real>  da_h_elem_;      ///< element length scales  (n_elems)

    Real* d_nodes_   = nullptr;
    int*  d_connect_ = nullptr;
    Real* d_gradN_   = nullptr;
    Real* d_detJ_    = nullptr;
    Real* d_h_elem_  = nullptr;

    // ----------------------------------------------------------------
    // Dirichlet BC data (host lists, applied each step)
    // ----------------------------------------------------------------
    std::vector<int>  dirichlet_node_ids_;   ///< nodes with velocity BC
    std::vector<Real> dirichlet_vel_;        ///< corresponding (ux,uy,uz) per node

    // Device-side BC flag / value arrays (rebuilt when BC changes)
    mophi::DualArray<int>  da_is_bc_dof_;    ///< 1 if DOF i is a velocity Dirichlet DOF
    mophi::DualArray<Real> da_bc_val_;       ///< prescribed value at DOF i (0 if not BC)
    int*  d_is_bc_dof_ = nullptr;
    Real* d_bc_val_    = nullptr;
    bool  bc_needs_rebuild_ = true;

    // ----------------------------------------------------------------
    // State vectors  (length n_dof; interleaved vx,vy,vz,p per node)
    // ----------------------------------------------------------------
    mophi::DualArray<Real> da_sol_;     ///< current solution  [u^{n+1}]
    mophi::DualArray<Real> da_sol_prev_;///< previous velocity [u^n] (3*n_nodes only)
    Real* d_sol_      = nullptr;
    Real* d_sol_prev_ = nullptr;

    // ----------------------------------------------------------------
    // System matrix  (CSR, n_dof × n_dof)
    // ----------------------------------------------------------------
    mophi::DualArray<int>  da_K_offsets_;
    mophi::DualArray<int>  da_K_columns_;
    mophi::DualArray<Real> da_K_values_;
    int*  d_K_offsets_ = nullptr;
    int*  d_K_columns_ = nullptr;
    Real* d_K_values_  = nullptr;
    int   K_nnz_       = 0;
    bool  pattern_built_ = false;

    // ----------------------------------------------------------------
    // RHS and BiCGSTAB workspace  (all length n_dof)
    // ----------------------------------------------------------------
    mophi::DualArray<Real> da_rhs_;
    mophi::DualArray<Real> da_x_;     ///< current iterate
    mophi::DualArray<Real> da_r_;     ///< residual
    mophi::DualArray<Real> da_rhat_;  ///< shadow residual
    mophi::DualArray<Real> da_p_;
    mophi::DualArray<Real> da_v_;
    mophi::DualArray<Real> da_s_;
    mophi::DualArray<Real> da_t_;

    Real* d_rhs_  = nullptr;
    Real* d_x_    = nullptr;
    Real* d_r_    = nullptr;
    Real* d_rhat_ = nullptr;
    Real* d_p_    = nullptr;
    Real* d_v_    = nullptr;
    Real* d_s_    = nullptr;
    Real* d_t_    = nullptr;

    // ----------------------------------------------------------------
    // cuBLAS / cuSPARSE handles
    // ----------------------------------------------------------------
    cublasHandle_t  cublas_   = nullptr;
    cusparseHandle_t cusparse_ = nullptr;

    // ----------------------------------------------------------------
    // Diagnostics
    // ----------------------------------------------------------------
    Real current_time_  = 0.0;
    int  last_iter_     = 0;
    Real last_residual_ = 0.0;
};

}  // namespace tlfea
