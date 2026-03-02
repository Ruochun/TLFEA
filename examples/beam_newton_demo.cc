/**
 * Beam Newton Solver Demo
 *
 * Demonstrates the Newton-Raphson implicit solver on a cantilever
 * beam (FEAT10 TET10 elements).
 *
 * Setup:
 *   - Mesh:     data/meshes/T10/beam.vtu  (TET10 elements)
 *   - Material: St. Venant-Kirchhoff (SVK), aluminium-like properties
 *   - BCs:      All nodes at x ≈ x_min are fully fixed.
 *   - Load:     Uniform downward (-z) force distributed over nodes
 *               at x ≈ x_max.
 *
 * Outputs:
 *   - VTK files for ParaView: output_newton_beam_XXXXX.vtk
 *   - Console: time, kinetic energy, strain energy, total energy
 *              printed every output frame.
 */

#include <cuda_runtime.h>

#include <MoPhiEssentials.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "elements/FEAT10Data.cuh"
#include "solvers/NewtonSolver.cuh"
#include "types.h"
#include "utils/cpu_utils.h"
#include "utils/mesh_utils.h"
#include "utils/quadrature_utils.h"

using namespace tlfea;

// ============================================================
// Material properties (aluminium-like)
// ============================================================
const Real E    = 7e10;   // Young's modulus [Pa]
const Real nu   = 0.33;   // Poisson's ratio
const Real rho0 = 2700.0; // density [kg/m³]

// ============================================================
// Simulation parameters
// ============================================================
const Real TOTAL_TIME      = 1.0;
const Real STEP_SIZE       = 1e-3;
const int  N_TIMESTEPS     = static_cast<int>(TOTAL_TIME / STEP_SIZE);
const int  FPS             = 50;
const int  OUTPUT_FREQ     = static_cast<int>(1.0 / (STEP_SIZE * FPS));
const int  TOTAL_FRAMES    = static_cast<int>(TOTAL_TIME * FPS);

int main() {
    std::cout << "=================================================\n"
              << "  Beam Newton Solver Demo\n"
              << "=================================================\n";

    // =========================================================
    // Load mesh
    // =========================================================
    const std::string vtu_file = "data/meshes/T10/beam.vtu";
    std::cout << "Loading mesh: " << vtu_file << "\n";

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading VTU: " << e.what() << "\n";
        return 1;
    }

    if (mesh.NumOwnedTet10s() == 0) {
        std::cerr << "No TET10 elements found in " << vtu_file << "\n";
        return 1;
    }

    int n_nodes = mesh.NumLocalNodes();
    int n_elems = mesh.NumOwnedTet10s();
    std::cout << "  Nodes: " << n_nodes
              << "  TET10 elements: " << n_elems << "\n";

    // Convert geometry
    MatrixXR nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i) {
        nodes(i, 0) = mesh.geom.nodes[i].x();
        nodes(i, 1) = mesh.geom.nodes[i].y();
        nodes(i, 2) = mesh.geom.nodes[i].z();
    }
    MatrixXi elements(n_elems, 10);
    for (int i = 0; i < n_elems; ++i)
        for (int j = 0; j < 10; ++j)
            elements(i, j) = mesh.topo.tet10s[i][j];

    // =========================================================
    // Setup GPU element data
    // =========================================================
    VectorXR h_x(n_nodes), h_y(n_nodes), h_z(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        h_x(i) = nodes(i, 0);
        h_y(i) = nodes(i, 1);
        h_z(i) = nodes(i, 2);
    }

    GPU_FEAT10_Data gpu_data(n_elems, n_nodes);
    gpu_data.Initialize();

    // Identify fixed / loaded nodes
    Real x_min = h_x.minCoeff(), x_max = h_x.maxCoeff();
    Real tol   = 0.1 * (x_max - x_min);

    std::vector<int> fixed_nodes, loaded_nodes;
    for (int i = 0; i < n_nodes; ++i) {
        if (std::abs(h_x(i) - x_min) < tol) fixed_nodes.push_back(i);
        if (std::abs(h_x(i) - x_max) < tol) loaded_nodes.push_back(i);
    }
    std::cout << "  Fixed nodes:  " << fixed_nodes.size()
              << "  Loaded nodes: " << loaded_nodes.size() << "\n";

    // Build fixed DOF list (all 3 DOFs per fixed node)
    std::vector<int> fixed_dofs;
    for (int n : fixed_nodes) {
        fixed_dofs.push_back(3*n + 0);
        fixed_dofs.push_back(3*n + 1);
        fixed_dofs.push_back(3*n + 2);
    }
    std::sort(fixed_dofs.begin(), fixed_dofs.end());

    // Set fixed nodes on GPU (for constraint book-keeping)
    VectorXi h_fixed(static_cast<int>(fixed_nodes.size()));
    for (int i = 0; i < (int)fixed_nodes.size(); ++i)
        h_fixed(i) = fixed_nodes[i];
    gpu_data.SetNodalFixed(h_fixed);

    // External force: -z load on free end
    const Real downward_load   = -1000.0; // [N], negative = downward in z
    const Real load_per_node = downward_load / (Real)loaded_nodes.size();
    VectorXR h_f_ext(n_nodes * 3);
    h_f_ext.setZero();
    for (int n : loaded_nodes)
        h_f_ext(3*n + 2) = load_per_node;
    gpu_data.SetExternalForce(h_f_ext);

    std::cout << "  Load: " << downward_load << " N distributed over "
              << loaded_nodes.size() << " nodes at x≈" << x_max << "\n";

    // Setup element (quadrature, connectivity, material)
    gpu_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                   Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                   h_x, h_y, h_z, elements);
    gpu_data.SetDensity(rho0);
    gpu_data.SetDamping(0.0, 0.0);
    gpu_data.SetSVK(E, nu);

    gpu_data.CalcDnDuPre();
    gpu_data.CalcMassMatrix();
    gpu_data.CalcConstraintData();
    gpu_data.ConvertToCSR_ConstraintJacT();

    std::cout << "  Material: E=" << E/1e9 << " GPa, nu=" << nu
              << ", rho=" << rho0 << " kg/m³\n";

    // =========================================================
    // Setup Newton solver
    // =========================================================
    NewtonSolverParams params;
    params.time_step       = STEP_SIZE;
    params.newton_tol      = 1e-6;
    params.max_newton_iters = 20;

    NewtonSolver solver(&gpu_data, elements, fixed_dofs);
    solver.Setup();
    solver.SetParameters(&params);

    std::cout << "  Solver: Newton-Raphson, h=" << STEP_SIZE
              << ", tol=" << params.newton_tol
              << ", max_iters=" << params.max_newton_iters << "\n";

    // =========================================================
    // Simulation loop
    // =========================================================
    std::cout << "\n"
              << std::setw(10) << "Time [s]"
              << std::setw(18) << "KE [J]"
              << std::setw(18) << "SE [J]"
              << std::setw(18) << "Total E [J]"
              << "\n"
              << std::string(64, '-') << "\n";

    int curr_frame = 0;

    // Write initial state
    {
        VectorXR x, y, z;
        gpu_data.RetrievePositionToCPU(x, y, z);
        std::ostringstream fn;
        fn << "output_newton_beam_" << std::setfill('0') << std::setw(5) << 0 << ".vtk";
        ANCFCPUUtils::WriteFEAT10ToVTK(fn.str(), nodes, elements, x, y, z);
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << 0.0
                  << std::setw(18) << std::scientific << std::setprecision(6) << 0.0
                  << std::setw(18) << 0.0
                  << std::setw(18) << 0.0
                  << "  [frame 0 → " << fn.str() << "]\n";
    }

    for (int step = 1; step <= N_TIMESTEPS; ++step) {
        solver.Solve();

        if (step % OUTPUT_FREQ == 0) {
            ++curr_frame;
            Real t = step * STEP_SIZE;

            VectorXR x, y, z;
            gpu_data.RetrievePositionToCPU(x, y, z);

            std::ostringstream fn;
            fn << "output_newton_beam_"
               << std::setfill('0') << std::setw(5) << curr_frame << ".vtk";
            ANCFCPUUtils::WriteFEAT10ToVTK(fn.str(), nodes, elements, x, y, z);

            Real KE = solver.GetKineticEnergy();
            Real SE = solver.GetStrainEnergy();
            Real TE = solver.GetTotalEnergy();

            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << t
                      << std::setw(18) << std::scientific << std::setprecision(6) << KE
                      << std::setw(18) << SE
                      << std::setw(18) << TE
                      << "  [frame " << curr_frame << "/" << TOTAL_FRAMES
                      << " → " << fn.str() << "]\n";
        }
    }

    // =========================================================
    // Cleanup
    // =========================================================
    gpu_data.Destroy();

    std::cout << "\n=================================================\n"
              << "  Simulation complete!\n"
              << "  ParaView output: output_newton_beam_*.vtk\n"
              << "=================================================\n";
    return 0;
}
