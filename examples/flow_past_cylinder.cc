/**
 * flow_past_cylinder.cc
 *
 * Transient Navier-Stokes simulation of incompressible flow past a cylinder.
 *
 * Physics
 * -------
 * 3-D flow past a circular cylinder (DFG benchmark-style geometry):
 *   • Channel:   x ∈ [0, 2.2],  y ∈ [0, 0.41],  z ∈ [0, 0.41]
 *   • Cylinder:  axis along z, centre (x=0.2, y=0.2), radius r=0.05
 *   • Inlet  (x=0):   uniform x-velocity  U_in = 1.5 m/s
 *   • Walls  (y=0, y=H, z=0, z=H):   no-slip
 *   • Cylinder surface (r ≈ 0.05 from axis): no-slip
 *   • Outlet (x=2.2):  natural outflow (do-nothing / zero-stress)
 *   • Fluid: ρ = 1.0 kg/m³,  μ = 0.001 Pa·s  →  Re ≈ 150 (U_in*D/ν)
 *
 * Solver
 * ------
 * NavierStokesSUPGPSPGSolver with backward-Euler time integration,
 * SUPG and PSPG stabilisation, and BiCGSTAB on the GPU.
 *
 * Output
 * ------
 * VTK legacy files written every OUTPUT_INTERVAL time steps to
 * output_cylinder_XXXXX.vtk, containing the node coordinates, velocity
 * vectors, and pressure scalars.
 */

#include <cuda_runtime.h>

#include <MoPhiEssentials.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/solvers/NavierStokesSUPGPSPGSolver.cuh"
#include "../src/types.h"

using namespace tlfea;

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
static constexpr Real U_INLET      = 1.5;     ///< Inlet x-velocity [m/s]
static constexpr Real RHO          = 1.0;     ///< Density [kg/m³]
static constexpr Real MU           = 0.001;   ///< Dynamic viscosity [Pa·s]
static constexpr Real DT           = 5e-4;    ///< Time step [s]
static constexpr int  N_STEPS      = 2000;    ///< Number of time steps
static constexpr int  OUTPUT_INTERVAL = 200;  ///< Output every N steps

// Geometry
static constexpr Real CYL_CX = 0.2;           ///< Cylinder axis x [m]
static constexpr Real CYL_CY = 0.2;           ///< Cylinder axis y [m]
static constexpr Real CYL_R  = 0.05;          ///< Cylinder radius [m]
static constexpr Real CYL_R_TOL = 0.007;      ///< Tolerance for surface detection

static constexpr Real H      = 0.41;           ///< Channel height/width [m]
static constexpr Real WALL_TOL = 5e-4;         ///< Tolerance for wall detection

// ---------------------------------------------------------------------------
// Write a VTK legacy file with velocity and pressure fields
// ---------------------------------------------------------------------------
static void WriteFluidVTK(const std::string&  filename,
                           const MatrixXR&     nodes,
                           const MatrixXi&     elements,
                           const VectorXR&     vel,    // size 3*n_nodes
                           const VectorXR&     pres)   // size n_nodes
{
    int n_nodes = static_cast<int>(nodes.rows());
    int n_elems = static_cast<int>(elements.rows());

    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Could not open " << filename << " for writing\n";
        return;
    }

    f << "# vtk DataFile Version 3.0\n";
    f << "Flow past cylinder\n";
    f << "ASCII\n";
    f << "DATASET UNSTRUCTURED_GRID\n";

    f << "POINTS " << n_nodes << " double\n";
    for (int i = 0; i < n_nodes; i++)
        f << nodes(i,0) << " " << nodes(i,1) << " " << nodes(i,2) << "\n";

    f << "\nCELLS " << n_elems << " " << (n_elems * 5) << "\n";
    for (int e = 0; e < n_elems; e++) {
        f << "4";
        for (int a = 0; a < 4; a++) f << " " << elements(e, a);
        f << "\n";
    }

    f << "\nCELL_TYPES " << n_elems << "\n";
    for (int e = 0; e < n_elems; e++) f << "10\n";

    f << "\nPOINT_DATA " << n_nodes << "\n";
    f << "VECTORS velocity double\n";
    for (int i = 0; i < n_nodes; i++)
        f << vel(3*i+0) << " " << vel(3*i+1) << " " << vel(3*i+2) << "\n";

    f << "SCALARS pressure double 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int i = 0; i < n_nodes; i++)
        f << pres(i) << "\n";

    f.close();
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=======================================================\n"
              << "  Transient Navier-Stokes  –  Flow Past Cylinder\n"
              << "  SUPG/PSPG stabilisation, TET4, GPU\n"
              << "=======================================================\n";

    // -----------------------------------------------------------------------
    // Load mesh
    // -----------------------------------------------------------------------
    const std::string vtu_path = "data/meshes/T4/flow_past_cylinder.vtu";
    std::cout << "Loading mesh: " << vtu_path << "\n";

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << "\n";
        return 1;
    }
    if (mesh.NumOwnedTets() == 0) {
        std::cerr << "No TET4 elements found in " << vtu_path << "\n";
        return 1;
    }

    const int n_nodes = mesh.NumLocalNodes();
    const int n_elems = static_cast<int>(mesh.NumOwnedTets());
    std::cout << "  Nodes: " << n_nodes << "   TET4 elements: " << n_elems << "\n";

    // -----------------------------------------------------------------------
    // Build node/element arrays
    // -----------------------------------------------------------------------
    MatrixXR nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        nodes(i, 0) = mesh.geom.nodes[i].x();
        nodes(i, 1) = mesh.geom.nodes[i].y();
        nodes(i, 2) = mesh.geom.nodes[i].z();
    }

    MatrixXi elements(n_elems, 4);
    for (int e = 0; e < n_elems; e++)
        for (int a = 0; a < 4; a++)
            elements(e, a) = mesh.topo.tets[e][a];

    // -----------------------------------------------------------------------
    // Identify boundary node sets geometrically
    // -----------------------------------------------------------------------
    std::vector<int> inlet_nodes, wall_nodes, cyl_nodes;

    for (int i = 0; i < n_nodes; i++) {
        const Real x = nodes(i, 0);
        const Real y = nodes(i, 1);
        const Real z = nodes(i, 2);

        // Inlet: x ≈ 0
        if (x < WALL_TOL) {
            inlet_nodes.push_back(i);
            continue;
        }

        // Channel walls: y or z at boundary
        if (y < WALL_TOL || y > H - WALL_TOL ||
            z < WALL_TOL || z > H - WALL_TOL) {
            wall_nodes.push_back(i);
            continue;
        }

        // Cylinder surface: distance from cylinder axis (x-y plane)
        Real dist = std::sqrt((x - CYL_CX)*(x - CYL_CX) + (y - CYL_CY)*(y - CYL_CY));
        if (std::abs(dist - CYL_R) < CYL_R_TOL) {
            cyl_nodes.push_back(i);
        }
    }

    std::cout << "  Inlet nodes:          " << inlet_nodes.size() << "\n";
    std::cout << "  Wall nodes:           " << wall_nodes.size()  << "\n";
    std::cout << "  Cylinder surface nodes: " << cyl_nodes.size() << "\n";

    // -----------------------------------------------------------------------
    // Solver setup
    // -----------------------------------------------------------------------
    NavierStokesSUPGPSPGParams params;
    params.rho           = RHO;
    params.mu            = MU;
    params.dt            = DT;
    params.bicgstab_tol  = 1e-6;
    params.max_bicgstab  = 3000;

    std::cout << "\nFluid properties:\n"
              << "  Density ρ = " << RHO << " kg/m³\n"
              << "  Viscosity μ = " << MU << " Pa·s\n"
              << "  Re ≈ " << std::fixed << std::setprecision(1)
              << (RHO * U_INLET * 2.0 * CYL_R / MU) << "\n"
              << "  dt = " << DT << " s   steps = " << N_STEPS << "\n";

    NavierStokesSUPGPSPGSolver solver(nodes, elements, params);

    // -------------------------------------------------------------------
    // Set boundary conditions
    //   1. No-slip on walls + cylinder (SetNoSlipBC clears previous BCs)
    //   2. Inlet: uniform x-velocity (SetDirichletVelocity ADDS to the list)
    //   Note: inlet nodes are geometrically disjoint from wall/cylinder nodes
    //   (the geometry loop uses 'continue' after detecting inlet nodes).
    // -------------------------------------------------------------------
    {
        // Combine wall + cylinder no-slip (clears any previously set BCs)
        std::vector<int> noslip;
        noslip.insert(noslip.end(), wall_nodes.begin(), wall_nodes.end());
        noslip.insert(noslip.end(), cyl_nodes.begin(),  cyl_nodes.end());
        solver.SetNoSlipBC(noslip);
    }

    // Inlet: uniform x-velocity (appends to BC set; inlet and noslip nodes are disjoint)
    solver.SetDirichletVelocity(inlet_nodes, U_INLET, Real(0), Real(0));

    // -----------------------------------------------------------------------
    // Initial condition: uniform flow field (helps convergence)
    // -----------------------------------------------------------------------
    {
        VectorXR vel0(n_nodes * 3);
        vel0.setZero();
        for (int i = 0; i < n_nodes; i++)
            vel0(3*i + 0) = U_INLET;  // start with uniform flow
        solver.SetInitialVelocity(vel0);
    }

    // -----------------------------------------------------------------------
    // Write initial state
    // -----------------------------------------------------------------------
    {
        VectorXR vel, pres;
        solver.GetVelocity(vel);
        solver.GetPressure(pres);
        WriteFluidVTK("output_cylinder_00000.vtk", nodes, elements, vel, pres);
        std::cout << "\nInitial state written to output_cylinder_00000.vtk\n";
    }

    // -----------------------------------------------------------------------
    // Time integration loop
    // -----------------------------------------------------------------------
    std::cout << "\n--- Time Integration ---\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time [s]"
              << std::setw(12) << "BiCGSTAB"
              << std::setw(14) << "Rel. Residual"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int step = 1; step <= N_STEPS; ++step) {
        solver.Step();

        if (step % OUTPUT_INTERVAL == 0 || step == N_STEPS) {
            std::cout << std::setw(8)  << step
                      << std::setw(12) << std::fixed << std::setprecision(5) << solver.GetTime()
                      << std::setw(12) << solver.GetLastIterCount()
                      << std::setw(14) << std::scientific << std::setprecision(3)
                      << solver.GetLastResidual()
                      << "\n";

            // Write VTK output
            VectorXR vel, pres;
            solver.GetVelocity(vel);
            solver.GetPressure(pres);

            std::string fname = "output_cylinder_" +
                std::to_string(10000 + step).substr(1) + ".vtk";
            WriteFluidVTK(fname, nodes, elements, vel, pres);
        }
    }

    // -----------------------------------------------------------------------
    // Final statistics
    // -----------------------------------------------------------------------
    {
        VectorXR vel, pres;
        solver.GetVelocity(vel);
        solver.GetPressure(pres);

        // Maximum velocity magnitude
        Real umax = 0.0;
        for (int i = 0; i < n_nodes; i++) {
            Real umag = std::sqrt(vel(3*i)*vel(3*i) + vel(3*i+1)*vel(3*i+1) + vel(3*i+2)*vel(3*i+2));
            if (umag > umax) umax = umag;
        }

        Real p_max = pres.maxCoeff();
        Real p_min = pres.minCoeff();

        std::cout << "\n--- Final State (t = " << std::fixed << std::setprecision(4)
                  << solver.GetTime() << " s) ---\n"
                  << "  Max velocity magnitude: " << std::scientific << std::setprecision(4)
                  << umax << " m/s\n"
                  << "  Pressure range: [" << p_min << ", " << p_max << "] Pa\n";
    }

    std::cout << "\n=======================================================\n"
              << "  Simulation Complete\n"
              << "=======================================================\n";
    return 0;
}
