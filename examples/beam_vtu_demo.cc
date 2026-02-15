/**
 * Beam VTU Demo - Realistic Engineering Analysis
 *
 * Author: GitHub Copilot Agent
 *
 * This example demonstrates loading a tetrahedral mesh from a VTU file and
 * performing realistic engineering analysis:
 * 1. Load mesh from beam.vtu using MoPhiEssentials' VTU loader
 * 2. Store mesh data in mophi::Mesh class
 * 3. Apply boundary conditions (fix nodes at x ≈ 0)
 * 4. Apply load at the free end (surface where x ≈ 10)
 * 5. Run simulation using TLFEA's solvers
 * 6. Save results to VTK format for visualization
 *
 * This simulates a cantilever beam with one end fixed and a load applied at
 * the free end.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <MoPhiEssentials.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "elements/FEAT10Data.cuh"
#include "solvers/SyncedNesterov.cuh"
#include "utils/cpu_utils.h"
#include "utils/mesh_utils.h"
#include "utils/quadrature_utils.h"

// Material properties for aluminum
const double E = 7e8;      // Young's modulus (Pa) - Aluminum
const double nu = 0.33;    // Poisson's ratio - Aluminum
const double rho0 = 2700;  // Density (kg/m³) - Aluminum

// Simulation parameters
const int N_TIMESTEPS = 100;      // Number of timesteps to simulate
const int OUTPUT_FREQUENCY = 10;  // Output every N timesteps

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "  Beam VTU Demo - Realistic Engineering Analysis" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Initialize MoPhiEssentials logging
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // ==========================================================================
    // Load mesh from VTU file using MoPhiEssentials
    // ==========================================================================

    std::string vtu_filename = "data/meshes/T10/beam.vtu";
    MOPHI_INFO("Loading mesh from: %s", vtu_filename.c_str());

    // Load the VTU file using MoPhiEssentials' VTU loader
    // Note: The beam.vtu file contains 10-node tetrahedral elements (FEAT10/quadratic tets)
    // but the mophi::LoadVtu() function expects 4-node linear tets.
    // We'll use mophi::Mesh to store the mesh and manually parse the VTU for FEAT10 elements.

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_filename);
        MOPHI_INFO("VTU file parsed successfully");
        MOPHI_INFO("Number of nodes loaded: %d", (int)mesh.NumLocalNodes());
    } catch (const std::exception& e) {
        std::cerr << "Error loading VTU file: " << e.what() << std::endl;
        return 1;
    }

    // ==========================================================================
    // Parse FEAT10 elements directly from VTU file
    // ==========================================================================
    // Since mophi::LoadVtu() doesn't support 10-node tets (it only supports 4-node tets),
    // we need to parse the connectivity directly from the VTU file.

    std::cout << "Parsing FEAT10 elements from VTU file..." << std::endl;

    pugi::xml_document doc;
    if (!doc.load_file(vtu_filename.c_str())) {
        std::cerr << "Error: Failed to parse VTU file" << std::endl;
        return 1;
    }

    auto vtk = doc.child("VTKFile");
    auto piece = vtk.child("UnstructuredGrid").child("Piece");
    if (!piece) {
        std::cerr << "Error: Invalid VTU file structure" << std::endl;
        return 1;
    }

    // Parse connectivity for 10-node tets
    pugi::xml_node conn_da, offset_da, types_da;
    for (auto da : piece.child("Cells").children("DataArray")) {
        std::string name = da.attribute("Name").as_string();
        if (name == "connectivity")
            conn_da = da;
        else if (name == "offsets")
            offset_da = da;
        else if (name == "types")
            types_da = da;
    }

    auto connectivity = mophi::ParseDataArray<int>(conn_da);
    auto offsets = mophi::ParseDataArray<int>(offset_da);
    auto types = mophi::ParseDataArray<int>(types_da);

    // Build FEAT10 elements
    std::vector<std::array<int, 10>> feat10_elements;
    size_t start = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        size_t end = (size_t)offsets[i];
        size_t nverts = end - start;

        // Check for 10-node tetrahedral elements
        if (nverts == 10) {
            std::array<int, 10> elem;
            for (size_t j = 0; j < 10; ++j) {
                elem[j] = connectivity[start + j];
            }
            feat10_elements.push_back(elem);
        }
        start = end;
    }

    int n_nodes = static_cast<int>(mesh.NumLocalNodes());
    int n_elems = static_cast<int>(feat10_elements.size());

    if (n_elems == 0) {
        std::cerr << "Error: No 10-node tetrahedral elements found in VTU file" << std::endl;
        return 1;
    }

    std::cout << "Mesh loaded successfully:" << std::endl;
    std::cout << "  Nodes: " << n_nodes << std::endl;
    std::cout << "  FEAT10 Elements: " << n_elems << std::endl;

    // ==========================================================================
    // Convert mophi::Mesh to TLFEA format
    // ==========================================================================

    // Convert nodes to Eigen matrices
    Eigen::MatrixXd nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        nodes(i, 0) = mesh.geom.nodes[i].x();
        nodes(i, 1) = mesh.geom.nodes[i].y();
        nodes(i, 2) = mesh.geom.nodes[i].z();
    }

    // Convert elements to Eigen matrix
    Eigen::MatrixXi elements(n_elems, 10);
    for (int e = 0; e < n_elems; e++) {
        for (int j = 0; j < 10; j++) {
            elements(e, j) = feat10_elements[e][j];
        }
    }

    int total_nodes = n_nodes;

    // ==========================================================================
    // Initialize GPU data structure
    // ==========================================================================

    GPU_FEAT10_Data gpu_t10_data(n_elems, total_nodes);
    gpu_t10_data.Initialize();

    // Extract coordinate vectors from nodes matrix
    Eigen::VectorXd h_x12(total_nodes), h_y12(total_nodes), h_z12(total_nodes);
    for (int i = 0; i < total_nodes; i++) {
        h_x12(i) = nodes(i, 0);
        h_y12(i) = nodes(i, 1);
        h_z12(i) = nodes(i, 2);
    }

    // ==========================================================================
    // Apply boundary conditions: Fix nodes at x ≈ 0 (left end)
    // ==========================================================================

    std::vector<int> fixed_node_indices;
    double x_min = h_x12.minCoeff();
    double x_max = h_x12.maxCoeff();
    double tolerance = 0.1;  // tolerance for boundary detection

    std::cout << "Mesh bounds: x ∈ [" << x_min << ", " << x_max << "]" << std::endl;

    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_min) < tolerance) {
            fixed_node_indices.push_back(i);
        }
    }

    Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
        h_fixed_nodes(i) = fixed_node_indices[i];
    }

    std::cout << "Boundary conditions: Fixed " << fixed_node_indices.size() << " nodes at x ≈ " << x_min << std::endl;
    gpu_t10_data.SetNodalFixed(h_fixed_nodes);

    // ==========================================================================
    // Apply external forces: Load at free end (x ≈ x_max)
    // ==========================================================================

    Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
    h_f_ext.setZero();

    // Find nodes at the free end (x ≈ x_max)
    std::vector<int> loaded_node_indices;
    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_max) < tolerance) {
            loaded_node_indices.push_back(i);
        }
    }

    // Apply a downward force (negative z-direction) at the free end
    double total_force = -1000.0;  // N (downward)
    double force_per_node = total_force / loaded_node_indices.size();

    for (int idx : loaded_node_indices) {
        h_f_ext(3 * idx + 2) = force_per_node;  // Apply force in z-direction
    }

    gpu_t10_data.SetExternalForce(h_f_ext);
    std::cout << "External forces: Applied " << total_force << " N at " << loaded_node_indices.size()
              << " nodes at x ≈ " << x_max << std::endl;

    // ==========================================================================
    // Setup material and element properties
    // ==========================================================================

    // Get quadrature data
    const Eigen::VectorXd& tet5pt_x_host = Quadrature::tet5pt_x;
    const Eigen::VectorXd& tet5pt_y_host = Quadrature::tet5pt_y;
    const Eigen::VectorXd& tet5pt_z_host = Quadrature::tet5pt_z;
    const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

    // Setup element data
    gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host, tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

    gpu_t10_data.SetDensity(rho0);
    gpu_t10_data.SetDamping(0.0, 0.1);  // Add some damping for stability
    gpu_t10_data.SetSVK(E, nu);         // St. Venant-Kirchhoff material model

    std::cout << "Material properties: E=" << std::scientific << E << std::defaultfloat << " Pa, nu=" << nu
              << ", rho=" << static_cast<int>(rho0) << " kg/m³" << std::endl;

    // ==========================================================================
    // Compute reference configuration data
    // ==========================================================================

    gpu_t10_data.CalcDnDuPre();
    gpu_t10_data.CalcMassMatrix();
    gpu_t10_data.CalcConstraintData();
    gpu_t10_data.ConvertToCSR_ConstraintJacT();

    std::cout << "Reference configuration computed" << std::endl;

    // ==========================================================================
    // Setup solver
    // ==========================================================================

    // alpha, rho, inner_tol, outer_tol, max_outer, max_inner, time_step
    SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300, 1.0e-3};

    SyncedNesterovSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
    solver.Setup();
    solver.SetParameters(&params);

    std::cout << "Solver initialized: SyncedNesterov with h=" << std::scientific << params.time_step << std::defaultfloat
              << std::endl;

    // ==========================================================================
    // Run simulation and output results
    // ==========================================================================

    std::cout << "Starting simulation: " << N_TIMESTEPS << " timesteps, output every " << OUTPUT_FREQUENCY << " steps"
              << std::endl;

    // Output initial configuration
    Eigen::VectorXd x12, y12, z12;
    gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

    std::stringstream ss;
    ss << "output_beam_vtu_" << std::setfill('0') << std::setw(5) << 0 << ".vtk";
    bool success = ANCFCPUUtils::WriteFEAT10ToVTK(ss.str(), nodes, elements, x12, y12, z12);
    if (success) {
        std::cout << "Saved initial state to " << ss.str() << std::endl;
    } else {
        std::cerr << "Failed to write initial VTK file" << std::endl;
    }

    // Run simulation loop
    for (int step = 1; step <= N_TIMESTEPS; step++) {
        solver.Solve();

        // Output at specified frequency
        if (step % OUTPUT_FREQUENCY == 0) {
            gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

            // Create filename with timestep
            std::stringstream filename;
            filename << "output_beam_vtu_" << std::setfill('0') << std::setw(5) << step << ".vtk";

            bool write_success = ANCFCPUUtils::WriteFEAT10ToVTK(filename.str(), nodes, elements, x12, y12, z12);
            if (write_success) {
                std::cout << "Step " << step << "/" << N_TIMESTEPS << ": Saved to " << filename.str() << std::endl;
            } else {
                std::cerr << "Step " << step << "/" << N_TIMESTEPS << ": Failed to write VTK file" << std::endl;
            }
        }
    }

    // ==========================================================================
    // Cleanup
    // ==========================================================================

    gpu_t10_data.Destroy();

    std::cout << "=======================================================" << std::endl;
    std::cout << "  Simulation Complete!" << std::endl;
    std::cout << "  Output files: output_beam_vtu_*.vtk" << std::endl;
    std::cout << "  Visualize with: paraview output_beam_vtu_*.vtk" << std::endl;
    std::cout << "=======================================================" << std::endl;

    return 0;
}
