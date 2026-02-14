/**
 * Cantilever Beam Simulation with Physics
 *
 * Author: Github Copilot Agent
 *
 * This example demonstrates a realistic use case of the TLFEA framework:
 * 1. Load a tetrahedral mesh (FEAT10 elements) representing a cantilever beam
 * 2. Apply boundary conditions (fix one end of the beam)
 * 3. Apply a realistic load (gravity acting on the entire beam)
 * 4. Run a dynamic simulation over multiple timesteps
 * 5. Export the results to VTK files that can be visualized in ParaView or similar tools
 *
 * This simulates a cantilever beam fixed at one end (z=0 plane) with gravity
 * acting downward (-z direction). The beam deflects over time under its own weight.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <MoPhiEssentials.h>
#include <sstream>

#include "elements/FEAT10Data.cuh"
#include "solvers/SyncedNesterov.cuh"
#include "utils/cpu_utils.h"
#include "utils/mesh_utils.h"
#include "utils/quadrature_utils.h"

// Material properties for aluminum
const double E    = 7e10;   // Young's modulus (Pa) - Aluminum
const double nu   = 0.33;   // Poisson's ratio - Aluminum
const double rho0 = 2700;   // Density (kg/m³) - Aluminum

// Simulation parameters
const int N_TIMESTEPS = 100;  // Number of timesteps to simulate
const int OUTPUT_FREQUENCY = 10;  // Output every N timesteps

int main() {
  // Initialize MoPhiEssentials logging
  mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);
  
  MOPHI_INFO("=======================================================");
  MOPHI_INFO("  Cantilever Beam Simulation with Physics");
  MOPHI_INFO("=======================================================");
  
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/cube.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements("data/meshes/T10/cube.1.ele",
                                                   elements);

  MOPHI_INFO("Mesh loaded: %d nodes, %d elements", n_nodes, n_elems);

  // Initialize GPU data structure
  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

  // ==========================================================================
  // Apply boundary conditions: Fix nodes at z=0 (one end of the beam)
  // ==========================================================================

  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_z12.size(); ++i) {
    if (std::abs(h_z12(i)) < 1e-8) {  // tolerance for floating point
      fixed_node_indices.push_back(i);
    }
  }

  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  MOPHI_INFO("Boundary conditions: Fixed %zu nodes at z=0", fixed_node_indices.size());
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // ==========================================================================
  // Apply external forces: Gravity acting on the entire beam
  // ==========================================================================

  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();
  
  // Apply gravity force (F = m * g) to all nodes
  // We'll compute mass per node later and apply gravity based on that
  // For now, apply a uniform gravity load in the -z direction
  double gravity = -9.81;  // m/s² (negative because downward)
  double mass_per_node = rho0 * (1.0 * 1.0 * 1.0) / n_nodes;  // Approximate mass distribution
  
  for (int i = 0; i < n_nodes; i++) {
    h_f_ext(3 * i + 2) = mass_per_node * gravity;  // Apply force in z-direction
  }
  
  gpu_t10_data.SetExternalForce(h_f_ext);
  MOPHI_INFO("External forces: Applied gravity (g = %.2f m/s²)", gravity);

  // ==========================================================================
  // Setup material and element properties
  // ==========================================================================

  // Get quadrature data
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  // Setup element data
  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

  gpu_t10_data.SetDensity(rho0);
  gpu_t10_data.SetDamping(0.0, 0.1);  // Add some damping for stability
  gpu_t10_data.SetSVK(E, nu);  // St. Venant-Kirchhoff material model

  MOPHI_INFO("Material properties: E=%.2e Pa, nu=%.2f, rho=%.0f kg/m³", E, nu, rho0);

  // ==========================================================================
  // Compute reference configuration data
  // ==========================================================================

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  
  MOPHI_INFO("Reference configuration computed");

  // ==========================================================================
  // Setup solver
  // ==========================================================================

  SyncedNesterovParams params = {
    1.0e-8,  // h (timestep)
    1e14,    // penalty stiffness
    1.0e-6,  // residual tolerance
    1.0e-6,  // correction tolerance
    5,       // min iterations
    300,     // max iterations
    1.0e-3   // Nesterov momentum parameter
  };
  
  SyncedNesterovSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  MOPHI_INFO("Solver initialized: SyncedNesterov with h=%.2e", params.h);

  // ==========================================================================
  // Run simulation and output results
  // ==========================================================================

  MOPHI_INFO("Starting simulation: %d timesteps, output every %d steps", 
             N_TIMESTEPS, OUTPUT_FREQUENCY);

  // Output initial configuration
  Eigen::VectorXd x12, y12, z12;
  gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);
  
  std::stringstream ss;
  ss << "output_beam_" << std::setfill('0') << std::setw(5) << 0 << ".vtk";
  bool success = ANCFCPUUtils::WriteFEAT10ToVTK(ss.str(), nodes, elements, x12, y12, z12);
  if (success) {
    MOPHI_INFO("Saved initial state to %s", ss.str().c_str());
  } else {
    MOPHI_ERROR("Failed to write initial VTK file");
  }

  // Run simulation loop
  for (int step = 1; step <= N_TIMESTEPS; step++) {
    solver.Solve();
    
    // Output at specified frequency
    if (step % OUTPUT_FREQUENCY == 0) {
      gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);
      
      // Create filename with timestep
      std::stringstream filename;
      filename << "output_beam_" << std::setfill('0') << std::setw(5) << step << ".vtk";
      
      bool write_success = ANCFCPUUtils::WriteFEAT10ToVTK(filename.str(), nodes, elements, 
                                                          x12, y12, z12);
      if (write_success) {
        MOPHI_INFO("Step %d/%d: Saved to %s", step, N_TIMESTEPS, filename.str().c_str());
      } else {
        MOPHI_ERROR("Step %d/%d: Failed to write VTK file", step, N_TIMESTEPS);
      }
    }
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  gpu_t10_data.Destroy();

  MOPHI_INFO("=======================================================");
  MOPHI_INFO("  Simulation Complete!");
  MOPHI_INFO("  Output files: output_beam_*.vtk");
  MOPHI_INFO("  Visualize with: paraview output_beam_*.vtk");
  MOPHI_INFO("=======================================================");

  return 0;
}
