# AGENTS.md — Guidance for Future Development Using AI Agents

This document summarises the design philosophy and conventions of TLFEA so
that AI coding agents can extend the project consistently and correctly.

---

## 1. Project Overview

TLFEA is a GPU-accelerated finite-element analysis framework written in C++17
and CUDA.  Its primary target is high-performance scientific computing; every
design decision is made with execution speed, memory efficiency, and
maintainability in mind.

The project contains:
- **Element implementations** (`src/elements/`) — one self-contained class per
  element type (shape × interpolation order).
- **Solver implementations** (`src/solvers/`) — one self-contained class per
  algorithm (e.g. linear static CG, AdamW, Nesterov, transient Navier-Stokes).
- **Material models** (`src/materials/`) — device-callable functors (e.g. SVK,
  Mooney-Rivlin).
- **Utilities** (`src/utils/`) — CPU helpers (I/O, mesh processing, quadrature).
- **Examples** (`examples/`) — one driver per physics scenario; each links
  directly against `tlfea_lib`.

---

## 2. Algorithm-Specific Solvers — No Forced Inheritance

Every solver is a **standalone class** tailored to one algorithm.  Do **not**
try to unify solvers through deep inheritance hierarchies:

```
LinearStaticSolver<TData>       – conjugate gradient, linear structural
SyncedNesterovSolver            – Nesterov-accelerated gradient, structural dynamics
SyncedAdamWSolver               – AdamW, structural dynamics
NavierStokesSUPGPSPGSolver      – backward-Euler + BiCGSTAB, incompressible NS
```

Even when two solvers solve "the same physical problem" (e.g. Newton for a
solid vs. Newton for a fluid), they should be **separate classes** with no
forced common base.  Shared implementation details (e.g. CSR pattern building)
may be extracted into free functions or CUDA kernel templates, but inheritance
for algorithmic reuse is explicitly avoided.

Why: virtual dispatch incurs overhead on the GPU; inheritance ties unrelated
algorithms together; composition is easier to test and optimise independently.

---

## 3. One Element Class per Element Type

Each distinct combination of (element shape, polynomial order, quadrature rule)
gets **its own class and its own CUDA kernels**.  Do not parametrise a single
class over order or quadrature:

| Class             | Shape  | Order | Quadrature |
|-------------------|--------|-------|------------|
| `GPU_FEAT4_Data`  | TET4   | 1     | 1-point centroid |
| `GPU_FEAT10_Data` | TET10  | 2     | 5-point Keast |
| `GPU_ANCF3243_Data` | Beam | —     | 7×7×3 Gauss |

For a new element (e.g. HEX8), create `src/elements/FEHex8Data.cuh/.cu` from
scratch.  Copying the structure of an existing element is fine; sharing device
kernels via templates is fine; inheriting from another element's implementation
is not.

### Element class checklist
- `static constexpr int N_NODES_PER_ELEM`
- `static constexpr int N_QP_PER_ELEM`
- `GPU_XxxData* d_data` — device-side mirror of the host struct
- `Initialize()` — allocate all `mophi::DualArray<>` members
- `Setup(...)` — fill arrays from host data and push to device
- `Destroy()` — free all GPU memory
- `CalcDnDuPre()` — precompute reference-configuration geometry
- `SetNodalFixed()` / `SetExternalForce()` — boundary conditions
- `RetrievePositionToCPU()` etc. — read-back accessors

---

## 4. Self-Contained Solver Ownership

A solver **owns its own data**.  When a solver needs mesh/state data it either:
1. Receives a non-owning pointer to an `ElementData` object (structural
   solvers, e.g. `LinearStaticSolver<TData>`), or
2. Allocates and owns all GPU arrays itself (fluid solver
   `NavierStokesSUPGPSPGSolver`).

Fluid solvers in particular should **not** reuse structural element classes
(e.g. `GPU_FEAT4_Data`) — the Navier-Stokes DOF layout, boundary conditions,
and stabilisation terms are too different to share cleanly.

---

## 5. GPU Data Layout and Memory Management

### `mophi::DualArray<T>`
All large arrays are managed via `mophi::DualArray<T>`, which maintains a
pinned host vector and a device pointer.  Pattern:

```cpp
mophi::DualArray<Real> da_myArray_;
Real* d_myArray_ = nullptr;          // raw device pointer

// Allocation
da_myArray_.resize(n);
da_myArray_.BindDevicePointer(&d_myArray_);
da_myArray_.SetVal(Real(0));
da_myArray_.MakeReadyDevice();

// Host → Device
std::copy(src.begin(), src.end(), da_myArray_.host());
da_myArray_.ToDevice();

// Device → Host
da_myArray_.MakeReadyHost();
// then read da_myArray_.host()[i]
```

Never call `cudaMalloc` for arrays that exist for the lifetime of the object
— use `DualArray`.  Use bare `cudaMalloc` / `cudaFree` only for temporary
scratch buffers (e.g. inside a single kernel launch sequence).

### CSR Sparse Matrices
Sparse system matrices use CSR format: `offsets` (n_dof+1), `columns` (nnz),
`values` (nnz).  The sparsity pattern is built **once** using Thrust
sort + unique on packed `(row<<32|col)` keys, then reused every assembly step.

The `values` array is zeroed and re-filled at each assembly.

---

## 6. CUDA Kernel Design Principles

- **One kernel = one well-defined task.**  Do not write monolithic kernels.
- **Parallelism level**: prefer element-level or DOF-level parallelism
  (one thread per element, or one thread per (element, quadrature point), etc.).
- **Atomic operations** are acceptable for CSR assembly (scatter), but avoid
  them for dense inner-loop arithmetic.
- **`#pragma unroll`** for small fixed-size loops (3×3 matrices, etc.).
- **`__forceinline__`** for frequently called device helpers (e.g. 3×3 matrix
  inverse, binary search in CSR column array).
- Use `constexpr` block/thread sizes; avoid dynamic shared memory unless
  demonstrably beneficial.
- Every kernel call must be followed by `MOPHI_GPU_CALL(cudaDeviceSynchronize())`
  during development; production code may pipeline asynchronously.

---

## 7. Error Handling

- CUDA runtime errors: wrap all CUDA calls in `MOPHI_GPU_CALL(...)`.
- cuSPARSE errors: use `CHECK_CUSPARSE(...)`.
- cuBLAS errors: use `CHECK_CUBLAS(...)`.
- Host-side errors: use `MOPHI_ERROR("message")` for non-fatal warnings;
  throw `std::runtime_error` for unrecoverable failures in host code.
- Do not `exit()` or `abort()` in library code; propagate errors to the caller.

---

## 8. Type System

The floating-point type `Real` is defined in `src/types.h` as `double`.
Use `Real` everywhere (not `float` or `double` directly) so that a future
switch to single precision only requires one change.

Eigen matrix aliases:
- `MatrixXR` — dynamic real matrix
- `VectorXR` — dynamic real vector
- `MatrixXi` — dynamic integer matrix
- `VectorXi` — dynamic integer vector

On the device, use raw C arrays and Eigen `Map<>` for fixed-size subsets.

---

## 9. Build System

The project uses CMake 3.18+.  Key conventions:
- Source files discovered via `file(GLOB ... CONFIGURE_DEPENDS)` — adding a
  new `.cu` file to `src/elements/` or `src/solvers/` automatically includes
  it in `tlfea_lib`.
- New **examples** must be explicitly added to `examples/CMakeLists.txt`.
- CUDA architecture list: `60;70;75;80;86` — extend if targeting newer GPUs.
- The MoPhiEssentials submodule lives at `external/MoPhiEssentials/` and must
  be initialised with `git submodule update --init` before building.

---

## 10. Adding a New Solver — Step-by-Step

1. **Header** `src/solvers/MySolver.cuh`:
   - Define a `MyParams` struct with all tuneable parameters.
   - Define `class MySolver` with `Setup`, `Step`/`Solve`, result accessors.
   - Document the mathematical formulation in the file header.
2. **Implementation** `src/solvers/MySolver.cu`:
   - Define all CUDA kernels in an anonymous namespace.
   - Implement host methods that orchestrate kernel calls.
   - Include `explicit template instantiations` if templated.
3. **Demo** `examples/my_demo.cc`:
   - Load a mesh, set BCs, construct the solver, run the loop, write output.
   - Follow the style of `examples/beam_linear_static_t4.cc` or
     `examples/flow_past_cylinder.cc`.
4. **CMakeLists** `examples/CMakeLists.txt`:
   - Add `add_executable`, `target_link_libraries`, `target_include_directories`,
     `set_target_properties` blocks (copy from an existing target).
5. **Build and run**:
   ```
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target my_demo --parallel
   ./build/bin/my_demo
   ```

---

## 11. Adding a New Element — Step-by-Step

1. **Header** `src/elements/FEXxxData.cuh`:
   - Follow the structure of `FEAT4Data.cuh` or `FEAT10Data.cuh`.
   - Add `static constexpr int N_NODES_PER_ELEM` and `N_QP_PER_ELEM`.
   - Add a new enum value to `ElementType` in `ElementBase.h`.
2. **Device functions** `src/elements/FEXxxDataFunc.cuh`:
   - Implement `compute_p`, `compute_hessian_assemble_csr`, and any other
     kernels the element needs.
3. **Implementation** `src/elements/FEXxxData.cu`:
   - Implement `Initialize`, `Setup`, `CalcDnDuPre`, `CalcMassMatrix`,
     `CalcInternalForce`, `Destroy`, and retrieval functions.
4. The file is automatically compiled into `tlfea_lib` via the glob.

---

## 12. Performance Guidelines

- Profile before optimising: use `nsys` (Nsight Systems) or `ncu` (Nsight
  Compute) to find bottlenecks.
- For assembly kernels with atomic contention, consider element colouring or
  hierarchical assembly.
- For repeated solves with the same sparsity pattern, cache the CSR structure
  (the `pattern_built_` flag pattern).
- Avoid unnecessary host–device round-trips; batch transfers.
- Use `cudaMemcpyAsync` and CUDA streams for overlap when profiling shows
  compute–transfer overlap is beneficial.

---

## 13. Coding Style

- Follow the `.clang-format` at the repo root (run `./.format_all` to reformat).
- Every new public API function must have a Doxygen block (`/** ... */`).
- File header blocks mirror the existing `/*======= ... =======*/` style.
- `namespace tlfea` wraps all library code; examples do `using namespace tlfea;`.
- Anonymous namespaces in `.cu` files hide device helpers from the linker.
- Use `MOPHI_INFO(...)` for informational progress messages (not `std::cout`
  inside library code).

---

## 14. Testing

There is no automated test suite yet.  New physics should include at minimum:
- A **smoke test** (does it run without crashing?).
- A **validation comparison** (e.g. compare FEA tip deflection to Euler-Bernoulli,
  or NS drag coefficient to a reference value).

The existing examples serve as integration tests.  Do not remove or weaken
existing examples.
