/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    FEASolver.h
 * Brief:   Declares the FEASolver class, the top-level manager that serves as
 *          the one-stop interface for all element and solver objects in a
 *          TLFEA simulation.  Users instantiate one FEASolver, register their
 *          element and solver objects under string names, and retrieve them by
 *          name when they need to invoke type-specific methods, e.g.
 *
 *            feasolver.GetSolver("solver1")->Solve();
 *            feasolver.GetElement("body1")->CalcInternalForce();
 *==============================================================
 *==============================================================*/

#pragma once

#include <map>
#include <stdexcept>
#include <string>

#include "elements/ElementBase.h"
#include "solvers/SolverBase.h"

namespace tlfea {

/**
 * @brief Top-level manager class for a TLFEA simulation.
 *
 * FEASolver owns no heap memory of its own; it stores raw (non-owning)
 * pointers to ElementBase and SolverBase objects whose lifetimes are
 * managed by the caller.  This matches the existing usage pattern where
 * element and solver objects are stack- or heap-allocated by the user.
 */
class FEASolver {
  public:
    FEASolver()  = default;
    ~FEASolver() = default;

    // Non-copyable, non-movable (the maps hold raw pointers, so copying
    // would create dangling aliased ownership).
    FEASolver(const FEASolver&)            = delete;
    FEASolver& operator=(const FEASolver&) = delete;
    FEASolver(FEASolver&&)                 = delete;
    FEASolver& operator=(FEASolver&&)      = delete;

    // -----------------------------------------------------------------------
    // Element management
    // -----------------------------------------------------------------------

    /**
     * @brief Register an element object under a unique name.
     * @param name   Unique identifier for the element.
     * @param element Non-owning pointer to the element; must remain valid for
     *                the lifetime of this FEASolver.
     * @throws std::invalid_argument if @p name is already registered or if
     *         @p element is nullptr.
     */
    void AddElement(const std::string& name, ElementBase* element) {
        if (!element)
            throw std::invalid_argument("FEASolver::AddElement: element pointer is null");
        auto [it, inserted] = elements_.emplace(name, element);
        if (!inserted)
            throw std::invalid_argument("FEASolver::AddElement: element '" + name + "' already registered");
    }

    /**
     * @brief Retrieve a previously registered element by name.
     * @param name Unique identifier supplied to AddElement.
     * @return Non-owning pointer to the element.
     * @throws std::out_of_range if @p name is not found.
     */
    ElementBase* GetElement(const std::string& name) {
        auto it = elements_.find(name);
        if (it == elements_.end())
            throw std::out_of_range("FEASolver::GetElement: element '" + name + "' not found");
        return it->second;
    }

    /** @brief Const overload of GetElement. */
    const ElementBase* GetElement(const std::string& name) const {
        auto it = elements_.find(name);
        if (it == elements_.end())
            throw std::out_of_range("FEASolver::GetElement: element '" + name + "' not found");
        return it->second;
    }

    /** @brief Returns the number of registered elements. */
    int GetNumElements() const { return static_cast<int>(elements_.size()); }

    // -----------------------------------------------------------------------
    // Solver management
    // -----------------------------------------------------------------------

    /**
     * @brief Register a solver object under a unique name.
     * @param name   Unique identifier for the solver.
     * @param solver Non-owning pointer to the solver; must remain valid for
     *               the lifetime of this FEASolver.
     * @throws std::invalid_argument if @p name is already registered or if
     *         @p solver is nullptr.
     */
    void AddSolver(const std::string& name, SolverBase* solver) {
        if (!solver)
            throw std::invalid_argument("FEASolver::AddSolver: solver pointer is null");
        auto [it, inserted] = solvers_.emplace(name, solver);
        if (!inserted)
            throw std::invalid_argument("FEASolver::AddSolver: solver '" + name + "' already registered");
    }

    /**
     * @brief Retrieve a previously registered solver by name.
     * @param name Unique identifier supplied to AddSolver.
     * @return Non-owning pointer to the solver.
     * @throws std::out_of_range if @p name is not found.
     */
    SolverBase* GetSolver(const std::string& name) {
        auto it = solvers_.find(name);
        if (it == solvers_.end())
            throw std::out_of_range("FEASolver::GetSolver: solver '" + name + "' not found");
        return it->second;
    }

    /** @brief Const overload of GetSolver. */
    const SolverBase* GetSolver(const std::string& name) const {
        auto it = solvers_.find(name);
        if (it == solvers_.end())
            throw std::out_of_range("FEASolver::GetSolver: solver '" + name + "' not found");
        return it->second;
    }

    /** @brief Returns the number of registered solvers. */
    int GetNumSolvers() const { return static_cast<int>(solvers_.size()); }

  private:
    std::map<std::string, ElementBase*> elements_;
    std::map<std::string, SolverBase*>  solvers_;
};

}  // namespace tlfea
