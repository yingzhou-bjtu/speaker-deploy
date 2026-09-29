/-
Copyright (c) 2026 The Tau Ceti contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: The Tau Ceti contributors
-/
module

public import Mathlib.Data.List.Defs
public import Mathlib.Tactic.Linarith
public import Mathlib.Topology.MetricSpace.Basic

/-!
# A small formal core for risk-aware streaming reclustering

This file formalizes the control layer of the proposed streaming speaker
embedding method. It intentionally does not claim that nearest-representative
assignment is equivalent to HDBSCAN. That equivalence is a data/model
assumption and must be checked separately.

The formal results cover three reusable facts: a positive nearest/second
nearest margin gives a unique fast-path choice, bounded enqueue preserves the
queue bound, and exact fast-path/refresh-path contracts imply prefix
consistency with a reference clusterer.
-/

public section

namespace TauCeti

namespace Streaming

/-- The two decisions made by the risk gate. -/
inductive Decision where
  | fast
  | refresh
deriving DecidableEq

/-- Distances to the closest and second-closest cached representatives. -/
structure GateEvidence where
  nearest : Nat
  second : Nat

/-- A conservative gate: a fast update is allowed only with a strict margin. -/
def fastAllowed (e : GateEvidence) : Prop := e.nearest < e.second

theorem fastAllowed_unique (e : GateEvidence) (h : fastAllowed e) :
    e.nearest ≠ e.second := by
  exact Nat.ne_of_lt h

/- The geometric part of the argument is independent of HDBSCAN. It says
   that bounded sample/representative error plus center separation yields a
   lower bound on the nearest-vs-second-nearest margin. -/
theorem separated_margin {α : Type*} [PseudoMetricSpace α]
    (x c₁ c₂ r₁ r₂ : α) (η ρ Δ : ℝ)
    (hx : dist x c₁ ≤ η) (hr₁ : dist c₁ r₁ ≤ ρ)
    (hr₂ : dist c₂ r₂ ≤ ρ) (hcenters : Δ ≤ dist c₁ c₂)
    (hsep : 2 * (η + ρ) ≤ Δ) :
    dist x r₁ ≤ η + ρ ∧
      Δ - (η + ρ) ≤ dist x r₂ ∧
        Δ - 2 * (η + ρ) ≤ dist x r₂ - dist x r₁ ∧
          0 ≤ dist x r₂ - dist x r₁ := by
  have h₁ : dist x r₁ ≤ dist x c₁ + dist c₁ r₁ := dist_triangle _ _ _
  have h₂ : dist c₁ c₂ ≤ dist c₁ x + dist x r₂ + dist r₂ c₂ := by
    calc
      dist c₁ c₂ ≤ dist c₁ r₂ + dist r₂ c₂ := dist_triangle _ _ _
      _ ≤ (dist c₁ x + dist x r₂) + dist r₂ c₂ := by
        have htri : dist c₁ r₂ ≤ dist c₁ x + dist x r₂ := dist_triangle _ _ _
        have hnonneg : 0 ≤ dist r₂ c₂ := dist_nonneg
        linarith
      _ = dist c₁ x + dist x r₂ + dist r₂ c₂ := by ring
  have hx' : dist c₁ x ≤ η := by simpa [dist_comm] using hx
  have hr₂' : dist r₂ c₂ ≤ ρ := by simpa [dist_comm] using hr₂
  have hupper : dist x r₁ ≤ η + ρ := by linarith
  have hlower : Δ - (η + ρ) ≤ dist x r₂ := by linarith
  exact ⟨hupper, hlower, by linarith, by linarith [hsep]⟩

/-- Enqueue one maintenance job, saturating at the configured capacity. -/
def enqueue (capacity pending : Nat) : Nat := min capacity (pending + 1)

theorem enqueue_le_capacity (capacity pending : Nat) :
    enqueue capacity pending ≤ capacity := by
  exact Nat.min_le_left _ _

theorem enqueue_preserves_bound (capacity pending : Nat) (_h : pending ≤ capacity) :
    enqueue capacity pending ≤ capacity := by
  exact enqueue_le_capacity capacity pending

/-- A one-step abstract state. `label` is the emitted tentative cluster label. -/
structure State (Label : Type*) where
  label : Label
  pending : Nat

variable {Label : Type*} (reference : List Label → Label)

/- The semantic contract needed for a correctness theorem. The fast branch is
   not proved from geometry here: it is supplied by a separation/calibration
   argument for a concrete embedding model. -/
def StepExact (fast refresh : List Label → Label) : Prop :=
  (∀ history, fast history = reference history) ∧
    (∀ history, refresh history = reference history)

/-- If both branches satisfy the reference contract, every processed prefix is
    reference-consistent. -/
theorem step_exact (fast refresh : List Label → Label)
    (h : StepExact reference fast refresh) (history : List Label)
    (choose : Decision) :
    (match choose with
      | Decision.fast => fast history
      | Decision.refresh => refresh history) = reference history := by
  cases choose with
  | fast => exact h.1 history
  | refresh => exact h.2 history

/-- The emitted label for one gate decision. -/
def emit (fast refresh : List Label → Label) :
    Decision → List Label → Label
  | Decision.fast, history => fast history
  | Decision.refresh, history => refresh history

/-- The labels emitted by a sequence of decisions. The history is extended
   with each emitted label, matching the streaming prefix semantics. -/
def trace (fast refresh : List Label → Label) :
    List Decision → List Label → List Label
  | [], _ => []
  | decision :: decisions, history =>
      emit fast refresh decision history ::
        trace fast refresh decisions (history ++ [emit fast refresh decision history])

/-- The corresponding trace produced by the reference clusterer. -/
def referenceTrace :
    (List Label → Label) → List Decision → List Label → List Label
  | _, [], _ => []
  | ref, _ :: decisions, history =>
      ref history ::
        referenceTrace ref decisions (history ++ [ref history])

/-- The full streaming trace is reference-consistent, not only each isolated
   branch. This is the formal prefix version of `step_exact`. -/
theorem trace_exact (fast refresh : List Label → Label)
    (h : StepExact reference fast refresh) (decisions : List Decision)
    (history : List Label) :
    trace fast refresh decisions history =
      referenceTrace reference decisions history := by
  induction decisions generalizing history with
  | nil => rfl
  | cons decision decisions ih =>
      cases decision with
      | fast =>
          simp only [trace, referenceTrace, emit]
          rw [h.1 history]
          rw [ih (history := history ++ [reference history])]
      | refresh =>
          simp only [trace, referenceTrace, emit]
          rw [h.2 history]
          rw [ih (history := history ++ [reference history])]

/- The following theorem formalizes the local optimization step. A certified
   state may use the cheap FAST branch; an uncertified state must use REFRESH.
   Under this policy contract, the maximal gate is pointwise cost-optimal when
   FAST is strictly cheaper. -/
def policyCost (fastCost refreshCost : Nat) : Decision → Nat
  | Decision.fast => fastCost
  | Decision.refresh => refreshCost

def maximalPolicy (certified : Prop) [Decidable certified] : Decision :=
  if certified then Decision.fast else Decision.refresh

def Admissible (certified : Prop) (choose : Decision) : Prop :=
  choose = Decision.fast → certified

theorem maximalPolicy_cost_le (certified : Prop) [Decidable certified]
    (fastCost refreshCost : Nat) (hcost : fastCost < refreshCost)
    (choose : Decision) (hadmissible : Admissible certified choose) :
    policyCost fastCost refreshCost (maximalPolicy certified) ≤
      policyCost fastCost refreshCost choose := by
  by_cases hcertified : certified
  · unfold maximalPolicy
    rw [ite_eq_left hcertified]
    simp only [policyCost]
    cases choose with
    | fast => exact Nat.le_refl _
    | refresh => exact Nat.le_of_lt hcost
  · have hrefresh : choose = Decision.refresh := by
      cases choose with
      | fast => exact False.elim (hcertified (hadmissible rfl))
      | refresh => rfl
    unfold maximalPolicy
    rw [ite_eq_right hcertified]
    simp only [policyCost, hrefresh]
    exact Nat.le_refl _

/-- One abstract queue update with truncated subtraction. -/
def queueStep (capacity pending submitted completed : Nat) : Nat :=
  min capacity ((pending + submitted) - completed)

/-- The queue update never exceeds its configured capacity. -/
theorem queueStep_le_capacity (capacity pending submitted completed : Nat) :
    queueStep capacity pending submitted completed ≤ capacity := by
  exact Nat.min_le_left _ _

/-- The queue update is always nonnegative because it is represented by `Nat`. -/
theorem queueStep_nonnegative (capacity pending submitted completed : Nat) :
    0 ≤ queueStep capacity pending submitted completed := by
  exact Nat.zero_le _

end Streaming

end TauCeti
