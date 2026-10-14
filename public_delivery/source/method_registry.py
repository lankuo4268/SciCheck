"""Method registry: trusted whitelist of candidate actions for the task mechanism.

Actions come ONLY from this registry. The controller never invents an action.
A method card declares the solver it maps to and its structure/budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MethodCard:
    card_id: str
    structure: str          # e.g. "polynomial", "piecewise_constant"
    budget: int             # max solver launches this card allows
    solver: str             # solver identifier to use
    notes: str = ""


# ---- trusted whitelist ----
# Only these card ids and only their declared solvers are allowed.
_WHITELIST: List[MethodCard] = [
    MethodCard(
        card_id="quad_plain",
        structure="polynomial|piecewise_constant",
        budget=1,
        solver="scipy.integrate.quad",
        notes="plain quad over the whole interval; may miss narrow sub-interval contributions",
    ),
    MethodCard(
        card_id="quad_breakpoints",
        structure="piecewise_constant",
        budget=1,
        solver="scipy.integrate.quad",
        notes="quad with explicit interior breakpoints; robust for piecewise-constant",
    ),
    MethodCard(
        card_id="quad_tightened",
        structure="polynomial",
        budget=1,
        solver="scipy.integrate.quad",
        notes="quad with tightened epsabs/epsrel; used as a repair candidate",
    ),
    MethodCard(
        card_id="root_newton",
        structure="root",
        budget=1,
        solver="newton",
        notes="COARSE Newton diagnostic (xtol=1e-3, maxiter=24): DEVELOPMENT/DIAGNOSTIC "
              "route, NOT a library-default pass; keep the residual as a diagnostic and "
              "judge by the root-position forward error",
    ),
    MethodCard(
        card_id="root_brent",
        structure="root",
        budget=1,
        solver="scipy.optimize.brentq",
        notes="strict Brent (xtol=1e-13, rtol=1e-14, maxiter=300); trustworthy root route",
    ),
    MethodCard(
        card_id="root_direct",
        structure="root",
        budget=1,
        solver="structural",
        notes="public DIRECT route: the unique real root of scale*(x-shift)^power is x=shift "
              "derived from the PUBLIC structure only; no private reference interface is called",
    ),
    MethodCard(
        card_id="ls_np32",
        structure="linear_system",
        budget=1,
        solver="numpy.linalg.solve.float32",
        notes="NumPy float32 candidate for a rational square-matrix system (development/diagnostic: "
              "24-bit mantissa, coarse); verdict by per-component forward error",
    ),
    MethodCard(
        card_id="ls_np64",
        structure="linear_system",
        budget=1,
        solver="numpy.linalg.solve.float64",
        notes="NumPy float64 candidate for a rational square-matrix system (repair route: "
              "53-bit mantissa); verdict by per-component forward error",
    ),
    MethodCard(
        card_id="ls_decimal_high",
        structure="linear_system",
        budget=1,
        solver="decimal-elimination",
        notes="public HIGH-PRECISION Decimal route: same math input + original tolerance, a "
              "different (higher-precision) output format and cost; does NOT read the private "
              "Fraction reference. Recorded as its own task/route, does not change the original "
              "task's terminal state",
    ),
    MethodCard(
        card_id="ode_euler10",
        structure="ode",
        budget=1,
        solver="euler10",
        notes="10-step forward-Euler candidate for the scalar linear IVP y'=a*y+b; "
              "coarse development/diagnostic route (h=(t1-t0)/10); NOT a library-default "
              "pass. Verdict by forward error of y(t1) against the original tolerance",
    ),
    MethodCard(
        card_id="ode_dop853",
        structure="ode",
        budget=1,
        solver="scipy.integrate.solve_ivp.dop853",
        notes="SciPy DOP853 explicit Runge-Kutta 8th-order candidate (confirmed against the local scipy source); "
              "rtol=1e-11, atol=1e-13. Repair route: trustworthy candidate for the "
              "scalar linear IVP. Verdict by forward error of y(t1) against the original "
              "tolerance",
    ),
    MethodCard(
        card_id="ode_closed_naive",
        structure="ode",
        budget=1,
        solver="closed_form_naive",
        notes="UNSTABLE closed-form diagnostic route for the scalar linear IVP y'=a*y+b: "
              "casts public rational inputs to binary64 FIRST, then performs the naive float "
              "endpoint subtraction dt=float(t1)-float(t0); deliberately preserves the "
              "cancellation/endpoint-subtraction problem. NOT a pass route. Verdict by forward "
              "error of y(t1) against the original tolerance",
    ),
    MethodCard(
        card_id="ode_closed_stable",
        structure="ode",
        budget=1,
        solver="closed_form_stable",
        notes="STRONG PUBLIC closed-form route for the scalar linear IVP y'=a*y+b: forms the "
              "exact Fraction dt=t1-t0, z=a*dt, bdt=b*dt FIRST, then casts to binary64; "
              "candidate = fsum([y0*exp(z), bdt*expm1(z)/z]) with expm1(z)/z=1 when z=0. "
              "No reference read inside the candidate; verdict by forward error of y(t1) "
              "against the original tolerance",
    ),
]


class MethodLookupError(Exception):
    pass


class TaskRegistry:
    """Registry for method cards. Management interface only; never returns
    reference answers to business tools."""

    def __init__(self, cards: Optional[List[MethodCard]] = None):
        self._cards: Dict[str, MethodCard] = {}
        for c in (cards if cards is not None else _WHITELIST):
            self._cards[c.card_id] = c

    def method_lookup(self, task_id: str, card_id: str) -> MethodCard:
        """Return the trusted card for card_id. Reject unknown card or extra fields.

        task_id is accepted but not used to change the whitelist; the whitelist is
        global and frozen. Unknown card_id raises MethodLookupError.
        """
        card = self._cards.get(card_id)
        if card is None:
            raise MethodLookupError(f"unknown card_id {card_id!r}; not in trusted whitelist")
        return card

    def allowed_actions(self, task_id: str, card_id: str) -> List[str]:
        """The only actions permitted for a card: its solver id. This is the
        whitelist the controller enforces on execute_task(action)."""
        card = self.method_lookup(task_id, card_id)
        return [card.solver]

    def all_cards(self) -> Dict[str, MethodCard]:
        return dict(self._cards)
