"""
_bridge.py
----------
The bridge between ``mechanicsdsl-core`` and this package's code generators.

Every real generation path in ``mechanicsdsl-datasets`` starts here: a DSL spec (``.msl``
file, raw DSL string, or core preset name) is compiled by the core
:class:`~mechanics_dsl.PhysicsCompiler`, and the resulting equations of
motion, coordinates, parameters and initial conditions are packaged into a
plain, generator-ready :class:`CompiledSystem`.

On top of compilation the bridge offers a small, dependency-light numeric
toolkit shared by every generator and test:

* :attr:`CompiledSystem.energy` - the energy function ``E = sum(qdot dL/dqdot) - L``
  derived symbolically from the Lagrangian (``None`` when unavailable),
* :func:`numeric_rhs` - a NumPy first-order right-hand side via ``lambdify``,
* :func:`simulate` - a fixed-step RK4 reference integrator,
* :func:`energy_drift` - relative energy drift over a reference run,
* :func:`operation_count` - floating-point operation counts per RHS call.

The core package is imported as ``mechanics_dsl`` (underscore), not
``mechanicsdsl``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    import sympy as sp
except ImportError:  # pragma: no cover - sympy ships with core
    sp = None  # type: ignore


class CoreNotInstalled(RuntimeError):
    """Raised when ``mechanicsdsl-core`` is not importable."""


class DSLCompileError(RuntimeError):
    """Raised when the core compiler rejects a DSL spec."""


@dataclass
class CompiledSystem:
    """A DSL system compiled down to generator-ready primitives.

    Attributes
    ----------
    name:
        System name from ``\\system{...}``.
    coordinates:
        Generalized coordinate names, e.g. ``["theta1", "theta2"]``.
    parameters:
        Flat ``name -> float`` map of physical parameters.
    initial_conditions:
        Initial state, including both ``q`` and ``q_dot`` entries.
    equations:
        Acceleration equations as ``"{coord}_ddot" -> sympy.Expr``.
    units:
        ``param -> unit`` map, preserved for documentation/comments.
    source:
        The originating DSL text, embedded into generated files for
        provenance.
    lagrangian:
        The Lagrangian as a SymPy expression in ``q`` and ``q_dot`` symbols,
        or ``None`` for systems defined another way.
    energy:
        ``sum_i q_dot_i * dL/dq_dot_i - L`` (the Jacobi energy integral), or
        ``None`` when no Lagrangian is available. Conserved exactly for
        conservative, time-independent systems.
    conservative:
        ``False`` when the spec declares non-conservative forces or damping,
        in which case :attr:`energy` is expected to change over time.
    """

    name: str
    coordinates: List[str]
    parameters: Dict[str, float]
    initial_conditions: Dict[str, float]
    equations: Dict[str, "sp.Expr"]
    units: Dict[str, str] = field(default_factory=dict)
    source: str = ""
    lagrangian: Optional["sp.Expr"] = None
    energy: Optional["sp.Expr"] = None
    conservative: bool = True

    @property
    def dof(self) -> int:
        """Number of generalized coordinates."""
        return len(self.coordinates)

    @property
    def state_dim(self) -> int:
        """Phase-space dimension (2 per coordinate)."""
        return 2 * len(self.coordinates)

    @property
    def state_names(self) -> List[str]:
        """State vector layout: ``[q0..qN-1, q0_dot..qN-1_dot]``."""
        return list(self.coordinates) + [f"{c}_dot" for c in self.coordinates]

    def initial_state(self) -> List[float]:
        """Initial state vector in :attr:`state_names` order."""
        return [float(self.initial_conditions.get(s, 0.0)) for s in self.state_names]

    def describe(self) -> str:
        """Human-readable multi-line summary of the compiled system."""
        lines = [
            f"System:       {self.name}",
            f"DOF:          {self.dof} ({', '.join(self.coordinates)})",
            f"Conservative: {'yes' if self.conservative else 'no (forces/damping declared)'}",
            "Parameters:",
        ]
        for p in sorted(self.parameters):
            unit = self.units.get(p, "")
            lines.append(f"  {p:<12} = {self.parameters[p]:g} {unit}".rstrip())
        lines.append("Initial state:")
        for s, v in zip(self.state_names, self.initial_state()):
            lines.append(f"  {s:<12} = {v:g}")
        lines.append("Equations of motion:")
        for c in self.coordinates:
            lines.append(f"  {c}_ddot = {self.equations[f'{c}_ddot']}")
        if self.energy is not None:
            lines.append(f"Energy:\n  E = {self.energy}")
        return "\n".join(lines)


def list_presets() -> List[str]:
    """Return the names of all presets shipped with ``mechanicsdsl-core``."""
    try:
        from mechanics_dsl import list_presets as _lp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CoreNotInstalled(
            "mechanicsdsl-core is required. Install it with: pip install mechanicsdsl-core"
        ) from exc
    return sorted(_lp())


def _read_source(spec: str) -> str:
    """Return DSL text from a path or a raw DSL string."""
    if "\\system" in spec or "\\lagrangian" in spec or "\\hamiltonian" in spec:
        return spec
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as fh:
            return fh.read()
    # Fall back to literal DSL so a helpful compile error surfaces.
    return spec


def _symbols_by_name(exprs) -> Dict[str, "sp.Symbol"]:
    out: Dict[str, "sp.Symbol"] = {}
    for e in exprs:
        if e is None:
            continue
        for s in e.free_symbols:
            out.setdefault(str(s), s)
    return out


def _derive_energy(lagrangian, coordinates) -> Optional["sp.Expr"]:
    if lagrangian is None or sp is None:
        return None
    syms = _symbols_by_name([lagrangian])
    energy = -lagrangian
    for c in coordinates:
        v = syms.get(f"{c}_dot")
        if v is not None:
            energy += v * sp.diff(lagrangian, v)
    return sp.simplify(energy)


_MAX_SYMBOLIC_SOLVE_DOF = 4


def _explicit_equations(equations, coordinates) -> Dict[str, "sp.Expr"]:
    """Normalise core's equations to explicit ``{coord}_ddot`` expressions.

    For large systems core returns ``{"__mass_matrix__": (M, F)}`` meaning
    ``M * qdd = F``. A diagonal ``M`` (e.g. point masses in Cartesian
    coordinates) is inverted exactly; a small dense ``M`` is solved
    symbolically; anything larger is rejected rather than silently slow.
    """
    equations = dict(equations or {})
    if "__mass_matrix__" not in equations:
        return equations
    M, F = equations["__mass_matrix__"]
    n = M.shape[0]
    if M.is_diagonal():
        qdd = [sp.simplify(F[i] / M[i, i]) for i in range(n)]
    elif n <= _MAX_SYMBOLIC_SOLVE_DOF:
        qdd = list(M.LUsolve(F))
    else:
        raise DSLCompileError(
            f"core returned a dense {n}x{n} mass matrix; explicit accelerations "
            f"are only derived for diagonal matrices or <= {_MAX_SYMBOLIC_SOLVE_DOF} DOF"
        )
    return {f"{c}_ddot": qdd[i] for i, c in enumerate(coordinates)}


def compile_spec(spec: str) -> CompiledSystem:
    """Compile a DSL spec into a :class:`CompiledSystem`.

    Parameters
    ----------
    spec:
        A path to a ``.msl`` file, a raw DSL string, or the name of a core
        preset (e.g. ``"pendulum"``, ``"double_pendulum"``).

    Raises
    ------
    CoreNotInstalled
        If ``mechanicsdsl-core`` cannot be imported.
    DSLCompileError
        If the core compiler fails to compile the spec.
    """
    try:
        from mechanics_dsl import PhysicsCompiler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CoreNotInstalled(
            "mechanicsdsl-core is required for code generation.\n"
            "Install it with: pip install mechanicsdsl-core"
        ) from exc

    source = spec
    if "\\" not in spec and not os.path.isfile(spec):
        try:
            from mechanics_dsl import get_preset

            source = get_preset(spec)
        except Exception:
            source = spec
    source = _read_source(source)

    compiler = PhysicsCompiler()
    try:
        result = compiler.compile_dsl(source)
    except Exception as exc:  # core raises on some malformed inputs
        raise DSLCompileError(str(exc)) from exc
    if not result.get("success"):
        raise DSLCompileError(result.get("error", "unknown compilation error"))

    parameters = {
        name: float(info["value"]) for name, info in (compiler.parameters_def or {}).items()
    }
    units = {
        name: str(info.get("unit", "")) for name, info in (compiler.parameters_def or {}).items()
    }
    coordinates = list(compiler.get_coordinates())

    lagrangian = None
    if getattr(compiler, "lagrangian", None) is not None:
        try:
            lagrangian = compiler.symbolic.ast_to_sympy(compiler.lagrangian)
        except Exception:
            lagrangian = None
    conservative = not (
        getattr(compiler, "forces", None)
        or getattr(compiler, "damping_forces", None)
        or getattr(compiler, "rayleigh_dissipation", None) is not None
        or getattr(compiler, "non_holonomic_constraints", None)
    )

    return CompiledSystem(
        name=compiler.system_name,
        coordinates=coordinates,
        parameters=parameters,
        initial_conditions=dict(compiler.initial_conditions or {}),
        equations=_explicit_equations(compiler.equations, coordinates),
        units=units,
        source=source.strip(),
        lagrangian=lagrangian,
        energy=_derive_energy(lagrangian, coordinates),
        conservative=conservative,
    )


# ---------------------------------------------------------------------------
# Numeric toolkit
# ---------------------------------------------------------------------------
def _arg_symbols(system: CompiledSystem, extra=()) -> Tuple[list, list]:
    exprs = list(system.equations.values()) + [e for e in extra if e is not None]
    by_name = _symbols_by_name(exprs)
    state = [by_name.get(n, sp.Symbol(n)) for n in system.state_names]
    params = [by_name.get(p, sp.Symbol(p)) for p in sorted(system.parameters)]
    return state, params


def numeric_rhs(system: CompiledSystem, parameters: Optional[Dict[str, float]] = None
                ) -> Callable:
    """Return ``f(y) -> dy/dt`` (NumPy) for the first-order system.

    ``parameters`` overrides individual values from :attr:`CompiledSystem.parameters`.
    """
    import numpy as np

    state, params = _arg_symbols(system)
    accels = [system.equations[f"{c}_ddot"] for c in system.coordinates]
    fn = sp.lambdify(state + params, accels, modules="numpy")
    pvals = dict(system.parameters)
    pvals.update(parameters or {})
    pv = [float(pvals[p]) for p in sorted(system.parameters)]
    n = system.dof

    def rhs(y):
        y = np.asarray(y, dtype=float)
        qdd = fn(*y, *pv)
        return np.concatenate([y[n:], np.asarray(qdd, dtype=float).reshape(n)])

    return rhs


def numeric_energy(system: CompiledSystem, parameters: Optional[Dict[str, float]] = None
                   ) -> Optional[Callable]:
    """Return ``E(y)`` as a NumPy callable, or ``None`` without a Lagrangian."""
    if system.energy is None:
        return None
    state, params = _arg_symbols(system, extra=[system.energy])
    fn = sp.lambdify(state + params, system.energy, modules="numpy")
    pvals = dict(system.parameters)
    pvals.update(parameters or {})
    pv = [float(pvals[p]) for p in sorted(system.parameters)]
    return lambda y: float(fn(*y, *pv))


def simulate(system: CompiledSystem, t_end: float = 10.0, dt: float = 1e-3,
             y0: Optional[List[float]] = None,
             parameters: Optional[Dict[str, float]] = None):
    """Integrate with fixed-step classic RK4.

    This is the same scheme every generated target uses, so it doubles as the
    numeric reference generated code is checked against.

    Returns ``(t, Y)`` with ``Y.shape == (len(t), state_dim)``.
    """
    import numpy as np

    f = numeric_rhs(system, parameters)
    steps = int(round(t_end / dt))
    y = np.array(y0 if y0 is not None else system.initial_state(), dtype=float)
    out = np.empty((steps + 1, y.size))
    out[0] = y
    for k in range(steps):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = y
    return np.linspace(0.0, steps * dt, steps + 1), out


def energy_drift(system: CompiledSystem, t_end: float = 10.0, dt: float = 1e-3
                 ) -> Optional[float]:
    """Maximum relative energy change ``max|E(t)-E0| / max(|E0|, 1e-12)``.

    Returns ``None`` when the system has no Lagrangian-derived energy.
    """
    E = numeric_energy(system)
    if E is None:
        return None
    _, Y = simulate(system, t_end=t_end, dt=dt)
    e = [E(row) for row in Y]
    e0 = e[0]
    return max(abs(v - e0) for v in e) / max(abs(e0), 1e-12)


def operation_count(system: CompiledSystem, cse: bool = True) -> Dict[str, int]:
    """Count floating-point operations in one acceleration evaluation.

    With ``cse=True`` common subexpressions are shared, which is what an
    optimising compiler (or the generated code) actually pays for.
    Returns ``{"ops": int, "transcendental": int, "temporaries": int}``.
    """
    exprs = [system.equations[f"{c}_ddot"] for c in system.coordinates]
    temps: list = []
    if cse:
        temps, exprs = sp.cse(exprs)
    all_exprs = [e for _, e in temps] + list(exprs)
    ops = sum(int(sp.count_ops(e)) for e in all_exprs)
    trans = 0
    for e in all_exprs:
        trans += sum(1 for a in sp.preorder_traversal(e)
                     if isinstance(a, sp.Function) and a.func.__name__ in
                     {"sin", "cos", "tan", "exp", "log", "sqrt", "atan2", "asin",
                      "acos", "atan", "sinh", "cosh", "tanh"})
    return {"ops": ops, "transcendental": trans, "temporaries": len(temps)}
