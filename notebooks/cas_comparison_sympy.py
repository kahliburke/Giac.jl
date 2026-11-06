"""SymPy's side of cas_comparison.jl, run natively in Python.

Each task is the notebook's, under the same label. A task runs once, then SymPy's cache is cleared
and it is timed again, as the best of three runs (or one run when it takes more than half a
second). A task that runs past its time limit is interrupted by an alarm, which a pure-Python
computation can receive. Each answer is kept with its `srepr`, so the notebook can rebuild it and
compare it with the other systems' through `describe`. The results are written as TOML, which
Julia reads without a package:

    python cas_comparison_sympy.py ~/.cache/giac/benchmarks/sympy_native.toml
"""

import json
import math
import platform
import re as _regex          # `from sympy import *` brings its own `re`, the real part
import signal
import sys
import time
import timeit

import sympy
from sympy import *
from sympy.core.cache import clear_cache

x, y, z, t, a = symbols("x y z t a")
f = Function("f")(t)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def _status(answer):
    return "unevaluated" if _regex.search(r"\b(Integral|Limit)\(", answer) else "ok"


def measure(task, limit=30):
    """The answer of `task` and the best time of its runs after a first one, cache cleared."""
    signal.alarm(limit)
    try:
        r = task()
        times = []
        for _ in range(3):
            clear_cache()
            t0 = time.perf_counter()
            task()
            times.append(time.perf_counter() - t0)
            if times[-1] > 0.5:
                break
        signal.alarm(0)
    except Timeout:
        return {"answer": f"no answer within {limit} s", "seconds": None, "status": "error"}
    except Exception as e:
        signal.alarm(0)
        return {"answer": f"error: {type(e).__name__}: {e}", "seconds": None, "status": "error"}
    answer = str(r)
    return {"answer": answer, "seconds": min(times), "status": _status(answer), "srepr": srepr(r)}


class _Names(dict):
    """SymPy's names, including those its top level doesn't export, such as `ExprCondPair`."""

    def __missing__(self, name):
        if hasattr(sympy, name):
            return getattr(sympy, name)
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("sympy") and hasattr(module, name):
                return getattr(module, name)
        raise KeyError(name)


def revive(s):
    """An answer from its `srepr`."""
    return eval(s, vars(sympy), _Names())


def point(values):
    """The comparison point, from names and values written as text."""
    return {name: Rational(v) for name, v in values.items()}


def _at(e, p, **over):
    p = {**p, **over}
    e = sympify(e)
    v = complex(N(e.subs({s: p[s.name] for s in e.free_symbols}), 20))
    return [v.real, v.imag]


def values(r, kind, p):
    """The answer's values for a comparison of `kind`: lists of solutions, each a list of [re, im]."""
    if kind == "at":
        return [[_at(r, p)]]
    if kind == "antiderivative":
        hi, lo = _at(r, p), _at(r, p, x=p["x0"])
        return [[[hi[0] - lo[0], hi[1] - lo[1]]]]
    if kind == "series":
        return [[_at(r.removeO(), p)]]
    if kind == "solutions":
        out = []
        for s in (r if isinstance(r, (list, tuple)) else [r]):
            parts = [s[k] for k in sorted(s, key=str)] if isinstance(s, dict) else \
                    list(s) if isinstance(s, (list, tuple)) else [s]
            out.append([_at(c, p) for c in parts])
        return out
    return None


def describe(r, kind, p):
    """The answer typeset and its values at the point `p`, as TOML."""
    try:
        v = values(r, kind, p)
    except Exception:
        v = None
    return f"latex = {_toml(latex(r))}\n" + ("" if v is None else f"values = {v!r}\n")


def cyclic(v):
    n = len(v)
    polys = [sum(prod(v[(i + j) % n] for j in range(k)) for i in range(n)) for k in range(1, n)]
    return polys + [prod(v) - 1]


def dsolve_eq(lhs, rhs):
    return lambda: dsolve(Eq(lhs, rhs), f)


xpos, xreal = Symbol("x", positive=True), Symbol("x", real=True)
n_int = Symbol("n", integer=True)
c = symbols("c1:6")

TASKS = {
    "algebra": {
        "expand (1 + x + y + z)²⁰: the number of terms, 1771": lambda: len(expand((1 + x + y + z) ** 20).args),
        "factor x¹²⁰ − 1": lambda: factor(x**120 - 1),
        "factor (x³y + 2xz − y² + 1)³ (x − y⁴ + 3z)², expanded":
            lambda: factor(expand((x**3 * y + 2 * x * z - y**2 + 1) ** 3 * (x - y**4 + 3 * z) ** 2)),
        "gcd of (x + y + 1)¹⁰(x − y)⁵ and (x + y + 1)⁷(x + 2y)⁶, expanded":
            lambda: factor(gcd(expand((x + y + 1) ** 10 * (x - y) ** 5), expand((x + y + 1) ** 7 * (x + 2 * y) ** 6))),
    },
    "calculus": {
        "∫ x³ eˣ sin x dx": lambda: integrate(x**3 * exp(x) * sin(x), x),
        "∫ 1 / (x⁴ + 1) dx": lambda: integrate(1 / (x**4 + 1), x),
        "∫ sec³ x dx": lambda: integrate(1 / cos(x) ** 3, x),
        "∫₀^∞ sin x / x dx = π/2": lambda: integrate(sin(x) / x, (x, 0, oo)),
        "∫₀^∞ e^(−x²) dx = √π / 2": lambda: integrate(exp(-x**2), (x, 0, oo)),
        "∫₀¹ log x log(1 − x) dx = 2 − π²/6": lambda: integrate(log(x) * log(1 - x), (x, 0, 1)),
        "lim x→0 (sin x − x) / x³ = −1/6": lambda: limit((sin(x) - x) / x**3, x, 0),
        "lim x→∞ (1 + a/x)ˣ = eᵃ": lambda: limit((1 + a / x) ** x, x, oo),
        "tan x to order x⁹": lambda: series(tan(x), x, 0, 10),
    },
    "equations": {
        "Gröbner basis of cyclic-5, graded reverse lexicographic: the number of elements":
            lambda: len(groebner(cyclic(c), *c, order="grevlex").exprs),
        "x² + y² = 1, x³ = y: the solutions": lambda: solve([x**2 + y**2 - 1, x**3 - y], [x, y]),
        "x eˣ = 2": lambda: solve(x * exp(x) - 2, x),
        "a x + y = 1, x − a y = 2": lambda: solve([a * x + y - 1, x - a * y - 2], [x, y]),
    },
    "odes": {
        "y″ + y = sin t": dsolve_eq(f.diff(t, 2) + f, sin(t)),
        "y″ − 2y′ + y = eᵗ / t": dsolve_eq(f.diff(t, 2) - 2 * f.diff(t) + f, exp(t) / t),
        "y′ = y (1 − y)": dsolve_eq(f.diff(t), f * (1 - f)),
        "y′ = t² + y²": dsolve_eq(f.diff(t), t**2 + f**2),
        "y″ + t y = 0 (Airy)": dsolve_eq(f.diff(t, 2) + t * f, 0),
    },
    "simplification": {
        "sin⁴x − cos⁴x + cos 2x = 0": lambda: simplify(sin(x) ** 4 - cos(x) ** 4 + cos(2 * x)),
        "(x⁶ − 1) / (x² − 1) = x⁴ + x² + 1": lambda: simplify((x**6 - 1) / (x**2 - 1)),
        "√(5 + 2√6) = √2 + √3": lambda: sqrtdenest(sqrt(5 + 2 * sqrt(6))),
        "√(x²) = x for x > 0": lambda: simplify(sqrt(xpos**2)),
        "log(eˣ) = x for real x": lambda: simplify(log(exp(xreal))),
        "sin(π n) = 0 for integer n": lambda: simplify(sin(n_int * pi)),
    },
}


def brusselator(N):
    u, v = symbols(f"u1:{N + 1}"), symbols(f"v1:{N + 1}")
    lap = lambda q, i: q[(i - 1) % N] - 2 * q[i] + q[(i + 1) % N]
    F = [1 + u[i] ** 2 * v[i] - 4 * u[i] + lap(u, i) / 50 for i in range(N)] + \
        [3 * u[i] - u[i] ** 2 * v[i] + lap(v, i) / 50 for i in range(N)]
    x0 = [1 + math.sin(2 * math.pi * (i + 1) / N) for i in range(N)] + [3.0] * N
    return list(u) + list(v), F, x0


def jacobian_trial(N, limit=300):
    """The symbolic Jacobian and `lambdify`, its first call and its best evaluation, after a small model."""
    w, F, x0 = brusselator(3)
    lambdify([w], Matrix(F).jacobian(w), "numpy")(x0)
    w, F, x0 = brusselator(N)
    clear_cache()
    signal.alarm(limit)
    try:
        t0 = time.perf_counter()
        jac = lambdify([w], Matrix(F).jacobian(w), "numpy")
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        jac(x0)
        first = time.perf_counter() - t0
        evaluate = min(timeit.repeat(lambda: jac(x0), number=1, repeat=20))
        signal.alarm(0)
    except Timeout:
        return {"answer": f"no answer within {limit} s", "seconds": None, "status": "error"}
    return {"build": build, "first_call": first, "evaluate": evaluate, "seconds": build, "status": "ok"}


def _toml(value):
    # a JSON string is a TOML basic string: the same quotes and escapes
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)


def write_toml(path, results):
    with open(path, "w") as io:
        for key in ("sympy", "python", "date"):
            io.write(f"{key} = {_toml(results[key])}\n")
        for section, entries in results["sections"].items():
            for label, entry in entries.items():
                io.write(f"\n[sections.{section}.{_toml(label)}]\n")
                for key, value in entry.items():
                    if value is not None:
                        io.write(f"{key} = {_toml(value)}\n")


def main(path):
    signal.signal(signal.SIGALRM, _alarm)
    results = {"sympy": sympy.__version__, "python": platform.python_version(),
               "date": time.strftime("%Y-%m-%d %H:%M"), "sections": {}}
    for section, tasks in TASKS.items():
        results["sections"][section] = {label: measure(task) for label, task in tasks.items()}
    results["sections"]["scale"] = {f"Brusselator, N = {N} ({2 * N} states)": jacobian_trial(N) for N in (50, 200)}
    write_toml(path, results)


if __name__ == "__main__":
    main(sys.argv[1])
