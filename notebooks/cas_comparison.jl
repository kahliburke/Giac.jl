try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Giac.jl among computer algebra systems

How the Giac.jl of this branch compares with Symbolics.jl and ModelingToolkit, SymPy, and Mathematica. The matrix first rates each capability from the systems' documented features. The sections after it run the same tasks in all four systems, side by side, with the answers and the time each took, so the ratings rest on evidence where a test exists.

Each Julia system is timed after a first run that compiles, as the best of three runs. Mathematica runs through `wolframscript`, one kernel per section, timed inside the kernel with `AbsoluteTiming` after a first run and with its caches cleared.

SymPy has two columns. **SymPy, from Julia** calls it through SymPyPythonCall, on a thread of its own, after a first run and with its cache cleared; a task that may not end runs instead in a separate Python process with a time limit, since a call into Python can't be interrupted. **SymPy, in Python** is the same task run natively by `cas_comparison_sympy.py`, beside this notebook: a first run, then the best of three with the cache cleared, each task under an alarm. Its results are kept in `~/.cache/giac/benchmarks/` and read here; the script runs again only when it changes.
"""

#%% code id=python nocache
using DedicatedThreads: DedicatedPool, spawn
# Python belongs to the OS thread that starts it, and a cell may run on any thread. So SymPy gets a
# thread of its own: SymPyPythonCall is loaded there, which starts Python on it, and every SymPy
# call is sent there with `py`. The module is bound from that load rather than by an `import`,
# which the notebook would run early, on another thread.
const PYTHON = DedicatedPool(1; name = :python)
py(f) = fetch(spawn(f, PYTHON))
const SP = py(() -> Base.require(Base.PkgId(Base.UUID("bc8888f7-b21e-4b7c-a06a-5d9c9496438c"), "SymPyPythonCall")))
py(() -> Threads.threadid()), py(() -> string(SP.sympy.factor(SP.sympy.sympify("x^12 - 1"))))

#%% code id=setup
using Giac, GiacSlate, LinearAlgebra, BenchmarkTools
import Symbolics as Sym, Groebner
const C = Giac.Commands
# `wolframscript` with `args`, its output as text: one kernel at a time, and a run that crashes is
# tried again, up to three times
WOLFRAM = ReentrantLock()
function wolfram(args...)
    lock(WOLFRAM) do
        for attempt in 1:3
            try
                return strip(read(`wolframscript $args`, String))
            catch e
                attempt == 3 && rethrow()
            end
        end
    end
end
# Wolfram Language code in a fresh Mathematica kernel, its output as text
wl(code) = wolfram("-code", code)
(; giac = pkgdir(Giac), symbolics = pkgversion(Sym), sympy = py(() -> string(SP.sympy.__version__)), mathematica = wl("\$VersionNumber"))

#%% code id=sympy_native
import TOML
# SymPy's tasks run natively in Python by the script beside this notebook. Its results are kept,
# and the script runs again only when it is newer than them.
native_script = joinpath(pkgdir(Giac), "notebooks", "cas_comparison_sympy.py")
native_file = joinpath(homedir(), ".cache", "giac", "benchmarks", "sympy_native.toml")
if !isfile(native_file) || mtime(native_file) < mtime(native_script)
    mkpath(dirname(native_file))
    run(`$(SP.PythonCall.C.CTX.exe_path) $native_script $native_file`)
end
native = TOML.parsefile(native_file)
(sympy = native["sympy"], python = native["python"], run = native["date"])

#%% code id=matrix collapsed
"The capability matrix: headings, and rows of a capability's rating in each system."
struct CapabilityMatrix
    systems::Vector{String}
    rows::Vector{Any}            # a heading `String`, or `capability => [rating, …]`
end

SYSTEMS = ["Giac.jl", "Symbolics / MTK", "SymPy", "Mathematica"]
capabilities = CapabilityMatrix(SYSTEMS, Any[
    "Core algebra",
    "Expand / factor polynomials" => ["● factors with square roots by default (`withsqrt`)", "○ no public `factor`; `gcd` takes numbers only", "●", "● fastest"],
    "Simplification" => ["● six of six tests; rule-based `simplest` search †", "○ two of six tests", "● six of six tests", "● `Simplify` five of six; FullSimplify"],
    "Assumptions" => ["● three of three tests; declared on variables, functions, matrices †", "○ none", "● three of three tests", "● three of three tests"],
    "Pattern matching / rewrite rules" => ["● †", "●", "◐", "●"],
    "Calculus",
    "Differentiation" => ["●", "●", "●", "●"],
    "Indefinite integration" => ["● three of three tests, fastest by 10–100×", "○ no `integrate`", "● three of three tests", "● three of three tests"],
    "Definite integrals, special functions" => ["◐ two of three tests", "○", "◐ two of three; a garbled third", "● three of three tests"],
    "Limits, series" => ["●", "◐ `limit` fails on `sin`; `taylor` works", "●", "●"],
    "Closed-form ODE solutions" => ["◐ three of five tests, each under 1 ms", "○ one of five; one wrong answer", "● four of five tests", "● five of five tests"],
    "Equations and algebra",
    "Polynomial systems (Gröbner bases)" => ["● fastest on cyclic-5; msolve †; `solve` gives a sextic's roots numerically", "● Groebner.jl", "◐ 100× slower; no answer for a sextic system", "● exact `Root` objects"],
    "Real solutions, parametric case classification" => ["● msolve †", "○", "◐", "● Reduce"],
    "Ideal theory (primary decomposition, radicals)" => ["● Oscar †", "○", "○", "○ ?"],
    "Exact algebraic numbers" => ["● QQBar †", "○", "◐ CRootOf", "● Root"],
    "Quantifier elimination / cylindrical decomposition" => ["○", "○", "○", "●"],
    "Linear algebra",
    "Symbolic det / inverse / eigen" => ["● fast for rational entries via Nemo †", "◐", "●", "●"],
    "Matrix calculus (symbolic matrices, Kronecker, vec)" => ["● †", "○", "◐", "◐ symbolic arrays"],
    "Control and dynamics",
    "Lie derivatives and brackets, center manifolds" => ["● †", "○", "○", "◐ ?"],
    "Lyapunov / region of attraction by sum-of-squares" => ["● †", "○ SumOfSquares.jl, separate", "○", "○ semidefinite solver only"],
    "Linear control (transfer functions, state space, LQR, Bode)" => ["○ ControlSystems.jl does the numerics", "◐ linearize", "◐ physics.control", "●"],
    "Mechanics (Lagrange, Kane)" => ["○ notebook only", "○", "●", "◐"],
    "Into numerical solvers",
    "Compiled native functions" => ["● zero-allocation, shared subexpressions computed once †", "●", "◐ lambdify", "◐ Compile"],
    "Exact Jacobians, sparse patterns, jvp into solvers" => ["● evaluates as fast as Symbolics; builds 5–7× slower †", "● fast build and evaluation", "○ in Julia; slow `lambdify`", "◐ fastest build; `Compile` dense, 1000× slower to evaluate"],
    "Events, stochastic models, inputs" => ["● †", "●", "○", "●"],
    "Component models, index reduction, tearing" => ["○", "●", "○", "●"],
    "Solutions indexed by symbolic variables" => ["● †", "●", "○", "◐"],
    "Parameter sensitivities / adjoints" => ["● via SciMLSensitivity, checked †", "●", "○", "◐"],
    "Turning Julia functions into expressions (tracing)" => ["○", "●", "n/a", "n/a"],
    "Platform",
    "Typeset output" => ["● †", "◐", "●", "●"],
    "Parallel symbolic work" => ["○ one GIAC thread", "●", "◐ one Python thread", "◐"],
    "Native in Julia" => ["● C++ engine wrapped", "●", "◐ via Python", "○ external"],
    "Licence" => ["GPL-3 (GIAC)", "MIT", "BSD", "proprietary"],
    "Maturity" => ["◐ new, on a branch", "●", "●", "●"],
])

_MARK_COLOURS = Dict('●' => "#7ec699", '◐' => "#e5c07b", '○' => "#e06c75")
_esc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
function _rating_html(r)
    r = replace(_esc(r), r"`([^`]*)`" => s"<code>\1</code>")
    m = first(r)
    haskey(_MARK_COLOURS, m) || return r
    rest = strip(r[nextind(r, 1):end])
    "<span style=\"color:$(_MARK_COLOURS[m]);font-size:1.15em\">$m</span>" *
        (isempty(rest) ? "" : " <span style=\"opacity:.75;font-size:.88em\">$rest</span>")
end

function Base.show(io::IO, ::MIME"text/html", m::CapabilityMatrix)
    print(io, """<style>
      table.capmat { border-collapse: collapse; margin: 6px 0; font-size: .93em; }
      table.capmat th, table.capmat td { padding: 6px 12px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; vertical-align: top; }
      table.capmat td.area { padding-top: 14px; font-weight: 600; border-bottom: none; }
    </style><table class="capmat"><tr><th>Capability</th>""")
    foreach(s -> print(io, "<th>", s, "</th>"), m.systems)
    print(io, "</tr>")
    for r in m.rows
        if r isa String
            print(io, "<tr><td class=\"area\" colspan=\"$(length(m.systems) + 1)\">", _esc(r), "</td></tr>")
        else
            print(io, "<tr><td>", _esc(first(r)), "</td>")
            foreach(x -> print(io, "<td>", _rating_html(x), "</td>"), last(r))
            print(io, "</tr>")
        end
    end
    print(io, "</table><p style=\"opacity:.7;font-size:.88em\">● strong · ◐ partial or limited · ○ absent or weak · † added on this branch, rather than from the GIAC engine · ? unsure</p>")
end

function Base.show(io::IO, ::MIME"text/markdown", m::CapabilityMatrix)
    println(io, "| Capability | ", join(m.systems, " | "), " |\n|", repeat(":--|", length(m.systems) + 1))
    for r in m.rows
        r isa String ? println(io, "| **", r, "** |", repeat(" |", length(m.systems))) :
                       println(io, "| ", first(r), " | ", join(last(r), " | "), " |")
    end
end
capabilities

#%% code id=harness collapsed
"One system's answer to a task, shortened, the seconds it took, and whether it is an answer."
struct Trial
    answer::String
    seconds::Union{Nothing, Float64}
    status::Symbol               # :ok, :unevaluated, :error or :absent
end
Trial(::Nothing) = Trial("not available", nothing, :absent)

"Tasks run in each system: rows of `task => [trial in each system]`."
struct Comparison
    rows::Vector{Pair{String, Vector{Trial}}}
end

shorten(s; n = 160) = (s = replace(string(s), r"\s+" => " "); length(s) <= n ? s : first(s, n) * " … ($(length(s)) characters)")

# The answer of `f` and its time: a first run compiles and fills caches, then the best of three
# runs, or one run when it takes more than half a second. `reset` clears a system's cache of
# results before each timed run. `unevaluated` tells an answer from the question given back.
# With a `limit`, a first run that takes longer is abandoned (Julia can't interrupt it, so it
# finishes in the background).
function measure(f; reset = () -> nothing, unevaluated = _ -> false, show = string, limit = 60)
    try
        if limit === nothing
            r = f()
        else
            task = Threads.@spawn f()
            timedwait(() -> istaskdone(task), limit) === :ok || return Trial("no answer within $limit s", nothing, :error)
            r = fetch(task)
        end
        reset(); t = @elapsed f()
        if t < 0.5
            t = min(t, minimum(1:3) do _
                reset(); @elapsed f()
            end)
        end
        s = show(r)
        Trial(shorten(s), t, unevaluated(s) ? :unevaluated : :ok)
    catch e
        Trial(shorten("error: " * sprint(showerror, e); n = 120), nothing, :error)
    end
end

# The comparison tables' columns: SymPy twice, called from Julia and run natively in Python
COLUMNS = ["Giac.jl", "Symbolics / MTK", "SymPy, from Julia", "SymPy, in Python", "Mathematica"]

_sympy_unevaluated(s) = occursin(r"\b(Integral|Limit)\(", s)
giac(f) = measure(f; unevaluated = s -> occursin(r"\b(integrate|limit|desolve|int)\(|Unable to|^(list)?\[\]$", s))
symbolics(f) = measure(f; unevaluated = s -> s == "nothing")
# on Python's own thread, without a limit: a task spawned from there would leave the thread
sympy(f) = py(() -> measure(f; reset = () -> SP.sympy.core.cache.clear_cache(), unevaluated = _sympy_unevaluated, limit = nothing))

# SymPy code run in a separate Python process with a time limit, for a task that may not end
# (a call into Python can't be interrupted): its answer and the time of a run after a first one,
# with SymPy's cache cleared, or none within the limit. `x, y, z, t, a` are symbols.
function sympy(code::String; limit = 30)
    python = SP.PythonCall.C.CTX.exe_path
    script = """
    import time
    from sympy import *
    from sympy.core.cache import clear_cache
    x, y, z, t, a = symbols('x y z t a')
    r = $code
    clear_cache()
    t0 = time.time()
    r = $code
    print(time.time() - t0)
    print(r)
    """
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(`$python -c $script`; stdout = out, stderr = err); wait = false)
    timedwait(() -> process_exited(p), limit; pollint = 0.1) === :ok || (kill(p); return Trial("no answer within $limit s", nothing, :error))
    success(p) || return Trial(shorten("error: " * last(split(strip(String(take!(err))), '\n')); n = 120), nothing, :error)
    t, answer = split(strip(String(take!(out))), '\n'; limit = 2)
    Trial(shorten(answer), parse(Float64, t), _sympy_unevaluated(answer) ? :unevaluated : :ok)
end

# Each task's Wolfram Language code, run in one kernel with a time limit, timed inside it after
# a first run, with the kernel's caches cleared before each run and its messages silenced
function mathematica(codes::Vector{String}; limit = 120)
    script = """
    \$Messages = {};
    SetAttributes[time, HoldAll];
    time[e_] := Module[{r, t},
        ClearSystemCache[]; r = TimeConstrained[e, $limit, \$Aborted];
        If[r === \$Aborted, Return["\$Aborted\\t-"]];
        ClearSystemCache[]; {t, r} = AbsoluteTiming[TimeConstrained[e, $limit, \$Aborted]];
        ToString[r, InputForm] <> "\\t" <> ToString[t, InputForm]];
    Print[StringRiffle[{$(join(("time[" * c * "]" for c in codes), ", "))}, "\\n%%%\\n"]]
    """
    file = tempname() * ".wls"
    write(file, script)
    out = try
        wolfram("-file", file)
    catch e
        return fill(Trial(shorten("Mathematica failed: " * sprint(showerror, e); n = 120), nothing, :error), length(codes))
    end
    map(split(out, "\n%%%\n")) do entry
        answer, t = rsplit(entry, '\t'; limit = 2)
        answer == "\$Aborted" && return Trial("no answer within $(limit) s", nothing, :error)
        Trial(shorten(answer), parse(Float64, replace(t, r"`.*" => "")),
              occursin(r"^(Integrate|Limit|DSolve)\[", answer) ? :unevaluated : :ok)
    end
end

_timefmt(t) = t === nothing ? "" : t < 1e-6 ? "$(round(Int, t * 1e9)) ns" : t < 1e-3 ? "$(round(t * 1e6; digits = 1)) μs" : t < 1 ? "$(round(t * 1e3; digits = 2)) ms" : "$(round(t; digits = 2)) s"
_STATUS = Dict(:ok => ("✓", "#7ec699"), :unevaluated => ("unevaluated", "#e5c07b"),
               :error => ("✗", "#e06c75"), :absent => ("—", "#888"),
               :wrong => ("wrong", "#e06c75"), :approximate => ("approximate", "#e5c07b"),
               :unsimplified => ("not simplified", "#e5c07b"))

# A judgement the status can't make by itself: the answer of `system` to the task starting with
# `task` is `:wrong`, `:approximate` or `:unsimplified`
function judge!(c::Comparison, task, system, status)
    row = c.rows[findfirst(r -> startswith(r.first, task), c.rows)].second
    col = findfirst(==(system), COLUMNS)
    row[col] = Trial(row[col].answer, row[col].seconds, status)
    return c
end

# Answers compared without spaces, with Python's `**` as `^`, without `*`, and with Mathematica's
# brackets as parentheses
_normal(s) = lowercase(replace(replace(s, "**" => "^"), r"[\s*]" => "", "[" => "(", "]" => ")"))

# Each answer given that is not one of its task's expected forms, marked not simplified
function expect!(c::Comparison, forms)
    for ((_, trials), ok) in zip(c.rows, forms), (i, tr) in enumerate(trials)
        tr.status === :ok && !(_normal(tr.answer) in _normal.(ok)) && (trials[i] = Trial(tr.answer, tr.seconds, :unsimplified))
    end
    return c
end

function Base.show(io::IO, ::MIME"text/html", c::Comparison)
    print(io, """<style>
      table.cmp { border-collapse: collapse; margin: 6px 0; font-size: .9em; table-layout: fixed; width: 100%; }
      table.cmp th, table.cmp td { padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; vertical-align: top; }
      table.cmp td.ans { font-family: var(--mono, monospace); font-size: .86em; overflow-wrap: anywhere; }
      table.cmp .t { display: block; margin-top: 3px; font-family: inherit; font-weight: 600; }
    </style><table class="cmp"><tr><th style="width:18%">Task</th>""")
    foreach(s -> print(io, "<th>", s, "</th>"), COLUMNS)
    print(io, "</tr>")
    for (task, trials) in c.rows
        print(io, "<tr><td>", _esc(task), "</td>")
        for tr in trials
            mark, colour = _STATUS[tr.status]
            print(io, "<td class=\"ans\">", _esc(tr.answer), "<span class=\"t\" style=\"color:$colour\">", mark, " ", _timefmt(tr.seconds), "</span></td>")
        end
        print(io, "</tr>")
    end
    print(io, "</table>")
end

function Base.show(io::IO, ::MIME"text/markdown", c::Comparison)
    println(io, "| Task | ", join(COLUMNS, " | "), " |\n|", repeat(":--|", length(COLUMNS) + 1))
    for (task, trials) in c.rows
        cells = [replace(first(tr.answer, 60), "|" => "\\|") * " (" * first(_STATUS[tr.status]) * " " * _timefmt(tr.seconds) * ")" for tr in trials]
        println(io, "| ", task, " | ", join(cells, " | "), " |")
    end
end

# SymPy's result for a task of `section` from its native run, by the task's label
native_sections = native["sections"]
function native_trial(section, label)
    r = get(get(native_sections, section, Dict()), label, nothing)
    r === nothing && return Trial("not in the native run", nothing, :absent)
    haskey(r, "build") && return Trial("first call: " * _timefmt(r["first_call"]) * ", evaluate: " * _timefmt(r["evaluate"]), r["build"], :ok)
    Trial(shorten(r["answer"]), get(r, "seconds", nothing), Symbol(r["status"]))
end

# Rows of `section` from tasks given as `label => (giac, symbolics, sympy)`: functions, `nothing`
# where a system has no such operation, or for SymPy a string of Python code run in its own
# process; the Mathematica code for each task; and SymPy's native run of the same tasks
function compare(section, tasks, wolfram::Vector{String})
    ms = mathematica(wolfram)
    Comparison([label => [giac(g), s === nothing ? Trial(nothing) : symbolics(s), sympy(p), native_trial(section, label), m]
                for ((label, (g, s, p)), m) in zip(tasks, ms)])
end

#%% code id=symbols
@giac_var x y z t a
sy = NamedTuple(n => Sym.variable(n) for n in (:x, :y, :z, :t, :a))
sp = py(() -> NamedTuple(n => SP.symbols(string(n)) for n in (:x, :y, :z, :t, :a)))
(x + y, sy.x + sy.y, py(() -> string(sp.x + sp.y)))

#%% md id=algebra_md
@md raw"""
## Polynomial algebra

Expanding, factoring and greatest common divisors: the core of any computer algebra system. Each cell gives the answer, shortened, and below it the time. Symbolics has no public `factor`, and its `gcd` takes numbers only.

By default GIAC factors further than the others: its `withsqrt` mode splits a quadratic factor with square roots, so $x^4 + 1 = (x^2 - \sqrt2 x + 1)(x^2 + \sqrt2 x + 1)$. The tasks here run with that mode off (`over_rationals`), so GIAC factors over the rationals, as the other systems do.
"""

#%% code id=algebra
# GIAC's `factor` over the rationals, as the other systems factor: its `withsqrt` mode (the tenth
# entry of `cas_setup`), which splits quadratic factors with square roots, is off while `f` runs
function over_rationals(f)
    saved = string(Giac.giac_eval("cas_setup()"))
    Giac.giac_eval("cas_setup(op(subsop(cas_setup(), 9 = 0)))")
    try
        f()
    finally
        Giac.giac_eval("cas_setup(op($saved))")
    end
end
algebra = over_rationals() do
    compare("algebra", [
    "expand (1 + x + y + z)²⁰: the number of terms, 1771" => (
        () -> length(Giac.arguments(C.expand((1 + x + y + z)^20))),
        () -> length(Sym.arguments(Sym.value(Sym.expand((1 + sy.x + sy.y + sy.z)^20)))),
        () -> length(SP.sympy.expand((1 + sp.x + sp.y + sp.z)^20).args)),
    "factor x¹²⁰ − 1" => (
        () -> C.factor(x^120 - 1),
        nothing,
        () -> SP.sympy.factor(sp.x^120 - 1)),
    "factor (x³y + 2xz − y² + 1)³ (x − y⁴ + 3z)², expanded" => (
        () -> C.factor(C.expand((x^3 * y + 2x * z - y^2 + 1)^3 * (x - y^4 + 3z)^2)),
        nothing,
        () -> SP.sympy.factor(SP.sympy.expand((sp.x^3 * sp.y + 2sp.x * sp.z - sp.y^2 + 1)^3 * (sp.x - sp.y^4 + 3sp.z)^2))),
    "gcd of (x + y + 1)¹⁰(x − y)⁵ and (x + y + 1)⁷(x + 2y)⁶, expanded" => (
        () -> C.factor(C.gcd(C.expand((x + y + 1)^10 * (x - y)^5), C.expand((x + y + 1)^7 * (x + 2y)^6))),
        () -> gcd(Sym.expand((sy.x + sy.y + 1)^10 * (sy.x - sy.y)^5), Sym.expand((sy.x + sy.y + 1)^7 * (sy.x + 2sy.y)^6)),
        () -> SP.sympy.factor(SP.sympy.gcd(SP.sympy.expand((sp.x + sp.y + 1)^10 * (sp.x - sp.y)^5), SP.sympy.expand((sp.x + sp.y + 1)^7 * (sp.x + 2sp.y)^6)))),
], [
    "Length[Expand[(1 + x + y + z)^20]]",
    "Factor[x^120 - 1]",
    "Factor[Expand[(x^3 y + 2 x z - y^2 + 1)^3 (x - y^4 + 3 z)^2]]",
    "Factor[PolynomialGCD[Expand[(x + y + 1)^10 (x - y)^5], Expand[(x + y + 1)^7 (x + 2 y)^6]]]",
])
end

#%% md id=calculus_md
@md raw"""
## Calculus

Integrals, indefinite and definite, limits and series. A definite integral or limit has its value in the task, to check the answers against. An integral given back unevaluated is marked so. Symbolics has no `integrate`.

GIAC's integrator is the fastest here by one to two orders of magnitude, and answers every indefinite and improper integral. Only Mathematica evaluates $\int_0^1 \log x \log(1-x)\,dx$: GIAC gives it back, and SymPy returns a partly evaluated expression with an infinite imaginary part. Symbolics' `limit` handles neither `sin` nor a non-integer power; its `taylor` gives the series.
"""

#%% code id=calculus
∞ = Giac.giac_eval("+infinity")
calculus = compare("calculus", [
    "∫ x³ eˣ sin x dx" => (
        () -> C.integrate(x^3 * exp(x) * sin(x), x), nothing,
        () -> SP.sympy.integrate(sp.x^3 * exp(sp.x) * sin(sp.x), sp.x)),
    "∫ 1 / (x⁴ + 1) dx" => (
        () -> C.integrate(1 / (x^4 + 1), x), nothing,
        () -> SP.sympy.integrate(1 / (sp.x^4 + 1), sp.x)),
    "∫ sec³ x dx" => (
        () -> C.integrate(1 / cos(x)^3, x), nothing,
        () -> SP.sympy.integrate(1 / cos(sp.x)^3, sp.x)),
    "∫₀^∞ sin x / x dx = π/2" => (
        () -> C.integrate(sin(x) / x, x, 0, ∞), nothing,
        () -> SP.sympy.integrate(sin(sp.x) / sp.x, (sp.x, 0, SP.sympy.oo))),
    "∫₀^∞ e^(−x²) dx = √π / 2" => (
        () -> C.integrate(exp(-x^2), x, 0, ∞), nothing,
        () -> SP.sympy.integrate(exp(-sp.x^2), (sp.x, 0, SP.sympy.oo))),
    "∫₀¹ log x log(1 − x) dx = 2 − π²/6" => (
        () -> C.integrate(log(x) * log(1 - x), x, 0, 1), nothing,
        () -> SP.sympy.integrate(log(sp.x) * log(1 - sp.x), (sp.x, 0, 1))),
    "lim x→0 (sin x − x) / x³ = −1/6" => (
        () -> C.limit((sin(x) - x) / x^3, x, 0),
        () -> Sym.limit(Sym.value((sin(sy.x) - sy.x) / sy.x^3), Sym.value(sy.x), 0),
        () -> SP.sympy.limit((sin(sp.x) - sp.x) / sp.x^3, sp.x, 0)),
    "lim x→∞ (1 + a/x)ˣ = eᵃ" => (
        () -> C.limit((1 + a / x)^x, x, ∞),
        () -> Sym.limit(Sym.value((1 + sy.a / sy.x)^sy.x), Sym.value(sy.x), Inf),
        () -> SP.sympy.limit((1 + sp.a / sp.x)^sp.x, sp.x, SP.sympy.oo)),
    "tan x to order x⁹" => (
        () -> C.series(tan(x), x, 0, 9),
        () -> Sym.taylor(tan(sy.x), sy.x, 0, 0:9),
        () -> SP.sympy.series(tan(sp.x), sp.x, 0, 10)),
], [
    "Integrate[x^3 Exp[x] Sin[x], x]",
    "Integrate[1/(x^4 + 1), x]",
    "Integrate[Sec[x]^3, x]",
    "Integrate[Sin[x]/x, {x, 0, Infinity}]",
    "Integrate[Exp[-x^2], {x, 0, Infinity}]",
    "Integrate[Log[x] Log[1 - x], {x, 0, 1}]",
    "Limit[(Sin[x] - x)/x^3, x -> 0]",
    "Limit[(1 + a/x)^x, x -> Infinity]",
    "Series[Tan[x], {x, 0, 9}]",
])

#%% md id=equations_md
@md raw"""
## Equations

A Gröbner basis of the cyclic-5 system, in the graded reverse lexicographic order (its number of elements), a polynomial system, a transcendental equation whose solution is Lambert's W, and a linear system with a parameter.

All four find the 20-element Gröbner basis; GIAC is fastest, then Groebner.jl behind Symbolics, then Mathematica, with SymPy a hundred times slower. For $x^2 + y^2 = 1,\ x^3 = y$, GIAC's `solve` gives the two real solutions numerically, of the six; Mathematica gives all six exactly as `Root` objects and Symbolics as roots of a polynomial, while SymPy finds nothing in 30 seconds (in an earlier run, nothing in 15 minutes). Symbolics' `symbolic_solve` cannot take $x e^x = 2$.
"""

#%% code id=equations
cyclic(v) = [[sum(prod(v[mod1(i + j, 5)] for j in 0:k-1) for i in 1:5) for k in 1:4]; prod(v) - 1]
equations = let
    gv = [Giac.giac_identifier("c$i") for i in 1:5]
    sv = [Sym.variable(Symbol("c$i")) for i in 1:5]
    wolfram_cyclic = "{" * join(string.(cyclic(gv)), ", ") * "}"
    sympy_cyclic = "[" * join(replace.(string.(cyclic(gv)), "^" => "**"), ", ") * "]"
    compare("equations", [
        "Gröbner basis of cyclic-5, graded reverse lexicographic: the number of elements" => (
            () -> length(C.gbasis(cyclic(gv), gv, Giac.giac_eval("revlex"))),
            () -> length(Sym.groebner_basis(cyclic(sv))),
            "(lambda c1, c2, c3, c4, c5: len(groebner($sympy_cyclic, c1, c2, c3, c4, c5, order='grevlex').exprs))(*symbols('c1:6'))"),
        "x² + y² = 1, x³ = y: the solutions" => (
            () -> C.solve([x^2 + y^2 ~ 1, x^3 ~ y], [x, y]),
            () -> Sym.symbolic_solve([sy.x^2 + sy.y^2 ~ 1, sy.x^3 ~ sy.y], [sy.x, sy.y]),
            "solve([x**2 + y**2 - 1, x**3 - y], [x, y])"),
        "x eˣ = 2" => (
            () -> C.solve(x * exp(x) ~ 2, x),
            () -> Sym.symbolic_solve(sy.x * exp(sy.x) ~ 2, sy.x),
            "solve(x*exp(x) - 2, x)"),
        "a x + y = 1, x − a y = 2" => (
            () -> C.solve([a * x + y ~ 1, x - a * y ~ 2], [x, y]),
            () -> Sym.symbolic_linear_solve([sy.a * sy.x + sy.y ~ 1, sy.x - sy.a * sy.y ~ 2], [sy.x, sy.y]),
            "solve([a*x + y - 1, x - a*y - 2], [x, y])"),
    ], [
        "Length[GroebnerBasis[$wolfram_cyclic, {c1, c2, c3, c4, c5}, MonomialOrder -> DegreeReverseLexicographic]]",
        "Solve[{x^2 + y^2 == 1, x^3 == y}, {x, y}]",
        "Solve[x Exp[x] == 2, x]",
        "Solve[{a x + y == 1, x - a y == 2}, {x, y}]",
    ])
end
# the two real solutions, numerically, of the six
judge!(equations, "x² + y² = 1", "Giac.jl", :approximate)

#%% md id=odes_md
@md"""
## Ordinary differential equations in closed form

From a linear equation with constant coefficients to one whose solutions are Bessel and Airy functions. Symbolics solves linear equations with constant coefficients and Bernoulli and Clairaut equations.

Mathematica solves all five. GIAC and SymPy solve the linear equations and the logistic equation, GIAC in well under a millisecond and SymPy in tens to hundreds; SymPy also finds Airy's functions, and GIAC says it is unable to. Neither solves the Riccati equation, whose solution needs Bessel functions: GIAC returns no solution and SymPy fails with an error. Symbolics solves only the first, and its answer to the Riccati equation, a constant, is wrong.
"""

#%% code id=odes
odes = let
    @giac_var f(t)
    Dg = Differential(t)
    Ds = Sym.Differential(sy.t)
    # SymPy's dsolve of `lhs = rhs` in f(t), written with `f`
    dsolve(lhs, rhs) = "(lambda f: dsolve(Eq($lhs, $rhs), f))(Function('f')(t))"
    compare("odes", [
        "y″ + y = sin t" => (
            () -> C.desolve(Dg(Dg(f)) + f ~ sin(t), t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) + sy.y ~ sin(sy.t), sy.y, sy.t),
            dsolve("f.diff(t, 2) + f", "sin(t)")),
        "y″ − 2y′ + y = eᵗ / t" => (
            () -> C.desolve(Dg(Dg(f)) - 2Dg(f) + f ~ exp(t) / t, t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) - 2Ds(sy.y) + sy.y ~ exp(sy.t) / sy.t, sy.y, sy.t),
            dsolve("f.diff(t, 2) - 2*f.diff(t) + f", "exp(t)/t")),
        "y′ = y (1 − y)" => (
            () -> C.desolve(Dg(f) ~ f * (1 - f), t, f),
            () -> Sym.symbolic_solve_ode(Ds(sy.y) ~ sy.y * (1 - sy.y), sy.y, sy.t),
            dsolve("f.diff(t)", "f*(1 - f)")),
        "y′ = t² + y²" => (
            () -> C.desolve(Dg(f) ~ t^2 + f^2, t, f),
            () -> Sym.symbolic_solve_ode(Ds(sy.y) ~ sy.t^2 + sy.y^2, sy.y, sy.t),
            dsolve("f.diff(t)", "t**2 + f**2")),
        "y″ + t y = 0 (Airy)" => (
            () -> C.desolve(Dg(Dg(f)) + t * f ~ 0, t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) + sy.t * sy.y ~ 0, sy.y, sy.t),
            dsolve("f.diff(t, 2) + t*f", "0")),
    ], [
        "DSolve[y''[t] + y[t] == Sin[t], y[t], t]",
        "DSolve[y''[t] - 2 y'[t] + y[t] == Exp[t]/t, y[t], t]",
        "DSolve[y'[t] == y[t] (1 - y[t]), y[t], t]",
        "DSolve[y'[t] == t^2 + y[t]^2, y[t], t]",
        "DSolve[y''[t] + t y[t] == 0, y[t], t]",
    ])
end
# a constant, whose derivative 0 is not t² + y²
judge!(odes, "y′ = t² + y²", "Symbolics / MTK", :wrong)

#%% md id=simplify_md
@md raw"""
## Simplification and assumptions

Each system's own simplifier, with assumptions where it has them: Giac's `assuming` or a declared variable (`@giac_var n::integer`), a positive, real or integer symbol in SymPy, `Simplify[…, x > 0]` in Mathematica. Symbolics has no assumptions, so it simplifies the expression as given. An answer that isn't the simplified form is marked so.

GIAC and SymPy reach every simplified form; Mathematica's `Simplify` leaves $\sqrt{5 + 2\sqrt6}$ alone (its `FullSimplify` denests it). Symbolics, without assumptions, keeps $\sqrt{x^2}$ as $|x|$ and $\sin \pi n$ as it is, and doesn't see the trigonometric identity.
"""

#%% code id=simplification
simplification = let
    S = SP.sympy
    xpos, xreal = py(() -> (S.Symbol("x"; positive = true), S.Symbol("x"; real = true)))
    compare("simplification", [
        "sin⁴x − cos⁴x + cos 2x = 0" => (
            () -> C.simplify(sin(x)^4 - cos(x)^4 + cos(2x)),
            () -> Sym.simplify(sin(sy.x)^4 - cos(sy.x)^4 + cos(2sy.x)),
            () -> S.simplify(sin(sp.x)^4 - cos(sp.x)^4 + cos(2sp.x))),
        "(x⁶ − 1) / (x² − 1) = x⁴ + x² + 1" => (
            () -> C.simplify((x^6 - 1) / (x^2 - 1)),
            () -> Sym.simplify((sy.x^6 - 1) / (sy.x^2 - 1)),
            () -> S.simplify((sp.x^6 - 1) / (sp.x^2 - 1))),
        "√(5 + 2√6) = √2 + √3" => (
            () -> C.simplify(sqrt(5 + 2sqrt(Giac.giac_eval("6")))),
            () -> Sym.simplify(sqrt(5 + 2sqrt(Sym.Num(6)))),
            () -> S.sqrtdenest(sqrt(5 + 2sqrt(S.Integer(6))))),
        "√(x²) = x for x > 0" => (
            () -> assuming(() -> sqrt(x^2), x > 0),
            () -> Sym.simplify(sqrt(sy.x^2)),
            () -> S.simplify(sqrt(xpos^2))),
        "log(eˣ) = x for real x" => (
            () -> let; @giac_var xr::real; C.simplify(log(exp(xr))); end,
            () -> Sym.simplify(log(exp(sy.x))),
            () -> S.simplify(log(exp(xreal)))),
        "sin(π n) = 0 for integer n" => (
            () -> let; @giac_var n::integer; sin(n * Giac.giac_eval("pi")); end,
            () -> Sym.simplify(sin(Sym.variable(:n) * π)),
            () -> S.simplify(sin(S.Symbol("n"; integer = true) * S.pi))),
    ], [
        "Simplify[Sin[x]^4 - Cos[x]^4 + Cos[2 x]]",
        "Simplify[(x^6 - 1)/(x^2 - 1)]",
        "Simplify[Sqrt[5 + 2 Sqrt[6]]]",
        "Simplify[Sqrt[x^2], x > 0]",
        "Simplify[Log[Exp[x]], Element[x, Reals]]",
        "Simplify[Sin[Pi n], Element[n, Integers]]",
    ])
end
expect!(simplification, [["0"], ["x^4+x^2+1", "1+x^2+x^4"], ["sqrt(2)+sqrt(3)", "sqrt(3)+sqrt(2)"],
                         ["x"], ["x", "xr"], ["0"]])

#%% md id=scale_md
@md"""
## From a model to a compiled Jacobian

The step a numerical solver needs from a CAS: the Jacobian of a large sparse model, compiled to a function. The model is the Brusselator on a ring of $N$ cells, $2N$ states. Each system takes its natural route: Giac's SciML extension (the exact sparse Jacobian, compiled), Symbolics' `sparsejacobian` and `build_function`, SymPy's `jacobian` and `lambdify`, Mathematica's `D` and `Compile`.

Each system first builds a small model, so no time below includes compiling the builders themselves. Then each cell gives the time of the symbolic build and code generation, and below it the first call, which compiles the generated code (in Julia), and the best of 20 evaluations.

The two Julia systems compile to the same kind of code, writing the sparse matrix's stored values directly, and evaluate it in the same time, a few hundred nanoseconds for 400 states. Building takes Giac five to seven times as long as Symbolics, because each derivative is a call into GIAC, where Symbolics differentiates in Julia. Mathematica builds fastest, but its compiled function returns a dense matrix and evaluates a thousand times slower. SymPy is slowest at both.
"""

#%% code id=scale
using SciMLBase: ODEFunction
using SparseArrays: sparse
# the Brusselator's right-hand sides in `u`, `v`, with A = 1, B = 3, α = 0.02, in any system's
# expressions
function brusselator(u, v)
    N = length(u)
    lap(w, i) = w[mod1(i - 1, N)] - 2w[i] + w[mod1(i + 1, N)]
    vcat([1 + u[i]^2 * v[i] - 4u[i] + lap(u, i) / 50 for i in 1:N], [3u[i] - u[i]^2 * v[i] + lap(v, i) / 50 for i in 1:N])
end
state(N) = vcat(1 .+ sin.(2π .* (1:N) ./ N), fill(3.0, N))

# The symbolic build and code generation, then the first call, which compiles the generated code,
# and the best of 20 evaluations, as a trial timed by its build
function build_trial(build, evaluate)
    try
        tb = @elapsed f = build()
        tc = @elapsed evaluate(f)
        te = minimum(@elapsed(evaluate(f)) for _ in 1:20)
        Trial("first call: " * _timefmt(tc) * ", evaluate: " * _timefmt(te), tb, :ok)
    catch e
        Trial(shorten("error: " * sprint(showerror, e); n = 120), nothing, :error)
    end
end

function giac_jacobian(N)
    u, v = [Giac.giac_identifier("u$i") for i in 1:N], [Giac.giac_identifier("v$i") for i in 1:N]
    x0 = state(N)
    build_trial(() -> (f = ODEFunction(brusselator(u, v), vcat(u, v); sparse = true, tgrad = false, paramjac = false);
                       (f, copy(f.jac_prototype))),
                ((f, J),) -> f.jac(J, x0, nothing, 0.0))
end
function symbolics_jacobian(N)
    u, v = [Sym.variable(Symbol("u$i")) for i in 1:N], [Sym.variable(Symbol("v$i")) for i in 1:N]
    x0 = state(N)
    build_trial(() -> begin
                    J = Sym.sparsejacobian(brusselator(u, v), vcat(u, v))
                    (Sym.build_function(J, vcat(u, v); expression = Val(false))[2], similar(J, Float64))
                end,
                ((f, J),) -> f(J, x0))
end
# in a Python process of its own: `lambdify` reads its caller's Python frame, which a call from
# Julia doesn't have
function sympy_jacobian(N; limit = 300)
    python = SP.PythonCall.C.CTX.exe_path
    script = """
    import math, time, timeit
    from sympy import *
    from sympy.core.cache import clear_cache
    N = $N
    u, v = symbols(f'u1:{N + 1}'), symbols(f'v1:{N + 1}')
    w = list(u) + list(v)
    lap = lambda q, i: q[(i - 1) % N] - 2*q[i] + q[(i + 1) % N]
    F = [1 + u[i]**2*v[i] - 4*u[i] + lap(u, i)/50 for i in range(N)] + [3*u[i] - u[i]**2*v[i] + lap(v, i)/50 for i in range(N)]
    x0 = [1 + math.sin(2*math.pi*(i + 1)/N) for i in range(N)] + [3.0]*N
    lambdify([w[:2]], Matrix(F[:2]).jacobian(w[:2]), 'numpy')([1.0, 2.0])     # a small model first
    clear_cache()
    t0 = time.time(); f = lambdify([w], Matrix(F).jacobian(w), 'numpy'); tb = time.time() - t0
    t0 = time.time(); f(x0); tc = time.time() - t0
    print(tb); print(tc); print(min(timeit.repeat(lambda: f(x0), number = 1, repeat = 20)))
    """
    out = IOBuffer()
    p = run(pipeline(`$python -c $script`; stdout = out, stderr = devnull); wait = false)
    timedwait(() -> process_exited(p), limit; pollint = 0.1) === :ok || (kill(p); return Trial("no answer within $limit s", nothing, :error))
    success(p) || return Trial("error", nothing, :error)
    tb, tc, te = parse.(Float64, split(strip(String(take!(out))), '\n'))
    Trial("first call: " * _timefmt(tc) * ", evaluate: " * _timefmt(te), tb, :ok)
end
# compiled as a function of the 2N states themselves, after a small model warms up the kernel
function mathematica_jacobian(N)
    out = wl("""
        \$Messages = {};
        model[n_] := Module[{u, v, lap},
            u = Table[Symbol["u" <> ToString[i]], {i, n}]; v = Table[Symbol["v" <> ToString[i]], {i, n}];
            lap[q_, i_] := q[[Mod[i - 2, n] + 1]] - 2 q[[i]] + q[[Mod[i, n] + 1]];
            {Join[u, v],
             Join[Table[1 + u[[i]]^2 v[[i]] - 4 u[[i]] + lap[u, i]/50, {i, n}], Table[3 u[[i]] - u[[i]]^2 v[[i]] + lap[v, i]/50, {i, n}]],
             Join[Table[1. + Sin[2 Pi i/n], {i, n}], Table[3., {n}]]}];
        jacobian[{w_, F_, x0_}] := Compile[Evaluate[Map[{#, _Real} &, w]], Evaluate[D[F, {w}]]];
        small = model[3]; jacobian[small] @@ Last[small];
        {w, F, x0} = big = model[$N];
        tb = First[AbsoluteTiming[cf = jacobian[big]]];
        tc = First[AbsoluteTiming[cf @@ x0]];
        te = First[RepeatedTiming[cf @@ x0]];
        ToString[{tb, tc, te}, InputForm]""")
    tb, tc, te = parse.(Float64, split(replace(strip(out), r"[{}\s]" => "", r"`[^,]*" => ""), ","))
    Trial("first call: " * _timefmt(tc) * ", evaluate: " * _timefmt(te), tb, :ok)
end

# a small model first, so that no build below includes compiling the builders themselves
giac_jacobian(5), symbolics_jacobian(5)
scale = Comparison(["Brusselator, N = $N ($(2N) states)" => [giac_jacobian(N), symbolics_jacobian(N), sympy_jacobian(N),
                                                                native_trial("scale", "Brusselator, N = $N ($(2N) states)"), mathematica_jacobian(N)]
                    for N in (50, 200)])

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   BenchmarkTools 1.8.0 6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf
#   DedicatedThreads 0.1.0 9d6577c3-9664-4440-a53f-cec765ebc045
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   Groebner 0.10.11 0b43b601-686d-58a3-8a1c-6623616c7cd4
#   SciMLBase 3.58.0 0bca4576-84f4-4d90-8ffe-ffa030f20462
#   SparseArrays 1.13.0 2f01184e-e22b-5df5-ae63-d93ebab69eaf
#   SymPyPythonCall 0.5.3 bc8888f7-b21e-4b7c-a06a-5d9c9496438c
#   Symbolics 7.48.0 0c5d862f-8b57-4792-8d23-62f2024744c7
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = e00c1f95-b52d-42d0-9994-884253e0d6bf
# ╚═╡
