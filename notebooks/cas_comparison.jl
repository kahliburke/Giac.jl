try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Giac.jl among computer algebra systems

How the Giac.jl of this branch compares with Symbolics.jl and ModelingToolkit, SymJ, SymPy, and Mathematica. The matrix first rates each capability from the systems' documented features. The sections after it run the same tasks in all five systems, side by side, with the answers and the time each took, so the ratings rest on evidence where a test exists.

SymJ is a computer algebra system written in Julia, with GruntzLimit for limits and RischIntegrate for integrals. It is not registered; the notebook runs it from a checkout, whose commit the environment records.

Each Julia system is timed after a first run that compiles, as the best of three runs. Giac.jl runs on a development build of its C++ wrapper, from the branch that goes with this one (expression trees passed to GIAC directly rather than printed and parsed, command lookups cached); the setup output records which. Mathematica runs through `wolframscript`, one kernel per section, timed inside the kernel with `AbsoluteTiming` after a first run and with its caches cleared.

SymPy has two columns. **SymPy, from Julia** calls it through SymPyPythonCall, on a thread of its own, after a first run and with its cache cleared; a task that may not end runs instead in a separate Python process with a time limit, since a call into Python can't be interrupted. **SymPy, in Python** is the same task run natively by `cas_comparison_sympy.py`, beside this notebook: a first run, then the best of three with the cache cleared, each task under an alarm. Its results are kept in `~/.cache/giac/benchmarks/` and read here; the script runs again only when it changes.

Answers are typeset, each system's in its own notation, and checked against one another by their values at one point, $x = 7/10$, $y = 2/5$, $z = 3/10$, $a = 13/10$: an antiderivative by its change from $x = 2/5$ to $x = 7/10$, a series by its polynomial part, and solutions as a set. Where a task states its value, answers are checked against it; otherwise against the value that most systems agree on, if at least two do. An answer that differs is marked wrong. Closed-form ODE solutions aren't checked, since each system names and places its constants differently.
"""

#%% code id=python nocache eval_barrier
using DedicatedThreads: DedicatedPool, spawn
# Python belongs to the OS thread that starts it, and a cell may run on any thread. So SymPy gets a
# thread of its own: SymPyPythonCall is loaded there, which starts Python on it, and every SymPy
# call is sent there with `py`. The module is bound from that load rather than by an `import`,
# which the notebook would run early, on another thread. The pool is kept in `Main`, so that a
# re-run of this cell sends its calls to the same thread, where Python already runs. An error is
# written out there and raised again as text: a Python exception shown on another thread crashes.
PYTHON = isdefined(Main, :PYTHON_POOL) ? Main.PYTHON_POOL :
         Core.eval(Main, :(const PYTHON_POOL = $(DedicatedPool(1; name = :python))))
function py(f)
    ok, r = fetch(spawn(() -> try (true, f()) catch e; (false, sprint(showerror, e)) end, PYTHON))
    ok ? r : error(r)
end
const SP = py(() -> Base.require(Base.PkgId(Base.UUID("bc8888f7-b21e-4b7c-a06a-5d9c9496438c"), "SymPyPythonCall")))
py(() -> Threads.threadid()), py(() -> string(SP.sympy.factor(SP.sympy.sympify("x^12 - 1"))))

#%% code id=setup
using Giac, GiacSlate, LinearAlgebra, BenchmarkTools
import Symbolics as Sym, Groebner, Latexify, libgiac_julia_jll
# SymJ exports `Sym`, the name Symbolics has here, so it is imported by a name of its own
import SymJ as SJ, RischIntegrate
const C = Giac.Commands
# Giac's C++ wrapper: a local build, as its checkout and commit, or the released one
giac_wrapper = let dir = dirname(dirname(dirname(libgiac_julia_jll.libgiac_wrapper_path)))
    isdir(joinpath(dir, ".git")) ? replace(dir, homedir() => "~") * " at " * readchomp(`git -C $dir rev-parse --short HEAD`) :
                                   "released " * Giac.GiacCxxBindings.wrapper_version()
end
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
(; giac = pkgdir(Giac), giac_wrapper, symbolics = pkgversion(Sym), symj = pkgversion(SJ), sympy = py(() -> string(SP.sympy.__version__)), mathematica = wl("\$VersionNumber"))

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

#%% md id=environment_md
@md"""
## Environment

What the timings were measured on, read when the notebook runs: each system's exact version and the libraries under it, the machine, and how each system is timed. Giac.jl's commit is marked *modified* when its sources differ from that commit. The performance plot is labelled with the short versions and the machine.
"""

#%% code id=environment
import Pkg, Dates
using Markdown
# `short_versions`: each column's system and version, for the plot; `machine_summary`: the
# machine and Julia in a line
short_versions, machine_summary, environment = let
    sh(cmd) = try readchomp(cmd) catch; "" end
    git(dir, args...) = sh(`git -C $dir $args`)
    # the packages of the notebook's environment, by name
    deps = Dict(d.name => d for d in values(Pkg.dependencies()))
    ver(name) = haskey(deps, name) && deps[name].version !== nothing ? string(deps[name].version) : "—"
    giac_dir = pkgdir(Giac)
    giac_commit = git(giac_dir, "rev-parse", "--short", "HEAD") *
                  (isempty(git(giac_dir, "status", "--porcelain", "--", "src", "ext")) ? "" : " (modified)")
    giac_engine = Giac.GiacCxxBindings.giac_version()
    sympy_version, python_version = py(() -> (string(SP.sympy.__version__), string(SP.PythonCall.pyimport("platform").python_version())))
    wl_version, wl_system, wl_processors = split(wl("StringRiffle[{\$Version, \$SystemID, ToString[\$ProcessorCount]}, \"\\n\"]"), '\n')
    cpu = Sys.cpu_info()
    macos = sh(`sw_vers -productVersion`)
    software = [
        "Giac.jl" => "$(pkgversion(Giac)), $(git(giac_dir, "rev-parse", "--abbrev-ref", "HEAD")) at $giac_commit",
        "GIAC (engine)" => "$giac_engine, GIAC_jll $(ver("GIAC_jll"))",
        "Giac's C++ wrapper" => "$giac_wrapper (libgiac_julia_jll $(ver("libgiac_julia_jll")), CxxWrap $(ver("CxxWrap")))",
        "Symbolics" => "Symbolics $(ver("Symbolics")), SymbolicUtils $(ver("SymbolicUtils")), Groebner $(ver("Groebner")), Latexify $(ver("Latexify"))",
        "SymJ" => "SymJ $(pkgversion(SJ)), $(git(pkgdir(SJ), "rev-parse", "--abbrev-ref", "HEAD")) at $(git(pkgdir(SJ), "rev-parse", "--short", "HEAD")), GruntzLimit $(ver("GruntzLimit")), RischIntegrate $(ver("RischIntegrate"))",
        "SciML" => "SciMLBase $(ver("SciMLBase")), RuntimeGeneratedFunctions $(ver("RuntimeGeneratedFunctions"))",
        "SymPy, from Julia" => "SymPy $sympy_version on Python $python_version, through SymPyPythonCall $(ver("SymPyPythonCall")) and PythonCall $(ver("PythonCall"))",
        "SymPy, in Python" => "SymPy $(native["sympy"]) on Python $(native["python"]), results of $(native["date"])",
        "Mathematica" => "$wl_version ($wl_system), kernel sees $wl_processors processors",
        "Julia" => "$(VERSION), $(Threads.nthreads()) threads, SymPy on its own thread (DedicatedThreads $(ver("DedicatedThreads")))",
    ]
    machine = [
        "Computer" => sh(`sysctl -n hw.model`),
        "CPU" => "$(cpu[1].model), $(sh(`sysctl -n hw.perflevel0.physicalcpu`)) performance and $(sh(`sysctl -n hw.perflevel1.physicalcpu`)) efficiency cores",
        "Memory" => "$(round(Int, Sys.total_memory() / 2^30)) GiB",
        "Operating system" => "macOS $macos ($(sh(`sw_vers -buildVersion`))), $(Sys.MACHINE)",
        "Power" => something(match(r"'(.*?)'", sh(`pmset -g batt`)), (; captures = ["unknown"])).captures[1],
        # other work on the machine shares its cores: the load average as the tasks start
        "Load when read" => join(round.(Sys.loadavg(); digits = 2), ", ") * " (1, 5, 15 min)",
    ]
    timing = [
        "Giac.jl, Symbolics, SymJ" => "after a first run that compiles, the best of three, or one run over half a second; 60 s limit",
        "SymPy, from Julia" => "the same, with SymPy's cache cleared before each run; 30 s limit for a task in a Python process of its own",
        "SymPy, in Python" => "after a first run, the best of three with the cache cleared, each task under an alarm",
        "Mathematica" => "one run after a first, with `ClearSystemCache[]` before each, by `AbsoluteTiming`; 120 s limit",
        "Run" => string(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")),
    ]
    cell(v) = replace(v, "|" => "\\|", "_" => "\\_", "*" => "\\*")
    table(title, rows) = "| $title | |\n|:--|:--|\n" * join(["| $k | $(cell(v)) |" for (k, v) in rows], "\n")
    short = Dict(
        "Giac.jl" => "Giac.jl $(pkgversion(Giac)) at $giac_commit (GIAC $giac_engine)",
        "Symbolics / MTK" => "Symbolics $(ver("Symbolics"))",
        "SymJ" => "SymJ $(pkgversion(SJ)) at $(git(pkgdir(SJ), "rev-parse", "--short", "HEAD"))",
        "SymPy, from Julia" => "SymPy $sympy_version from Julia (Python $python_version)",
        "SymPy, in Python" => "SymPy $(native["sympy"]) in Python $(native["python"])",
        "Mathematica" => "Mathematica $(first(split(wl_version)))")
    (short, "$(cpu[1].model), macOS $macos, Julia $VERSION, $(Dates.format(Dates.now(), "yyyy-mm-dd"))",
     Markdown.parse(join([table("Software", software), table("Machine", machine), table("Timing", timing)], "\n\n")))
end
environment

#%% code id=matrix collapsed
"The capability matrix: headings, and rows of a capability's rating in each system."
struct CapabilityMatrix
    systems::Vector{String}
    rows::Vector{Any}            # a heading `String`, or `capability => [rating, …]`
    untested::Set{String}        # capabilities rated from documentation, with no test below
end

SYSTEMS = ["Giac.jl", "Symbolics / MTK", "SymJ", "SymPy", "Mathematica"]
capabilities = CapabilityMatrix(SYSTEMS, Any[
    "Core algebra",
    "Expand / factor polynomials" => ["● factors with square roots by default (`withsqrt`)", "○ no public `factor`; `gcd` takes numbers only", "● fastest at multivariate polynomials; seconds for x¹²⁰ − 1", "●", "● fastest"],
    "Simplification" => ["● six of six tests; rule-based `simplest` search †", "○ two of six tests", "● six of six tests", "● six of six tests", "● `Simplify` five of six; FullSimplify"],
    "Assumptions" => ["● three of three tests; declared on variables, functions, matrices †", "○ none", "● three of three tests; `@assume`, `assuming`", "● three of three tests", "● three of three tests"],
    "Pattern matching / rewrite rules" => ["● †", "●", "●", "◐", "●"],
    "Calculus",
    "Differentiation" => ["● in Julia, on the expression tree †", "●", "●", "●", "●"],
    "Indefinite integration" => ["● three of three tests, fastest by 10–100×", "○ no `integrate`", "● three of three tests, RischIntegrate; fast", "● three of three tests", "● three of three tests"],
    "Definite integrals, special functions" => ["● three of three tests; Beta and Gamma integrals differentiated in their parameters †", "○", "◐ two of three; no dilogarithm limits", "◐ two of three; a garbled third", "● three of three tests"],
    "Limits, series" => ["●", "◐ `limit` fails on `sin`; `taylor` works", "● Gruntz; Puiseux and asymptotic series", "●", "●"],
    "Closed-form ODE solutions" => ["◐ three of five tests, each under 1 ms", "○ one of five; one wrong answer", "◐ three of five tests; Airy's as a series only", "● four of five tests", "● five of five tests"],
    "Equations and algebra",
    "Polynomial systems (Gröbner bases)" => ["● fastest on cyclic-5; msolve †; `cSolve` gives a sextic's roots numerically", "● Groebner.jl", "● as fast as GIAC on cyclic-5; the sextic system in radicals", "◐ 100× slower; no answer for a sextic system", "● exact `Root` objects"],
    "Real solutions, parametric case classification" => ["● msolve †", "○", "◐ `real_roots`, answers by cases", "◐", "● Reduce"],
    "Ideal theory (primary decomposition, radicals)" => ["● Oscar †", "○", "○", "○", "○ ?"],
    "Exact algebraic numbers" => ["● QQBar †", "○", "◐ `minimal_polynomial`", "◐ CRootOf", "● Root"],
    "Quantifier elimination / cylindrical decomposition" => ["○", "○", "○", "○", "●"],
    "Linear algebra",
    "Symbolic det / inverse / eigen" => ["● fast for rational entries via Nemo †", "◐", "● also Jordan form, matrix exponential", "●", "●"],
    "Matrix calculus (symbolic matrices, Kronecker, vec)" => ["● †", "○", "○", "◐", "◐ symbolic arrays"],
    "Control and dynamics",
    "Lie derivatives and brackets, center manifolds" => ["● †", "○", "○", "○", "◐ ?"],
    "Lyapunov / region of attraction by sum-of-squares" => ["● †", "○ SumOfSquares.jl, separate", "○", "○", "○ semidefinite solver only"],
    "Linear control (transfer functions, state space, LQR, Bode)" => ["○ ControlSystems.jl does the numerics", "◐ linearize", "◐ Laplace transforms", "◐ physics.control", "●"],
    "Mechanics (Lagrange, Kane)" => ["○ notebook only", "○", "○", "●", "◐"],
    "Into numerical solvers",
    "Compiled native functions" => ["● zero-allocation, shared subexpressions computed once †", "●", "● `lambdify`, shared subexpressions computed once", "◐ lambdify", "◐ Compile"],
    "Exact Jacobians, sparse patterns, jvp into solvers" => ["● builds and evaluates as fast as Symbolics †", "● fast build and evaluation", "◐ no sparse Jacobian; slow to compile hundreds of entries, slow calls", "○ in Julia; slow `lambdify`", "◐ fastest build; `Compile` dense, 1000× slower to evaluate"],
    "Events, stochastic models, inputs" => ["● †", "●", "○", "○", "●"],
    "Component models, index reduction, tearing" => ["○", "●", "○", "○", "●"],
    "Solutions indexed by symbolic variables" => ["● †", "●", "○", "○", "◐"],
    "Parameter sensitivities / adjoints" => ["● via SciMLSensitivity, checked †", "●", "○", "○", "◐"],
    "Turning Julia functions into expressions (tracing)" => ["○", "●", "◐ generic code only: `Sym` is not a `Number`", "n/a", "n/a"],
    "Platform",
    "Typeset output" => ["● †", "◐", "● Latexify", "●", "●"],
    "Parallel symbolic work" => ["○ one GIAC thread", "●", "◐ ?", "◐ one Python thread", "◐"],
    "Native in Julia" => ["● C++ engine wrapped", "●", "●", "◐ via Python", "○ external"],
    "Licence" => ["GPL-3 (GIAC)", "MIT", "MIT", "BSD", "proprietary"],
    "Maturity" => ["◐ new, on a branch", "●", "◐ new, unregistered", "●", "●"],
], Set([
    "Pattern matching / rewrite rules", "Real solutions, parametric case classification",
    "Ideal theory (primary decomposition, radicals)", "Exact algebraic numbers",
    "Quantifier elimination / cylindrical decomposition", "Symbolic det / inverse / eigen",
    "Matrix calculus (symbolic matrices, Kronecker, vec)", "Lie derivatives and brackets, center manifolds",
    "Lyapunov / region of attraction by sum-of-squares", "Linear control (transfer functions, state space, LQR, Bode)",
    "Mechanics (Lagrange, Kane)", "Events, stochastic models, inputs", "Component models, index reduction, tearing",
    "Solutions indexed by symbolic variables", "Parameter sensitivities / adjoints",
    "Turning Julia functions into expressions (tracing)", "Parallel symbolic work"]))

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
            print(io, "<tr><td>", _esc(first(r)))
            first(r) in m.untested && print(io, "<br><span style=\"opacity:.6;font-size:.82em\">not compared here</span>")
            print(io, "</td>")
            foreach(x -> print(io, "<td>", _rating_html(x), "</td>"), last(r))
            print(io, "</tr>")
        end
    end
    print(io, "</table><p style=\"opacity:.7;font-size:.88em\">● strong · ◐ partial or limited · ○ absent or weak · † added on this branch, rather than from the GIAC engine · ? unsure. A row marked <i>not compared here</i> is rated from the systems' documentation, with no test in this notebook.</p>")
end

function Base.show(io::IO, ::MIME"text/markdown", m::CapabilityMatrix)
    println(io, "| Capability | ", join(m.systems, " | "), " |\n|", repeat(":--|", length(m.systems) + 1))
    for r in m.rows
        r isa String ? println(io, "| **", r, "** |", repeat(" |", length(m.systems))) :
                       println(io, "| ", first(r), first(r) in m.untested ? " (not compared here)" : "", " | ", join(last(r), " | "), " |")
    end
end
capabilities

#%% code id=harness collapsed
"""
One system's answer to a task: shortened as text, typeset, the seconds it took, whether it is an
answer, its values at the comparison point, and how they compare with the other systems'.
"""
mutable struct Trial
    answer::String
    seconds::Union{Nothing, Float64}
    status::Symbol               # :ok, :unevaluated, :error, :absent, or a judgement below
    latex::String                # empty when there is none
    values::Union{Nothing, Vector{Vector{ComplexF64}}}   # solutions, each a vector of components
    check::String                # "agrees", "differs", "2 of 6 solutions", or empty if unchecked
end
Trial(answer, seconds, status; latex = "", values = nothing) = Trial(answer, seconds, status, latex, values, "")
Trial(::Nothing) = Trial("not available", nothing, :absent)

"Tasks run in each system: rows of `task => [trial in each system]`."
struct Comparison
    rows::Vector{Pair{String, Vector{Trial}}}
end

shorten(s; n = 160) = (s = replace(string(s), r"\s+" => " "); length(s) <= n ? s : first(s, n) * " … ($(length(s)) characters)")

# The point at which answers are compared, exact. An antiderivative is compared by its change from
# x0 to x, a series by its polynomial part, solutions as a set.
POINT = Dict("x" => 7//10, "x0" => 2//5, "xr" => 7//10, "y" => 2//5, "z" => 3//10, "t" => 3//5, "a" => 13//10, "n" => 3)
_text(v) = v isa Rational ? "$(numerator(v))/$(denominator(v))" : string(v)

# The answer of `f` and its time: a first run compiles and fills caches, then the best of three
# runs, or one run when it takes more than half a second. `reset` clears a system's cache of
# results before each timed run. `unevaluated` tells an answer from the question given back.
# `describe` gives the answer typeset and its values, untimed. With a `limit`, a first run that
# takes longer is abandoned (Julia can't interrupt it, so it finishes in the background).
function measure(f; reset = () -> nothing, unevaluated = _ -> false, show = string, describe = _ -> ("", nothing), limit = 60)
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
        latex, values = try describe(r) catch; ("", nothing) end
        Trial(shorten(s), t, unevaluated(s) ? :unevaluated : :ok; latex, values)
    catch e
        Trial(shorten("error: " * sprint(showerror, e); n = 120), nothing, :error)
    end
end

# ── Giac: the answer at the point, by exact substitution and 20-digit evaluation
_giac_tex(r) = r isa GiacExpr ? String(strip(sprint(show, MIME"text/latex"(), r), ['$', '\n', ' '])) : string(r)
function _giac_at(e; over...)
    e isa Number && return e
    p = merge(POINT, Dict(string(k) => v for (k, v) in over))
    names = [string(n) for n in C.lname(e)]
    isempty(names) || (e = C.subst(e, Giac.giac_eval("[" * join(names, ",") * "]"), Giac.giac_eval("[" * join(_text.(getindex.(Ref(p), names)), ",") * "]")))
    to_julia(C.evalf(e, 20))
end
function giac_values(r, kind)
    kind === :at && return [[_giac_at(r)]]
    kind === :antiderivative && return [[_giac_at(r) - _giac_at(r; x = POINT["x0"])]]
    kind === :series && return [[_giac_at(C.convert(r, Giac.giac_eval("polynom")))]]
    kind === :solutions && return [s isa AbstractVector ? collect(s) : [s] for s in _giac_at(r)]
    nothing
end

# ── Symbolics: the answer at the point, in floating point; typeset by Latexify
function _symbolics_at(e)
    e isa Union{Integer, AbstractFloat, Rational} && return e
    v = Sym.value(Sym.substitute(e, Dict(v => float(POINT[string(v)]) for v in Sym.get_variables(e))))
    v isa Number ? v : Sym.symbolic_to_float(v)
end
function symbolics_values(r, kind)
    kind in (:at, :series) && return [[_symbolics_at(r)]]
    kind === :solutions && return [s isa AbstractDict ? [_symbolics_at(s[k]) for k in sort!(collect(keys(s)); by = string)] :
                                   s isa AbstractVector ? _symbolics_at.(s) : [_symbolics_at(s)] for s in r]
    nothing
end
_symbolics_tex(r) = r === nothing ? "" : String(strip(Latexify.latexify(r; env = :raw)))

# ── SymJ: the answer at the point, by exact substitution and evaluation; typeset by Latexify.
# An answer by cases is taken in the case that holds at the point, and an antiderivative that
# RischIntegrate can only evaluate (sums over roots, ₂F₁ terms) is called at the point.
_symj_point(; over...) = [SJ.Sym(Symbol(k)) => v for (k, v) in merge(POINT, Dict(string(k) => v for (k, v) in over)) if k != "x0"]
function _symj_at(e; over...)
    e isa Number && return e
    e isa RischIntegrate.IntegralWithRootSums && return e(get(over, :x, POINT["x"]))
    e isa SJ.Cases && return _symj_at(e(_symj_point(; over...)...); over...)
    v = SJ.N(SJ.subs(e, _symj_point(; over...)...))
    v isa Number ? v : error("not a number: $v")
end
function symj_values(r, kind)
    r isa SJ.Cases && (r = r(_symj_point()...))
    kind === :at && return [[_symj_at(r)]]
    kind === :antiderivative && return [[_symj_at(r) - _symj_at(r; x = POINT["x0"])]]
    kind === :series && return [[_symj_at(SJ.Sym(r))]]
    kind === :solutions && return [s isa AbstractDict ? [_symj_at(s[k]) for k in sort!(collect(keys(s)); by = string)] : [_symj_at(s)] for s in r]
    nothing
end
_symj_tex(r) = String(strip(Latexify.latexify(r; env = :raw)))

# ── SymPy: typeset and evaluated by `describe` in the script beside this notebook, on Python's
# thread. `_sympy_describe` must be called there; `sympy_describe` sends it there.
# (imported again on a re-run, so that an edit to the script takes effect)
const SCRIPT = py(() -> (sys = SP.PythonCall.pyimport("sys");
                         dir = joinpath(pkgdir(Giac), "notebooks");
                         dir in SP.PythonCall.pyconvert(Vector{String}, sys.path) || sys.path.insert(0, dir);
                         SP.PythonCall.pyimport("importlib").reload(SP.PythonCall.pyimport("cas_comparison_sympy"))))
const PY_POINT = py(() -> SCRIPT.point(Dict(k => _text(v) for (k, v) in POINT)))
# SymPyPythonCall's own wrapper of a Python object, unwrapped
_python(r) = r isa SP.SymPyCore.SymbolicObject ? getfield(r, :o) : r isa Union{AbstractArray, Tuple} ? map(_python, r) : r
function _sympy_describe(r, kind)
    d = TOML.parse(SP.PythonCall.pyconvert(String, SCRIPT.describe(_python(r), string(kind), PY_POINT)))
    d["latex"], haskey(d, "values") ? [[ComplexF64(c[1], c[2]) for c in s] for s in d["values"]] : nothing
end
sympy_describe(srepr, kind) = try py(() -> _sympy_describe(SCRIPT.revive(srepr), kind)) catch; ("", nothing) end

# The comparison tables' columns: SymPy twice, called from Julia and run natively in Python
COLUMNS = ["Giac.jl", "Symbolics / MTK", "SymJ", "SymPy, from Julia", "SymPy, in Python", "Mathematica"]

# an answer's values from `f`, as complex numbers, or none if they can't be found
_values(f) = r -> try (v = f(r); v === nothing ? nothing : [ComplexF64.(s) for s in v]) catch; nothing end
_sympy_unevaluated(s) = occursin(r"\b(Integral|Limit)\(", s)
giac(f, kind) = measure(f; unevaluated = s -> occursin(r"\b(integrate|limit|desolve|int)\(|Unable to|^(list)?\[\]$", s),
                        describe = r -> (_giac_tex(r), _values(r -> giac_values(r, kind))(r)))
symbolics(f, kind) = measure(f; unevaluated = s -> s == "nothing",
                             describe = r -> (try _symbolics_tex(r) catch; "" end, _values(r -> symbolics_values(r, kind))(r)))
# SymJ throws where it finds no answer
symj(f, kind) = measure(f; describe = r -> (try _symj_tex(r) catch; "" end, _values(r -> symj_values(r, kind))(r)))
# on Python's own thread, without a limit: a task spawned from there would leave the thread
sympy(f, kind) = py(() -> measure(f; reset = () -> SP.sympy.core.cache.clear_cache(), unevaluated = _sympy_unevaluated,
                                     describe = r -> _sympy_describe(r, kind), limit = nothing))

# SymPy code run in a separate Python process with a time limit, for a task that may not end
# (a call into Python can't be interrupted): its answer and the time of a run after a first one,
# with SymPy's cache cleared, or none within the limit. `x, y, z, t, a` are symbols.
function sympy(code::String, kind; limit = 30)
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
    print(srepr(r))
    print(r)
    """
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(`$python -c $script`; stdout = out, stderr = err); wait = false)
    timedwait(() -> process_exited(p), limit; pollint = 0.1) === :ok || (kill(p); return Trial("no answer within $limit s", nothing, :error))
    success(p) || return Trial(shorten("error: " * last(split(strip(String(take!(err))), '\n')); n = 120), nothing, :error)
    t, srepr, answer = split(strip(String(take!(out))), '\n'; limit = 3)
    latex, values = sympy_describe(srepr, kind)
    Trial(shorten(answer), parse(Float64, t), _sympy_unevaluated(answer) ? :unevaluated : :ok; latex, values)
end

# Each task's Wolfram Language code, run in one kernel with a time limit, timed inside it after
# a first run, with the kernel's caches cleared before each run and its messages silenced; then
# typeset, and evaluated at the point for the comparison of its kind
function mathematica(codes::Vector{String}, kinds; limit = 120)
    rules = join(("$k -> $(_text(v))" for (k, v) in POINT if k != "x0"), ", ")
    script = """
    \$Messages = {};
    point = {$rules};
    num[e_] := With[{v = N[e /. point]}, If[NumericQ[v], {Re[v], Im[v]}, {Indeterminate, 0}]];
    vals["at", r_] := {{num[r]}};
    vals["antiderivative", r_] := {{num[r] - num[r /. x -> $(_text(POINT["x0"]))]}};
    vals["series", r_] := {{num[Normal[r]]}};
    vals["solutions", r_] := Map[num, Values[r], {2}];
    vals[_, _] := None;
    (* TeXForm without a front end leaves a Root empty: written root_k(p(z)) instead *)
    tex[r_] := StringReplace[ToString[TeXForm[r /. Root[p_, k_, ___] :> Subscript["root", k][p[z]]]], "\\n" -> " "];
    SetAttributes[time, HoldFirst];
    time[e_, kind_] := Module[{r, t},
        ClearSystemCache[]; r = TimeConstrained[e, $limit, \$Aborted];
        If[r === \$Aborted, Return["\$Aborted\\t\\t\\t-"]];
        ClearSystemCache[]; {t, r} = AbsoluteTiming[TimeConstrained[e, $limit, \$Aborted]];
        StringRiffle[{ToString[r, InputForm], tex[r],
                      ToString[N[vals[kind, r]], InputForm], ToString[t, InputForm]}, "\\t"]];
    Print[StringRiffle[{$(join(("time[$c, \"$k\"]" for (c, k) in zip(codes, kinds)), ", "))}, "\\n%%%\\n"]]
    """
    file = tempname() * ".wls"
    write(file, script)
    out = try
        wolfram("-file", file)
    catch e
        return fill(Trial(shorten("Mathematica failed: " * sprint(showerror, e); n = 120), nothing, :error), length(codes))
    end
    map(split(out, "\n%%%\n")) do entry
        answer, latex, vals, t = split(entry, '\t')
        answer == "\$Aborted" && return Trial("no answer within $(limit) s", nothing, :error)
        Trial(shorten(answer), parse(Float64, replace(t, r"`.*" => "")),
              occursin(r"^(Integrate|Limit|DSolve)\[", answer) ? :unevaluated : :ok;
              latex = String(latex), values = _wolfram_values(vals))
    end
end
# values written by Mathematica, `{{{re, im}, …}, …}` in InputForm, read as TOML
function _wolfram_values(s)
    s == "None" && return nothing
    s = replace(s, "{" => "[", "}" => "]", "*^" => "e", "Indeterminate" => "nan", r"(\d)\.(?!\d)" => s"\1.0")
    [[ComplexF64(c[1], c[2]) for c in sol] for sol in TOML.parse("v = " * s)["v"]]
end

# ── Agreement: answers are compared by their values at the point, solutions as sets
_close(u, v) = abs(u - v) <= 1e-6 * max(1, abs(u), abs(v))
_same(p, q) = length(p) == length(q) && all(_close.(p, q))
# how many of the solutions `vs` are among `ref`, each matched once
function _matched(vs, ref)
    free = trues(length(ref))
    count(vs) do v
        j = findfirst(k -> free[k] && _same(v, ref[k]), eachindex(ref))
        j === nothing ? false : (free[j] = false; true)
    end
end
function _check(vs, ref)
    m = _matched(vs, ref)
    m == length(vs) == length(ref) ? "agrees" : m == length(vs) ? "$m of $(length(ref)) solutions" : "differs"
end
_SYSTEM = [1, 2, 3, 4, 4, 5]     # the columns' systems: SymPy is two of them

# Each answer's values compared with the `expected` value, or else with the values most systems
# support (Mathematica's on a tie), when at least two systems do: a system supports values it
# gives, or of which it gives some solutions. An answer that differs is wrong.
function agree!(trials, expected)
    usable = [i for (i, t) in enumerate(trials) if t.values !== nothing && t.status === :ok]
    isempty(usable) && return trials
    ref = if expected !== nothing
        [[ComplexF64(expected)]]
    else
        support(i) = length(unique(_SYSTEM[j] for j in usable if _check(trials[j].values, trials[i].values) != "differs"))
        best = argmax(i -> (support(i), COLUMNS[i] == "Mathematica"), usable)
        support(best) >= 2 || return trials
        trials[best].values
    end
    for i in usable
        trials[i].check = _check(trials[i].values, ref)
        trials[i].check == "differs" && (trials[i].status = :wrong)
    end
    trials
end

_timefmt(t) = t === nothing ? "" : t < 1e-6 ? "$(round(Int, t * 1e9)) ns" : t < 1e-3 ? "$(round(t * 1e6; digits = 1)) μs" : t < 1 ? "$(round(t * 1e3; digits = 2)) ms" : "$(round(t; digits = 2)) s"
_STATUS = Dict(:ok => ("✓", "#7ec699"), :unevaluated => ("unevaluated", "#e5c07b"),
               :error => ("✗", "#e06c75"), :absent => ("—", "#888"),
               :wrong => ("wrong", "#e06c75"), :approximate => ("approximate", "#e5c07b"),
               :unsimplified => ("not simplified", "#e5c07b"))
_check_colour(c) = c == "agrees" ? "#7ec699" : c == "differs" ? "#e06c75" : "#e5c07b"

# A judgement the status can't make by itself: the answer of `system` to the task starting with
# `task` is `:wrong`, `:approximate` or `:unsimplified`
function judge!(c::Comparison, task, system, status)
    row = c.rows[findfirst(r -> startswith(r.first, task), c.rows)].second
    row[findfirst(==(system), COLUMNS)].status = status
    return c
end

# Answers compared without spaces, with Python's `**` as `^`, without `*`, and with Mathematica's
# brackets as parentheses
_normal(s) = lowercase(replace(replace(s, "**" => "^"), r"[\s*]" => "", "[" => "(", "]" => ")"))

# Each answer given that is not one of its task's expected forms, marked not simplified
function expect!(c::Comparison, forms)
    for ((_, trials), ok) in zip(c.rows, forms), tr in trials
        tr.status === :ok && !(_normal(tr.answer) in _normal.(ok)) && (tr.status = :unsimplified)
    end
    return c
end

function Base.show(io::IO, ::MIME"text/html", c::Comparison)
    print(io, """<style>
      table.cmp { border-collapse: collapse; margin: 6px 0; font-size: .9em; table-layout: fixed; width: 100%; }
      table.cmp th, table.cmp td { padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; vertical-align: top; }
      table.cmp td.ans { font-family: var(--mono, monospace); font-size: .86em; overflow-wrap: anywhere; }
      table.cmp .tex { font-family: initial; font-size: 1.05em; max-height: 9em; overflow: auto; }
      table.cmp .t { display: block; margin-top: 3px; font-family: inherit; font-weight: 600; }
      table.cmp .k { font-weight: 400; }
    </style><table class="cmp"><tr><th style="width:16%">Task</th>""")
    foreach(s -> print(io, "<th>", s, "</th>"), COLUMNS)
    print(io, "</tr>")
    for (task, trials) in c.rows
        print(io, "<tr><td>", _esc(task), "</td>")
        for tr in trials
            mark, colour = _STATUS[tr.status]
            body = isempty(tr.latex) || tr.status in (:error, :absent) ? _esc(tr.answer) :
                   "<div class=\"tex\">\$\\displaystyle " * _esc(tr.latex) * "\$</div>"
            check = isempty(tr.check) ? "" : " · <span class=\"k\" style=\"color:$(_check_colour(tr.check))\">$(tr.check)</span>"
            print(io, "<td class=\"ans\">", body, "<span class=\"t\" style=\"color:$colour\">", mark, " ", _timefmt(tr.seconds), check, "</span></td>")
        end
        print(io, "</tr>")
    end
    print(io, "</table>")
end

function Base.show(io::IO, ::MIME"text/markdown", c::Comparison)
    println(io, "| Task | ", join(COLUMNS, " | "), " |\n|", repeat(":--|", length(COLUMNS) + 1))
    for (task, trials) in c.rows
        cells = [replace(first(tr.answer, 60), "|" => "\\|") * " (" * join(filter(!isempty, [first(_STATUS[tr.status]), _timefmt(tr.seconds), tr.check]), " ") * ")" for tr in trials]
        println(io, "| ", task, " | ", join(cells, " | "), " |")
    end
end

# SymPy's result for a task of `section` from its native run, by the task's label, typeset and
# evaluated from its `srepr`
native_sections = native["sections"]
function native_trial(section, label, kind = :none)
    r = get(get(native_sections, section, Dict()), label, nothing)
    r === nothing && return Trial("not in the native run", nothing, :absent)
    haskey(r, "build") && return Trial("first call: " * _timefmt(r["first_call"]) * ", evaluate: " * _timefmt(r["evaluate"]), r["build"], :ok)
    latex, values = haskey(r, "srepr") ? sympy_describe(r["srepr"], kind) : ("", nothing)
    Trial(shorten(r["answer"]), get(r, "seconds", nothing), Symbol(r["status"]); latex, values)
end

# Rows of `section` from tasks given as `label => (check, giac, symbolics, symj, sympy)`. `check` is how
# answers are compared: `:at` (a value at the point), `:antiderivative`, `:series`, `:solutions`
# or `:none`, or `kind => expected` with the value the task states. Then functions, `nothing`
# where a system has no such operation, or for SymPy a string of Python code run in its own
# process; the Mathematica code for each task; and SymPy's native run of the same tasks.
_kind(c::Symbol) = (c, nothing)
_kind(c::Pair) = (c.first, c.second)
function compare(section, tasks, wolfram::Vector{String})
    checks = [_kind(first(t)) for (_, t) in tasks]
    ms = mathematica(wolfram, string.(first.(checks)))
    absent(f, run) = f === nothing ? Trial(nothing) : run(f)
    Comparison([label => agree!([giac(g, kind), absent(s, f -> symbolics(f, kind)), absent(j, f -> symj(f, kind)), sympy(p, kind),
                                 native_trial(section, label, kind), m], expected)
                for ((label, (_, g, s, j, p)), m, (kind, expected)) in zip(tasks, ms, checks)])
end

#%% code id=symbols
@giac_var x y z t a
sy = NamedTuple(n => Sym.variable(n) for n in (:x, :y, :z, :t, :a))
sj = NamedTuple(n => SJ.Sym(n) for n in (:x, :y, :z, :t, :a))
sp = py(() -> NamedTuple(n => SP.symbols(string(n)) for n in (:x, :y, :z, :t, :a)))
(x + y, sy.x + sy.y, sj.x + sj.y, py(() -> string(sp.x + sp.y)))

#%% md id=algebra_md
@md raw"""
## Polynomial algebra

Expanding, factoring and greatest common divisors: the core of any computer algebra system. Each cell gives the answer, shortened, and below it the time. Symbolics has no public `factor`, and its `gcd` takes numbers only.

SymJ, written in Julia, expands, factors the multivariate polynomial and finds the gcd fastest, but takes seconds to factor $x^{120} - 1$, which the others factor in milliseconds or less.

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
    "expand (1 + x + y + z)²⁰: the number of terms, 1771" => (:at => 1771,
        () -> length(Giac.arguments(C.expand((1 + x + y + z)^20))),
        () -> length(Sym.arguments(Sym.value(Sym.expand((1 + sy.x + sy.y + sy.z)^20)))),
        () -> length(SJ.TermInterface.arguments(SJ.expand((1 + sj.x + sj.y + sj.z)^20))),
        () -> length(SP.sympy.expand((1 + sp.x + sp.y + sp.z)^20).args)),
    "factor x¹²⁰ − 1" => (:at,
        () -> C.factor(x^120 - 1),
        nothing,
        () -> SJ.factor(sj.x^120 - 1),
        () -> SP.sympy.factor(sp.x^120 - 1)),
    "factor (x³y + 2xz − y² + 1)³ (x − y⁴ + 3z)², expanded" => (:at,
        () -> C.factor(C.expand((x^3 * y + 2x * z - y^2 + 1)^3 * (x - y^4 + 3z)^2)),
        nothing,
        () -> SJ.factor(SJ.expand((sj.x^3 * sj.y + 2sj.x * sj.z - sj.y^2 + 1)^3 * (sj.x - sj.y^4 + 3sj.z)^2)),
        () -> SP.sympy.factor(SP.sympy.expand((sp.x^3 * sp.y + 2sp.x * sp.z - sp.y^2 + 1)^3 * (sp.x - sp.y^4 + 3sp.z)^2))),
    "gcd of (x + y + 1)¹⁰(x − y)⁵ and (x + y + 1)⁷(x + 2y)⁶, expanded" => (:at,
        () -> C.factor(C.gcd(C.expand((x + y + 1)^10 * (x - y)^5), C.expand((x + y + 1)^7 * (x + 2y)^6))),
        () -> gcd(Sym.expand((sy.x + sy.y + 1)^10 * (sy.x - sy.y)^5), Sym.expand((sy.x + sy.y + 1)^7 * (sy.x + 2sy.y)^6)),
        () -> SJ.factor(gcd(SJ.expand((sj.x + sj.y + 1)^10 * (sj.x - sj.y)^5), SJ.expand((sj.x + sj.y + 1)^7 * (sj.x + 2sj.y)^6))),
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

GIAC and SymJ (through RischIntegrate) are the fastest integrators here, by one to two orders of magnitude over SymPy and Mathematica; GIAC is faster on most tasks, SymJ on $\int x^3 e^x \sin x\,dx$, and both answer every indefinite and improper integral. GIAC's own `integrate` gives $\int_0^1 \log x \log(1-x)\,dx$ back, since its antiderivative needs the dilogarithm; Giac.jl then finds it as a derivative of the Beta integral $\int_0^1 x^s (1-x)^r\,dx$ in $s$ and $r$. Mathematica evaluates it too, SymPy returns a partly evaluated expression with an infinite imaginary part, and SymJ finds the antiderivative but can't take its limit at $x = 1$, as GruntzLimit has no rules for the dilogarithm. Symbolics' `limit` handles neither `sin` nor a non-integer power; its `taylor` gives the series.
"""

#%% code id=calculus
∞ = Giac.giac_eval("+infinity")
calculus = compare("calculus", [
    "∫ x³ eˣ sin x dx" => (:antiderivative,
        () -> C.integrate(x^3 * exp(x) * sin(x), x), nothing,
        () -> RischIntegrate.integrate(sj.x^3 * exp(sj.x) * sin(sj.x), sj.x),
        () -> SP.sympy.integrate(sp.x^3 * exp(sp.x) * sin(sp.x), sp.x)),
    "∫ 1 / (x⁴ + 1) dx" => (:antiderivative,
        () -> C.integrate(1 / (x^4 + 1), x), nothing,
        () -> RischIntegrate.integrate(1 / (sj.x^4 + 1), sj.x),
        () -> SP.sympy.integrate(1 / (sp.x^4 + 1), sp.x)),
    "∫ sec³ x dx" => (:antiderivative,
        () -> C.integrate(1 / cos(x)^3, x), nothing,
        () -> RischIntegrate.integrate(1 / cos(sj.x)^3, sj.x),
        () -> SP.sympy.integrate(1 / cos(sp.x)^3, sp.x)),
    "∫₀^∞ sin x / x dx = π/2" => (:at => π/2,
        () -> C.integrate(sin(x) / x, x, 0, ∞), nothing,
        () -> RischIntegrate.integrate(sin(sj.x) / sj.x, (sj.x, 0, Inf)),
        () -> SP.sympy.integrate(sin(sp.x) / sp.x, (sp.x, 0, SP.sympy.oo))),
    "∫₀^∞ e^(−x²) dx = √π / 2" => (:at => √π / 2,
        () -> C.integrate(exp(-x^2), x, 0, ∞), nothing,
        () -> RischIntegrate.integrate(exp(-sj.x^2), (sj.x, 0, Inf)),
        () -> SP.sympy.integrate(exp(-sp.x^2), (sp.x, 0, SP.sympy.oo))),
    "∫₀¹ log x log(1 − x) dx = 2 − π²/6" => (:at => 2 - π^2 / 6,
        () -> C.integrate(log(x) * log(1 - x), x, 0, 1), nothing,
        () -> RischIntegrate.integrate(log(sj.x) * log(1 - sj.x), (sj.x, 0, 1)),
        () -> SP.sympy.integrate(log(sp.x) * log(1 - sp.x), (sp.x, 0, 1))),
    "lim x→0 (sin x − x) / x³ = −1/6" => (:at => -1/6,
        () -> C.limit((sin(x) - x) / x^3, x, 0),
        () -> Sym.limit(Sym.value((sin(sy.x) - sy.x) / sy.x^3), Sym.value(sy.x), 0),
        () -> SJ.limit((sin(sj.x) - sj.x) / sj.x^3, sj.x => 0),
        () -> SP.sympy.limit((sin(sp.x) - sp.x) / sp.x^3, sp.x, 0)),
    "lim x→∞ (1 + a/x)ˣ = eᵃ" => (:at => exp(1.3),
        () -> C.limit((1 + a / x)^x, x, ∞),
        () -> Sym.limit(Sym.value((1 + sy.a / sy.x)^sy.x), Sym.value(sy.x), Inf),
        () -> SJ.limit((1 + sj.a / sj.x)^sj.x, sj.x => Inf),
        () -> SP.sympy.limit((1 + sp.a / sp.x)^sp.x, sp.x, SP.sympy.oo)),
    "tan x to order x⁹" => (:series,
        () -> C.series(tan(x), x, 0, 9),
        () -> Sym.taylor(tan(sy.x), sy.x, 0, 0:9),
        () -> SJ.series(tan(sj.x), sj.x, 0, 10),
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

All five find the 20-element Gröbner basis; GIAC and SymJ are fastest, then Groebner.jl behind Symbolics, then Mathematica, with SymPy a hundred times slower. For $x^2 + y^2 = 1,\ x^3 = y$, GIAC's `cSolve` (its `solve` gives only the real solutions) gives all six numerically, since it solves exactly only up to degree four and doesn't see that $x^6 + x^2 - 1$ is a cubic in $x^2$; Mathematica gives all six exactly as `Root` objects, SymJ in radicals (Cardano's formula for the cubic in $x^2$), and Symbolics as roots of a polynomial, while SymPy finds nothing in 30 seconds (in an earlier run, nothing in 15 minutes). Symbolics' `symbolic_solve` cannot take $x e^x = 2$. SymJ answers the linear system with a parameter by cases: no solution where $a^2 + 1 = 0$, the usual one otherwise.
"""

#%% code id=equations
cyclic(v) = [[sum(prod(v[mod1(i + j, 5)] for j in 0:k-1) for i in 1:5) for k in 1:4]; prod(v) - 1]
equations = let
    gv = [Giac.giac_identifier("c$i") for i in 1:5]
    sv = [Sym.variable(Symbol("c$i")) for i in 1:5]
    jv = [SJ.Sym(Symbol("c$i")) for i in 1:5]
    wolfram_cyclic = "{" * join(string.(cyclic(gv)), ", ") * "}"
    sympy_cyclic = "[" * join(replace.(string.(cyclic(gv)), "^" => "**"), ", ") * "]"
    compare("equations", [
        "Gröbner basis of cyclic-5, graded reverse lexicographic: the number of elements" => (:at => 20,
            () -> length(C.gbasis(cyclic(gv), gv, Giac.giac_eval("revlex"))),
            () -> length(Sym.groebner_basis(cyclic(sv))),
            () -> length(SJ.groebner(cyclic(jv), jv...; order = :grevlex)),
            "(lambda c1, c2, c3, c4, c5: len(groebner($sympy_cyclic, c1, c2, c3, c4, c5, order='grevlex').exprs))(*symbols('c1:6'))"),
        "x² + y² = 1, x³ = y: the solutions" => (:solutions,
            () -> C.cSolve([x^2 + y^2 ~ 1, x^3 ~ y], [x, y]),
            () -> Sym.symbolic_solve([sy.x^2 + sy.y^2 ~ 1, sy.x^3 ~ sy.y], [sy.x, sy.y]),
            () -> SJ.solve([sj.x^2 + sj.y^2 ~ 1, sj.x^3 ~ sj.y], [sj.x, sj.y]),
            "solve([x**2 + y**2 - 1, x**3 - y], [x, y])"),
        "x eˣ = 2" => (:solutions,
            () -> C.solve(x * exp(x) ~ 2, x),
            () -> Sym.symbolic_solve(sy.x * exp(sy.x) ~ 2, sy.x),
            () -> SJ.solve(sj.x * exp(sj.x) ~ 2, sj.x),
            "solve(x*exp(x) - 2, x)"),
        "a x + y = 1, x − a y = 2" => (:solutions,
            () -> C.solve([a * x + y ~ 1, x - a * y ~ 2], [x, y]),
            # its one solution, in a list as the others give it
            () -> [Sym.symbolic_linear_solve([sy.a * sy.x + sy.y ~ 1, sy.x - sy.a * sy.y ~ 2], [sy.x, sy.y])],
            # by cases: none where a² + 1 = 0
            () -> SJ.solve([sj.a * sj.x + sj.y ~ 1, sj.x - sj.a * sj.y ~ 2], [sj.x, sj.y]),
            "solve([a*x + y - 1, x - a*y - 2], [x, y])"),
    ], [
        "Length[GroebnerBasis[$wolfram_cyclic, {c1, c2, c3, c4, c5}, MonomialOrder -> DegreeReverseLexicographic]]",
        "Solve[{x^2 + y^2 == 1, x^3 == y}, {x, y}]",
        "Solve[x Exp[x] == 2, x]",
        "Solve[{a x + y == 1, x - a y == 2}, {x, y}]",
    ])
end
# the six solutions, numerically
judge!(equations, "x² + y² = 1", "Giac.jl", :approximate)

#%% md id=odes_md
@md"""
## Ordinary differential equations in closed form

From a linear equation with constant coefficients to one whose solutions are Bessel and Airy functions. Symbolics solves linear equations with constant coefficients and Bernoulli and Clairaut equations.

Mathematica solves all five. GIAC, SymJ and SymPy solve the linear equations and the logistic equation, GIAC in well under a millisecond, SymJ in milliseconds and SymPy in tens to hundreds; SymPy also finds Airy's functions, and GIAC says it is unable to. Neither solves the Riccati equation, whose solution needs Bessel functions: GIAC returns no solution and SymPy fails with an error. SymJ has no method for either the Riccati or the Airy equation (it gives Airy's solutions as a series, with `series = n`). Symbolics solves only the first, and its answer to the Riccati equation, a constant, is wrong.
"""

#%% code id=odes
odes = let
    @giac_var f(t)
    Dg = Differential(t)
    Ds = Sym.Differential(sy.t)
    SJ.@syms y()
    tj = sj.t
    # SymPy's dsolve of `lhs = rhs` in f(t), written with `f`
    dsolve(lhs, rhs) = "(lambda f: dsolve(Eq($lhs, $rhs), f))(Function('f')(t))"
    compare("odes", [
        "y″ + y = sin t" => (:none,
            () -> C.desolve(Dg(Dg(f)) + f ~ sin(t), t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) + sy.y ~ sin(sy.t), sy.y, sy.t),
            () -> SJ.dsolve(y''(tj) + y(tj) ~ sin(tj), y(tj)),
            dsolve("f.diff(t, 2) + f", "sin(t)")),
        "y″ − 2y′ + y = eᵗ / t" => (:none,
            () -> C.desolve(Dg(Dg(f)) - 2Dg(f) + f ~ exp(t) / t, t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) - 2Ds(sy.y) + sy.y ~ exp(sy.t) / sy.t, sy.y, sy.t),
            () -> SJ.dsolve(y''(tj) - 2y'(tj) + y(tj) ~ exp(tj) / tj, y(tj)),
            dsolve("f.diff(t, 2) - 2*f.diff(t) + f", "exp(t)/t")),
        "y′ = y (1 − y)" => (:none,
            () -> C.desolve(Dg(f) ~ f * (1 - f), t, f),
            () -> Sym.symbolic_solve_ode(Ds(sy.y) ~ sy.y * (1 - sy.y), sy.y, sy.t),
            () -> SJ.dsolve(y'(tj) ~ y(tj) * (1 - y(tj)), y(tj)),
            dsolve("f.diff(t)", "f*(1 - f)")),
        "y′ = t² + y²" => (:none,
            () -> C.desolve(Dg(f) ~ t^2 + f^2, t, f),
            () -> Sym.symbolic_solve_ode(Ds(sy.y) ~ sy.t^2 + sy.y^2, sy.y, sy.t),
            () -> SJ.dsolve(y'(tj) ~ tj^2 + y(tj)^2, y(tj)),
            dsolve("f.diff(t)", "t**2 + f**2")),
        "y″ + t y = 0 (Airy)" => (:none,
            () -> C.desolve(Dg(Dg(f)) + t * f ~ 0, t, f),
            () -> Sym.symbolic_solve_ode(Ds(Ds(sy.y)) + sy.t * sy.y ~ 0, sy.y, sy.t),
            () -> SJ.dsolve(y''(tj) + tj * y(tj) ~ 0, y(tj)),
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

GIAC, SymJ and SymPy reach every simplified form; Mathematica's `Simplify` leaves $\sqrt{5 + 2\sqrt6}$ alone (its `FullSimplify` denests it). Symbolics, without assumptions, keeps $\sqrt{x^2}$ as $|x|$ and $\sin \pi n$ as it is, and doesn't see the trigonometric identity.
"""

#%% code id=simplification
simplification = let
    S = SP.sympy
    xpos, xreal = py(() -> (S.Symbol("x"; positive = true), S.Symbol("x"; real = true)))
    compare("simplification", [
        "sin⁴x − cos⁴x + cos 2x = 0" => (:at => 0,
            () -> C.simplify(sin(x)^4 - cos(x)^4 + cos(2x)),
            () -> Sym.simplify(sin(sy.x)^4 - cos(sy.x)^4 + cos(2sy.x)),
            () -> SJ.simplify(sin(sj.x)^4 - cos(sj.x)^4 + cos(2sj.x)),
            () -> S.simplify(sin(sp.x)^4 - cos(sp.x)^4 + cos(2sp.x))),
        "(x⁶ − 1) / (x² − 1) = x⁴ + x² + 1" => (:at => 0.7^4 + 0.7^2 + 1,
            () -> C.simplify((x^6 - 1) / (x^2 - 1)),
            () -> Sym.simplify((sy.x^6 - 1) / (sy.x^2 - 1)),
            () -> SJ.simplify((sj.x^6 - 1) / (sj.x^2 - 1)),
            () -> S.simplify((sp.x^6 - 1) / (sp.x^2 - 1))),
        "√(5 + 2√6) = √2 + √3" => (:at => √2 + √3,
            () -> C.simplify(sqrt(5 + 2sqrt(Giac.giac_eval("6")))),
            () -> Sym.simplify(sqrt(5 + 2sqrt(Sym.Num(6)))),
            () -> SJ.simplify(sqrt(5 + 2sqrt(SJ.Sym(6)))),
            () -> S.sqrtdenest(sqrt(5 + 2sqrt(S.Integer(6))))),
        "√(x²) = x for x > 0" => (:at => 0.7,
            () -> assuming(() -> sqrt(x^2), x > 0),
            () -> Sym.simplify(sqrt(sy.x^2)),
            () -> let x = sj.x; SJ.simplify(sqrt(x^2); assume = SJ.@assume(x > 0)); end,
            () -> S.simplify(sqrt(xpos^2))),
        "log(eˣ) = x for real x" => (:at => 0.7,
            () -> let; @giac_var xr::real; C.simplify(log(exp(xr))); end,
            () -> Sym.simplify(log(exp(sy.x))),
            () -> let x = sj.x; SJ.simplify(log(exp(x)); assume = SJ.@assume(x in SJ.Reals)); end,
            () -> S.simplify(log(exp(xreal)))),
        "sin(π n) = 0 for integer n" => (:at => 0,
            () -> let; @giac_var n::integer; sin(n * Giac.giac_eval("pi")); end,
            () -> Sym.simplify(sin(Sym.variable(:n) * π)),
            () -> let n = SJ.Sym(:n); SJ.simplify(sin(n * SJ.Sym(π)); assume = SJ.@assume(n in SJ.Integers)); end,
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

The step a numerical solver needs from a CAS: the Jacobian of a large sparse model, compiled to a function. The model is the Brusselator on a ring of $N$ cells, $2N$ states. Each system takes its natural route: Giac's SciML extension (the exact sparse Jacobian, compiled), Symbolics' `sparsejacobian` and `build_function`, SymJ's `diff` and `lambdify` (it has no sparse Jacobian, so the nonzero entries of the dense one are compiled), SymPy's `jacobian` and `lambdify`, Mathematica's `D` and `Compile`.

Each system first builds a small model, so no time below includes compiling the builders themselves. Then each cell gives the time of the symbolic build and code generation, and below it the first call, which compiles the generated code (in Julia), and the best of 20 evaluations.

Giac.jl and Symbolics compile to the same kind of code, writing the sparse matrix's stored values directly, and evaluate it in the same time, a few hundred nanoseconds for 400 states. Both differentiate in Julia, and build in about the same time. SymJ builds fastest of the Julia systems, but its compiled function takes each state as an argument of its own: compiling it, at the first call, takes seconds for 100 states and more than the five-minute limit for 400, and a call, which splats the state vector, takes thousands of times longer than the other Julia systems'. Mathematica builds fastest, but its compiled function returns a dense matrix and evaluates a thousand times slower. SymPy is slowest at both.
"""

#%% code id=scale
using SciMLBase: ODEFunction
using SparseArrays: sparse
# the Brusselator's right-hand sides in `u`, `v`, with A = 1, B = 3, α = 0.02, in any system's
# expressions, and a state to evaluate at; as code, which SymJ's Julia process of its own runs too
MODEL = raw"""
function brusselator(u, v)
    N = length(u)
    lap(w, i) = w[mod1(i - 1, N)] - 2w[i] + w[mod1(i + 1, N)]
    vcat([1 + u[i]^2 * v[i] - 4u[i] + lap(u, i) / 50 for i in 1:N], [3u[i] - u[i]^2 * v[i] + lap(v, i) / 50 for i in 1:N])
end
state(N) = vcat(1 .+ sin.(2π .* (1:N) ./ N), fill(3.0, N))
"""
include_string(@__MODULE__, MODEL)

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
# SymJ has no sparse Jacobian: the dense one, by `diff`, and its nonzero entries compiled by
# `lambdify`, in the order a sparse matrix stores them, written into its stored values. In a Julia
# process of its own, with a time limit: compiling the function, on its first call, can take
# minutes, and holds up every other task meanwhile.
function symj_jacobian(N; limit = 300)
    script = """
    import SymJ as SJ
    using SparseArrays: sparse
    $MODEL
    function build(N)
        u, v = [SJ.Sym(Symbol("u\$i")) for i in 1:N], [SJ.Sym(Symbol("v\$i")) for i in 1:N]
        w = vcat(u, v)
        J = [SJ.diff(F, q) for F in brusselator(u, v), q in w]
        nz = findall(!iszero, J)
        (SJ.lambdify(J[nz], w), sparse(getindex.(nz, 1), getindex.(nz, 2), zeros(length(nz)), 2N, 2N))
    end
    function timed(N)
        x0 = state(N)
        tb = @elapsed (f, J) = build(N)
        tc = @elapsed copyto!(J.nzval, f(x0))
        te = minimum(@elapsed(copyto!(J.nzval, f(x0))) for _ in 1:20)
        (tb, tc, te)
    end
    timed(5)                  # a small model first
    foreach(println, timed($N))
    """
    out = IOBuffer()
    p = run(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`; stdout = out, stderr = devnull); wait = false)
    timedwait(() -> process_exited(p), limit; pollint = 0.1) === :ok || (kill(p); return Trial("no answer within $limit s", nothing, :error))
    success(p) || return Trial("error", nothing, :error)
    tb, tc, te = parse.(Float64, split(strip(String(take!(out))), '\n'))
    Trial("first call: " * _timefmt(tc) * ", evaluate: " * _timefmt(te), tb, :ok)
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
scale = Comparison(["Brusselator, N = $N ($(2N) states)" => [giac_jacobian(N), symbolics_jacobian(N), symj_jacobian(N), sympy_jacobian(N),
                                                                native_trial("scale", "Brusselator, N = $N ($(2N) states)"), mathematica_jacobian(N)]
                    for N in (50, 200)])

#%% md id=performance_md
@md"""
## Performance, in summary

For each section and system: how many tasks it answered correctly, how often its answer was the fastest, and its typical slowdown, the geometric mean over the tasks it answered of its time divided by the fastest correct answer's. A system scores ×1.0 only when it is the fastest on every task of a row; where the fastest answers are split between systems, as in simplification, every slowdown is above 1. The lowest in each row is outlined. An approximate, unsimplified, unevaluated, wrong or missing answer doesn't count, so a system that answers fewer tasks is compared only on those. The last two rows are the model-to-Jacobian test: building the function, and evaluating it. The plot shows every time.
"""

#%% code id=performance
"Per system: tasks answered correctly, tasks in all, fastest answers, and the geometric mean slowdown."
struct Performance
    rows::Vector{Pair{String, Vector{NTuple{4, Any}}}}
end

# a time written by `_timefmt`, in seconds
function _seconds(s)
    m = match(r"([\d.]+) (ns|μs|ms|s)\b", s)
    m === nothing && return nothing
    parse(Float64, m[1]) * Dict("ns" => 1e-9, "μs" => 1e-6, "ms" => 1e-3, "s" => 1.0)[m[2]]
end
# the scale test's evaluation times, as trials of their own
_evaluation(c::Comparison) = Comparison([label => [Trial(t.answer, t.status === :ok ? _seconds(split(t.answer, "evaluate: ")[end]) : nothing, t.status)
                                                   for t in trials] for (label, trials) in c.rows])

function _performance(rows)
    k = length(COLUMNS)
    answered, wins, logs = zeros(Int, k), zeros(Int, k), [Float64[] for _ in 1:k]
    for (_, trials) in rows
        ok = [t.status === :ok && t.seconds !== nothing for t in trials]
        any(ok) || continue
        best = minimum(t.seconds for (t, o) in zip(trials, ok) if o)
        for (i, (t, o)) in enumerate(zip(trials, ok))
            o || continue
            answered[i] += 1
            t.seconds == best && (wins[i] += 1)
            push!(logs[i], log(t.seconds / best))
        end
    end
    [(answered[i], length(rows), wins[i], isempty(logs[i]) ? nothing : exp(sum(logs[i]) / length(logs[i]))) for i in 1:k]
end

sections = ["Polynomial algebra" => algebra, "Calculus" => calculus, "Equations" => equations,
            "ODEs in closed form" => odes, "Simplification" => simplification,
            "Model to Jacobian: build" => scale, "Model to Jacobian: evaluation" => _evaluation(scale)]
performance = Performance(vcat([name => _performance(c.rows) for (name, c) in sections],
                               ["All tasks" => _performance(reduce(vcat, [c.rows for (_, c) in sections]))]))

# green for the fastest, through amber, to red at a hundred times slower
function _slowdown_colour(r)
    f = clamp(log10(r) / 2, 0, 1)
    c1, c2, c3 = (126, 198, 153), (229, 192, 123), (224, 108, 117)
    a, b, g = f < 0.5 ? (c1, c2, 2f) : (c2, c3, 2f - 1)
    "rgba($(round(Int, a[1] + (b[1] - a[1]) * g)), $(round(Int, a[2] + (b[2] - a[2]) * g)), $(round(Int, a[3] + (b[3] - a[3]) * g)), 0.28)"
end

function Base.show(io::IO, ::MIME"text/html", p::Performance)
    print(io, """<style>
      table.perf { border-collapse: collapse; margin: 6px 0; font-size: .92em; }
      table.perf th, table.perf td { padding: 7px 12px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; }
      table.perf td.s { font-weight: 600; font-variant-numeric: tabular-nums; }
      table.perf td.best { outline: 2px solid #7ec699; outline-offset: -3px; }
      table.perf .d { opacity: .7; font-size: .85em; display: block; }
      table.perf tr.all td { border-top: 2px solid rgba(128,128,128,.5); font-weight: 600; }
    </style><table class="perf"><tr><th></th>""")
    foreach(s -> print(io, "<th>", s, "</th>"), COLUMNS)
    print(io, "</tr>")
    for (name, cells) in p.rows
        print(io, name == "All tasks" ? "<tr class=\"all\">" : "<tr>", "<td>", name, "</td>")
        lowest = minimum((r for (_, _, _, r) in cells if r !== nothing); init = Inf)
        for (a, n, w, r) in cells
            if r === nothing
                print(io, "<td class=\"s\" style=\"opacity:.5\">—<span class=\"d\">$a of $n answered</span></td>")
            else
                print(io, "<td class=\"s", r == lowest ? " best" : "", "\" style=\"background:", _slowdown_colour(r), "\">×",
                      r < 10 ? round(r; digits = 1) : round(Int, r), "<span class=\"d\">$a of $n answered, $w fastest</span></td>")
            end
        end
        print(io, "</tr>")
    end
    print(io, "</table>")
end

function Base.show(io::IO, ::MIME"text/markdown", p::Performance)
    println(io, "| | ", join(COLUMNS, " | "), " |\n|", repeat(":--|", length(COLUMNS) + 1))
    for (name, cells) in p.rows
        lowest = minimum((r for (_, _, _, r) in cells if r !== nothing); init = Inf)
        println(io, "| ", name, " | ", join([r === nothing ? "— ($a/$n)" :
                                             (r == lowest ? "**×$(round(r; digits = 1))**" : "×$(round(r; digits = 1))") * " ($a/$n, $w fastest)"
                                             for (a, n, w, r) in cells], " | "), " |")
    end
end
performance

#%% code id=performance_plot
using CairoMakie
set_theme!(theme_dark())
let rows = reduce(vcat, [[(name, label, trials) for (label, trials) in c.rows] for (name, c) in sections])
    colours = [:orange, :mediumpurple, :yellowgreen, :deepskyblue, :lightskyblue, :tomato]
    markers = [:circle, :diamond, :star5, :utriangle, :dtriangle, :rect]
    n = length(rows)
    fig = Figure(size = (1000, 26n + 120))
    # the model-to-Jacobian rows twice, as building and as evaluating
    labels = [startswith(name, "Model to Jacobian: ") ? split(name, ": ")[2] * ", " * replace(l, "Brusselator, " => "") : first(l, 46)
              for (name, l, _) in rows]
    ax = Axis(fig[2, 1]; xscale = log10, xlabel = "seconds", yticks = (1:n, labels),
              yreversed = true, title = "every task's time; hollow: not a correct answer", subtitle = machine_summary,
              yticklabelsize = 11)
    for (i, (col, mk)) in enumerate(zip(colours, markers))
        good = [(t[3][i].seconds, j) for (j, t) in enumerate(rows) if t[3][i].seconds !== nothing && t[3][i].status === :ok]
        poor = [(t[3][i].seconds, j) for (j, t) in enumerate(rows) if t[3][i].seconds !== nothing && t[3][i].status !== :ok]
        isempty(good) || scatter!(ax, first.(good), last.(good); color = col, marker = mk, markersize = 11, label = short_versions[COLUMNS[i]])
        isempty(poor) || scatter!(ax, first.(poor), last.(poor); color = :transparent, strokecolor = col, strokewidth = 1.5, marker = mk, markersize = 11)
    end
    # the sections, divided
    edges = cumsum([length(c.rows) for (_, c) in sections])[1:end-1] .+ 0.5
    hlines!(ax, edges; color = :gray40, linewidth = 1)
    Legend(fig[1, 1], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
    fig
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   BenchmarkTools 1.8.0 6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf
#   CairoMakie 0.15.17 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   DedicatedThreads 0.1.0 9d6577c3-9664-4440-a53f-cec765ebc045
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   Groebner 0.10.11 0b43b601-686d-58a3-8a1c-6623616c7cd4
#   Latexify 0.16.12 23fbe1c1-3f47-55db-b15f-69d7ec21a316
#   RischIntegrate 1.2.0 c180fd1b-065d-41df-ac4d-d8d9562c1957
#   SciMLBase 3.58.0 0bca4576-84f4-4d90-8ffe-ffa030f20462
#   SparseArrays 1.13.0 2f01184e-e22b-5df5-ae63-d93ebab69eaf
#   SymJ 0.1.0 73a6d648-42a5-483e-b8b5-fc71c1696ef5 path=~/devel/SymJ.jl
#   SymPyPythonCall 0.5.3 bc8888f7-b21e-4b7c-a06a-5d9c9496438c
#   Symbolics 7.48.0 0c5d862f-8b57-4792-8d23-62f2024744c7
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = e00c1f95-b52d-42d0-9994-884253e0d6bf
#   format = 2
# ╚═╡
