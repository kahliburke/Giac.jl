try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Giac.jl against Mathematica

A contest of capability: ninety problems in twelve areas, from the textbook-hard to the famously hard, each posed to Giac.jl and to Mathematica the way its own user would pose it. Speed is not the point here; whether a system can answer at all is.

Each system gets a minute per problem. Its answer is then checked in that system itself, numerically and with exact inputs, against what the problem asks:
- a value is compared with the known one, to twelve digits;
- an antiderivative is differentiated and compared with the integrand at several points;
- solutions are substituted into the equations, and counted;
- a general solution of a differential equation is substituted into it with its arbitrary constants given values, and they are counted;
- a set given by inequalities is tested at points inside and outside it;
- some problems also ask for a form: radicals denested, no trigonometric functions left, a factorization into a given number of factors.

An answer is **solved** when it passes and is exact, **partial** when it passes but is numeric, incomplete or not in the form asked for, **not solved** when the system gives the problem back, gives up or runs out of time, and **wrong** when the check fails.

Giac.jl is used as a whole: the GIAC engine with this branch's own layers and extensions loaded (Nemo, msolve, sum-of-squares). It runs on a development build of its C++ wrapper; the setup output records which.
"""

#%% code id=setup
import DedicatedThreads, Nemo, AlgebraicSolving, JuMP, Clarabel, JSON, libgiac_julia_jll
using Giac, GiacSlate, LinearAlgebra
const C = Giac.Commands
# Giac's C++ wrapper: a local build, as its checkout and commit, or the released one
giac_wrapper = let dir = dirname(dirname(dirname(libgiac_julia_jll.libgiac_wrapper_path)))
    isdir(joinpath(dir, ".git")) ? replace(dir, homedir() => "~") * " at " * readchomp(`git -C $dir rev-parse --short HEAD`) :
                                   "released " * Giac.GiacCxxBindings.wrapper_version()
end
# `wolframscript` with `args`, its output as text, one kernel at a time
WOLFRAM = ReentrantLock()
wolfram(args...) = lock(() -> strip(read(`wolframscript $args`, String)), WOLFRAM)
(; giac = pkgdir(Giac), giac_wrapper, giac_engine = Giac.GiacCxxBindings.giac_version(),
   mathematica = wolfram("-code", "\$VersionNumber"))

#%% code id=harness collapsed eval_barrier
"A problem: its statement, each system's way of asking it, and how an answer is checked."
struct Problem
    title::String
    tex::String          # the problem, in LaTeX
    giac::Function       # () -> Giac.jl's answer
    wolfram::String      # Wolfram Language code for Mathematica's answer
    check::Any           # one of the checks below
end

"How a system did on a problem."
mutable struct Attempt
    answer::String
    latex::String
    seconds::Union{Nothing, Float64}
    outcome::Symbol      # :solved, :partial, :unsolved, :wrong
    note::String
end

const LIMIT = 60         # seconds per problem and system
const WARM = 10          # a Giac.jl answer within this many seconds is computed again and timed

# ---------------------------------------------------------------------------------------------
# Giac.jl's side: an answer as Julia values
# ---------------------------------------------------------------------------------------------

# a GIAC list as a Julia vector of its entries (nested lists stay GiacExpr), anything else as is
_items(e::GiacExpr) = Giac.giac_type(e) == Giac.GenTypes.VECT ? [e[i] for i in 1:length(e)] : [e]
_items(v::AbstractArray) = vec(permutedims(v))           # row by row, as Mathematica flattens
_items(x) = [x]
_flat(e) = reduce(vcat, [x isa GiacExpr && Giac.giac_type(x) == Giac.GenTypes.VECT ? _flat(x) : [x] for x in _items(e)]; init = Any[])

# `e` with the variables `at` (name => exact value, as text) put in, as a complex number
function _num(e, at = Pair{String, String}[])
    e isa Number && return ComplexF64(e)
    g = convert(GiacExpr, e)
    isempty(at) || (g = C.subst(g, giac_eval("[" * join(first.(at), ",") * "]"), giac_eval("[" * join(last.(at), ",") * "]")))
    v = try to_julia(C.evalf(g, 30)) catch; NaN end
    v isa Number ? ComplexF64(v) : NaN + 0im
end

_tree(e) = e isa GiacExpr ? Giac._node(e) : nothing
function _any_part(pred, e)
    n = _tree(e)
    n === nothing ? false : Giac._any_node(pred, n)
end
_exact(e) = !any(x -> x isa GiacExpr && Giac._has_float(Giac._node(x)), _flat(e)) && !(e isa AbstractFloat)
_names(e) = e isa GiacExpr ? Giac._free_names(e) : Set{String}()

const _HEADS = Set(["integrate", "sum", "product", "limit", "desolve", "solve", "laplace", "ilaplace", "invlaplace", "series", "simplify", "undef"])
_unevaluated(e) = any(x -> x isa GiacExpr && (string(x) == "undef" ||
                      _any_part(n -> n isa Giac.CallNode && n.f in _HEADS, x)), _flat(e))

# the nesting depth of roots, and whether trigonometric functions are left
function _radical_depth(n)
    n isa Giac.MulNode && return maximum(((b, p),) -> (p isa Giac.NumNode && p.v isa Rational && !isinteger(p.v) ? 1 : 0) + _radical_depth(b),
                                         n.factors; init = 0)
    n isa Giac.AddNode && return maximum(t -> _radical_depth(first(t)), n.terms; init = 0)
    n isa Giac.CallNode && return maximum(_radical_depth, n.args; init = 0)
    return 0
end
_radical_depth(e::GiacExpr) = _radical_depth(Giac._node(e))
const _TRIG = Set(["sin", "cos", "tan", "acos", "asin", "atan", "sec", "csc", "cot"])
_has_trig(e) = any(x -> _any_part(n -> n isa Giac.CallNode && n.f in _TRIG, x), _flat(e))

# ---------------------------------------------------------------------------------------------
# Checks. Each gives Giac.jl's data from its answer, Mathematica's as a Wolfram Language function
# of its answer (returning an association), and judges the data, the same for both.
# ---------------------------------------------------------------------------------------------

_finite(v) = all(z -> isfinite(z), v)
const _NOT_A_NUMBER = (:unsolved, "no numeric value")

_wl_rules(at) = "{" * join(["$k -> ($v)" for (k, v) in at], ", ") * "}"
_judged(ok, exact, note = "numeric") = ok ? (exact ? (:solved, "") : (:partial, note)) : (:wrong, "check fails")

"A value, or values at several points; `part` picks it out of the answer."
Base.@kwdef struct Value
    ref::Vector{ComplexF64}
    at::Vector{Vector{Pair{String, String}}} = [Pair{String, String}[]]
    giac::Function = identity
    wolfram::String = "Identity"
    sorted::Bool = false
end
Value(ref; kw...) = Value(; ref = ComplexF64.(vec(permutedims(ref isa AbstractArray ? ref : [ref]))), kw...)
function giac_data(c::Value, ans)
    a = c.giac(ans)
    vals = reduce(vcat, [[_num(x, at) for x in _flat(a)] for at in c.at])
    Dict("value" => vals, "exact" => _exact(a))
end
wolfram_check(c::Value) = "Function[ans, With[{part\$ = $(c.wolfram)[ans]}, <|\"value\" -> Join @@ Table[num[Flatten[{part\$}] /. at\$], {at\$, {" *
    join(_wl_rules.(c.at), ", ") * "}}], \"exact\" -> exactQ[part\$]|>]]"
function judge(c::Value, d)
    v = d["value"]
    c.sorted && (v = sort(v; by = z -> (round(real(z); digits = 8), imag(z))))
    ref = c.sorted ? sort(c.ref; by = z -> (round(real(z); digits = 8), imag(z))) : c.ref
    length(v) == length(ref) || return (:wrong, "$(length(v)) values for $(length(ref))")
    _finite(v) || return _NOT_A_NUMBER
    _judged(all(abs.(v .- ref) .<= 1e-9 .* (1 .+ abs.(ref))), d["exact"])
end

"An antiderivative of `f` in `x`: its derivative is compared with `f` at a few points."
struct Antiderivative
    f::GiacExpr
    fwl::String
    x::String
    points::Vector{String}
end
Antiderivative(f, fwl, x = "x"; points = ["3/10", "7/10", "11/10"]) = Antiderivative(f, fwl, x, points)
giac_data(c::Antiderivative, ans) =
    Dict("residual" => [_num(C.diff(ans, giac_eval(c.x)) - c.f, [c.x => p]) for p in c.points], "exact" => _exact(ans))
wolfram_check(c::Antiderivative) = "Function[ans, <|\"residual\" -> num[Table[(D[ans, $(c.x)] - ($(c.fwl))) /. $(c.x) -> p, {p, {" *
    join(c.points, ", ") * "}}]], \"exact\" -> exactQ[ans]|>]"
judge(c::Antiderivative, d) = _finite(d["residual"]) ? _judged(all(r -> abs(r) < 1e-8, d["residual"]), d["exact"]) : _NOT_A_NUMBER

"Solutions of equations (expressions equal to zero) in `vars`, all `count` of them."
struct Solutions
    eqs::Vector{GiacExpr}
    eqswl::String
    vars::Vector{String}
    count::Int
end
function giac_data(c::Solutions, ans)
    sols = [length(c.vars) == 1 ? [s] : _items(s) for s in _items(ans)]
    pts = [[_num(v) for v in s] for s in sols if length(s) == length(c.vars)]
    res = [maximum(abs, [_num(C.subst(e, giac_eval("[" * join(c.vars, ",") * "]"), giac_eval("[" * join(string.(s), ",") * "]"))) for e in c.eqs])
           for s in sols if length(s) == length(c.vars)]
    Dict("points" => pts, "residual" => res, "exact" => _exact(ans), "shape" => all(s -> length(s) == length(c.vars), sols))
end
wolfram_check(c::Solutions) = "Function[ans, With[{sols\$ = Normal[ans]}, <|\"points\" -> Table[num[{" * join(c.vars, ", ") *
    "} /. r\$], {r\$, sols\$}], \"residual\" -> num[Table[Max[Abs[N[$(c.eqswl) /. r\$, 30]]], {r\$, sols\$}]], \"exact\" -> exactQ[sols\$], \"shape\" -> MatchQ[sols\$, {{__Rule}...}]|>]]"
function judge(c::Solutions, d)
    d["shape"] || return (:wrong, "not a list of solutions")
    _finite(d["residual"]) || return _NOT_A_NUMBER
    all(r -> abs(r) < 1e-8, d["residual"]) || return (:wrong, "a solution fails")
    n = length(unique(p -> round.(p; digits = 7), d["points"]))
    n < c.count && return (:partial, "$n of $(c.count) solutions")
    _judged(true, d["exact"])
end

"""
All solutions of `eq = 0` in `x`, families with integer parameters included: those in `interval`
are found by giving the parameters the values -8…8, and must be all `count` of them.
"""
struct Periodic
    eq::GiacExpr
    eqwl::String
    x::String
    interval::Tuple{Float64, Float64}
    count::Int
end
function giac_data(c::Periodic, ans)
    vals = ComplexF64[]
    for s in _items(ans)
        ps = setdiff(_names(s), [c.x])
        for k in -8:8
            push!(vals, _num(s, [p => string(k) for p in ps]))
        end
    end
    Dict("values" => vals, "residual" => [_num(c.eq, [c.x => string(real(v))]) for v in vals if isfinite(v)], "exact" => _exact(ans))
end
wolfram_check(c::Periodic) = "Function[ans, With[{vals\$ = Flatten[Table[($(c.x) /. Normal[r\$]) /. C[_] -> k\$, {r\$, ans}, {k\$, -8, 8}]]}, " *
    "<|\"values\" -> num[vals\$], \"residual\" -> num[Map[Function[p\$, ($(c.eqwl)) /. $(c.x) -> p\$], vals\$]], \"exact\" -> exactQ[ans]|>]]"
function judge(c::Periodic, d)
    inside = [v for v in d["values"] if abs(imag(v)) < 1e-9 && c.interval[1] < real(v) < c.interval[2]]
    all(r -> abs(r) < 1e-8, d["residual"][1:min(end, 200)]) || return (:wrong, "a solution fails")
    n = length(unique(v -> round(real(v); digits = 7), inside))
    n < c.count && return (:partial, "$n of the $(c.count) solutions in $(c.interval)")
    _judged(true, d["exact"])
end

"A set given by conditions on `x`, tested at sample points against `truth`."
struct Region
    x::String
    truth::Function
    samples::Vector{String}
end
function giac_data(c::Region, ans)
    conds = _items(ans)
    member(p) = any(cd -> string(giac_eval("evalb(subst($(string(cd)),$(c.x)=$p))")) in ("true", "1"), conds)
    Dict("member" => [member(p) for p in c.samples])
end
wolfram_check(c::Region) = "Function[ans, <|\"member\" -> Table[TrueQ[ans /. $(c.x) -> p], {p, {" * join(c.samples, ", ") * "}}]|>]"
function judge(c::Region, d)
    truth = [c.truth(Float64(eval(Meta.parse(replace(p, "/" => "//"))))) for p in c.samples]
    d["member"] == truth ? (:solved, "") : (:wrong, "$(count(d["member"] .!= truth)) of $(length(truth)) points misplaced")
end

"A statement to prove: the answer must be true."
struct Truth end
giac_data(::Truth, ans) = Dict("truth" => ans === true || (ans isa GiacExpr && string(ans) in ("true", "1")))
wolfram_check(::Truth) = "Function[ans, <|\"truth\" -> TrueQ[ans]|>]"
judge(::Truth, d) = d["truth"] ? (:solved, "") : (:unsolved, "not proved")

"""
A general solution of a differential equation: `residual(Y)` (and its Wolfram Language twin, a
function of the solution expression) vanishes at `points` with the arbitrary constants given
values, and there are `order` constants.
"""
struct DE
    residual::Function
    residualwl::String
    vars::Vector{String}
    points::Vector{Vector{String}}
    order::Int
    unknown::String      # Mathematica's unknown, `y[x]`
end
DE(r, rwl, vars, points, order; unknown = "y[x]") = DE(r, rwl, vars, points, order, unknown)
const _CONSTANT_VALUES = ["7/10", "-13/10", "2/5", "3/2"]
function giac_data(c::DE, ans)
    sol = first(_items(ans))
    consts = sort!(collect(setdiff(_names(sol), c.vars)))
    vals = Pair{String, String}[k => _CONSTANT_VALUES[mod1(i, end)] for (i, k) in enumerate(consts)]
    r = c.residual(sol)
    Dict("residual" => [_num(e, vcat(vals, [v => p for (v, p) in zip(c.vars, pt)])) for e in _items(r), pt in c.points][:],
         "constants" => length(consts), "exact" => _exact(sol))
end
wolfram_check(c::DE) = "Function[ans, With[{sol\$ = (($(c.unknown) /. First[ans]) /. (C[i\$_][arg\$_] :> Sin[arg\$] + i\$ arg\$^2)) /. Table[C[i\$] -> {" * join(_CONSTANT_VALUES, ", ") *
    "}[[Mod[i\$ - 1, 4] + 1]], {i\$, 8}]}, <|\"residual\" -> num[Flatten[Table[Flatten[{$(c.residualwl)[sol\$]}] /. Thread[{" * join(c.vars, ", ") *
    "} -> pt\$], {pt\$, {" * join(["{" * join(p, ", ") * "}" for p in c.points], ", ") * "}}]]], \"constants\" -> Length[Union[Cases[ans, C[_], Infinity, Heads -> True]]], \"exact\" -> exactQ[ans]|>]]"
function judge(c::DE, d)
    _finite(d["residual"]) || return _NOT_A_NUMBER
    all(r -> abs(r) < 1e-8, d["residual"]) || return (:wrong, "does not satisfy the equation")
    d["constants"] < c.order && return (:partial, "$(d["constants"]) of $(c.order) constants")
    _judged(true, d["exact"])
end

"Integers taken from the answer, which `pred` must accept."
struct Integers
    pred::Function
    giac::Function       # answer -> integers
    wolfram::String      # answer -> integers, Wolfram Language
end
giac_data(c::Integers, ans) = Dict("integers" => string.(c.giac(ans)))
wolfram_check(c::Integers) = "Function[ans, <|\"integers\" -> ToString /@ Flatten[{$(c.wolfram)[ans]}]|>]"
function judge(c::Integers, d)
    n = try parse.(BigInt, d["integers"]) catch; nothing end
    n !== nothing && !isempty(n) && c.pred(n) ? (:solved, "") : (:wrong, "check fails")
end

"""
A check, and a form the answer must have: `:denested` (roots not nested), `:no_trig`
(no trigonometric functions left), `(:factors, n)` (a product of `n` distinct non-constant factors),
or `(:free_of, name)`.
"""
struct Form
    check::Any
    form::Any
    note::String
end
function giac_data(c::Form, ans)
    d = giac_data(c.check, ans)
    d["form"] = if c.form === :denested
        all(x -> !(x isa GiacExpr) || _radical_depth(x) <= 1, _flat(ans))
    elseif c.form === :no_trig
        !_has_trig(ans)
    elseif c.form isa Tuple && c.form[1] === :factors
        n = _tree(ans)
        n isa Giac.MulNode && count(((b, e),) -> !isempty(Giac._symbol_names(b)), n.factors) == c.form[2]
    else
        # in each entry of a list too
        all(x -> !(c.form[2] in _names(x)) && !_any_part(n -> n isa Giac.CallNode && n.f == c.form[2], x), _flat(ans))
    end
    d
end
function wolfram_check(c::Form)
    test = c.form === :denested ? "radicalDepth[ans] <= 1" :
           c.form === :no_trig ? "FreeQ[ans, Sin | Cos | Tan | Sec | Csc | Cot | ArcSin | ArcCos | ArcTan]" :
           c.form isa Tuple && c.form[1] === :factors ? "Head[ans] === Times && Count[List @@ ans, f_ /; !FreeQ[f, _Symbol?(! NumericQ[#] &)]] == $(c.form[2])" :
           "FreeQ[ans, $(c.form[2])]"
    "Function[ans, Join[$(wolfram_check(c.check))[ans], <|\"form\" -> TrueQ[$test]|>]]"
end
function judge(c::Form, d)
    o = judge(c.check, d)
    o[1] === :solved && !d["form"] ? (:partial, c.note) : o
end

"An implicit equation: zero at the points `on`, not at `off`."
struct Vanishing
    on::Vector{Vector{Pair{String, String}}}
    off::Vector{Pair{String, String}}
end
giac_data(c::Vanishing, ans) = Dict("on" => [_num(ans, p) for p in c.on], "off" => [_num(ans, c.off)])
_wl_at(e, ps) = "num[Table[$e /. p, {p, {" * join(_wl_rules.(ps), ", ") * "}}]]"
wolfram_check(c::Vanishing) = "Function[ans, <|\"on\" -> " * _wl_at("ans", c.on) * ", \"off\" -> " * _wl_at("ans", [c.off]) * "|>]"
function judge(c::Vanishing, d)
    _finite(vcat(d["on"], d["off"])) || return _NOT_A_NUMBER
    all(v -> abs(v) < 1e-9, d["on"]) && abs(only(d["off"])) > 1e-6 ? (:solved, "") : (:wrong, "check fails")
end

"A decomposition `f = g(h(…))` into polynomials of degree two or more, outermost first."
struct Decomposition
    x::String
    at::String
    ref::ComplexF64
end
function giac_data(c::Decomposition, ans)
    parts = _items(ans)
    v = foldr((part, inner) -> _num(part, [c.x => string(real(inner))]), parts; init = Giac.giac_eval(c.at) |> _num)
    Dict("value" => [v], "parts" => length(parts))
end
wolfram_check(c::Decomposition) = "Function[ans, <|\"value\" -> num[Fold[#2 /. $(c.x) -> #1 &, $(c.at), Reverse[ans]]], \"parts\" -> Length[ans]|>]"
function judge(c::Decomposition, d)
    _finite(d["value"]) || return _NOT_A_NUMBER
    d["parts"] >= 2 && abs(only(d["value"]) - c.ref) < 1e-9 * (1 + abs(c.ref)) ? (:solved, "") : (:wrong, "check fails")
end

# ---------------------------------------------------------------------------------------------
# Running a problem in each system
# ---------------------------------------------------------------------------------------------

_giac_tex(a) = try
    a isa GiacExpr ? String(strip(sprint(show, MIME"text/latex"(), a), ['$', '\n', ' '])) :
    a isa AbstractVector ? "\\left[" * join(_giac_tex.(a), ",\\ ") * "\\right]" :
    a isa AbstractMatrix ? "\\begin{bmatrix}" * join([join(_giac_tex.(r), " & ") for r in eachrow(a)], " \\\\ ") * "\\end{bmatrix}" :
    string(a)
catch
    ""
end
_shorten(s, n = 400) = length(s) <= n ? s : first(s, n) * " …"

function giac_attempt(p::Problem)
    # an earlier failure inside PARI leaves it locked. `pari_unlock` unlocks a locked PARI, but locks
    # one that is free (it tests with a trylock), returning 0: then a second call unlocks it.
    string(giac_eval("pari_unlock()")) == "0" && giac_eval("pari_unlock()")
    # timed inside the task, so the time is the answer's and not the wait's. A quick answer is
    # computed again and that run timed, once Julia has compiled the code the first ran: neither
    # Giac.jl nor GIAC keeps results, so the second run computes the answer anew.
    task = Threads.@spawn begin
        t0 = time()
        r = try (true, Giac.with_timeout(p.giac, LIMIT)) catch e; (false, e) end
        t = time() - t0
        if r[1] && t < WARM
            t1 = time()
            try
                Giac.with_timeout(p.giac, LIMIT)
                t = time() - t1
            catch
            end
        end
        (r..., t)
    end
    if timedwait(() -> istaskdone(task), LIMIT + 15; pollint = 0.05) !== :ok
        return Attempt("no answer within $LIMIT s", "", nothing, :unsolved, "time limit")
    end
    ok, ans, seconds = fetch(task)
    ok || return Attempt(_shorten(ans isa Giac.GiacTimeout ? "no answer within $LIMIT s" : sprint(showerror, ans)), "", seconds, :unsolved,
                         ans isa Giac.GiacTimeout ? "time limit" : "error")
    a = Attempt(_shorten(string(ans)), _giac_tex(ans), seconds, :unsolved, "")
    if _unevaluated(ans)
        a.note = "given back"
        return a
    end
    if ans isa GiacExpr && Giac.giac_type(ans) == Giac.GenTypes.VECT && length(ans) == 0
        a.note = "nothing found"
        return a
    end
    gave_up = [x for x in _flat(ans) if x isa GiacExpr && Giac.giac_type(x) == Giac.GenTypes.STRNG]
    if !isempty(gave_up)
        a.note = "gave up: " * _shorten(strip(string(first(gave_up)), '"'), 60)
        return a
    end
    a.outcome, a.note = try
        judge(p.check, giac_data(p.check, ans))
    catch e
        (:wrong, "cannot be checked: " * _shorten(sprint(showerror, e), 80))
    end
    return a
end

# The Wolfram Language script for a list of problems: each answer, timed in a fresh cache under
# the time limit, typeset, and checked, as one line of JSON
const WL_PRELUDE = raw"""
$HistoryLength = 0;
num[e_] := Map[If[NumericQ[#], {Re[#], Im[#]} &[N[N[#, 30]]], "nan"] &, Flatten[{e}]];
exactQ[e_] := FreeQ[e, _Real];
radicalDepth[e_] := Max[0, Cases[e, Power[b_, r_Rational] :> 1 + radicalDepth[b], {0, Infinity}]];
tex[r_] := StringReplace[ToString[TeXForm[r /. Root[p_, k_, ___] :> Subscript["root", k][p[z]]]], "\n" -> " "];
unevaluatedQ[e_] := !FreeQ[e, Integrate | Sum | Product | Limit | DSolve | Solve | Reduce | LaplaceTransform | InverseLaplaceTransform | DifferentialRoot | Inactive | $Failed | Indeterminate];
attempt[code_Hold, check_] := Module[{t, ans, out},
  ClearSystemCache[];
  {t, ans} = AbsoluteTiming[TimeConstrained[Quiet[ReleaseHold[code]], LIMIT, $TimedOut]];
  out = <|"time" -> t, "answer" -> StringTake[ToString[ans, InputForm], UpTo[400]], "tex" -> Quiet[Check[tex[ans], ""]]|>;
  Which[ans === $TimedOut, out["outcome"] = "timeout",
        unevaluatedQ[ans], out["outcome"] = "unevaluated",
        True, out["check"] = Quiet[TimeConstrained[check[ans], LIMIT, "timeout"]]];
  json = Quiet[ExportString[out, "RawJSON", "Compact" -> True]];
  If[!StringQ[json], out["check"] = "failed"; json = ExportString[out, "RawJSON", "Compact" -> True]];
  Print["@@", json]];
"""

# Mathematica is warmed up on unrelated problems of the kinds in the contest before any is timed,
# as it loads the code for a kind of problem on first use; each contest problem is then timed
# once, as a repeat is answered from a cache (Sum and Product remember their results through
# ClearSystemCache)
const WL_WARM_UP = raw"""
Quiet[{Integrate[x^2 Sin[x], x], Integrate[Exp[-x] x^3, {x, 0, Infinity}], Integrate[Log[1 + x]/x, {x, 0, 1}],
  Sum[1/k^3, {k, 1, Infinity}], Sum[(-1)^k/k^2, {k, 1, Infinity}], Sum[Binomial[m, j], {j, 0, m}], Product[1 + 1/k^3, {k, 1, Infinity}],
  Limit[(Cos[x] - 1)/x^2, x -> 0], Limit[(1 + 2/x)^x, x -> Infinity], Solve[{u^2 + w^2 == 2, u == w}, {u, w}],
  Solve[u^4 - 3 u + 1 == 0, u], Reduce[u^2 - 3 u + 2 > 0, u, Reals], Factor[u^12 - 1], Factor[u^4 + 4],
  Simplify[Sin[u]^2 + Cos[u]^2], FullSimplify[Sqrt[3 + 2 Sqrt[2]]], MinimalPolynomial[Sqrt[2] + Sqrt[7], u],
  DSolve[w''[u] + w[u] == u, w[u], u], DSolve[w'[u] == w[u]^2, w[u], u], Eigenvalues[{{1, 2}, {3, 4}}], MatrixExp[{{0, 2}, {-2, 0}} s],
  FactorInteger[2^64 + 1], PrimeQ[2^127 - 1], LaplaceTransform[s^3 Exp[-2 s], s, p], InverseLaplaceTransform[1/(p^2 + 4), p, s],
  Series[Sin[u], {u, 0, 6}], Resolve[ForAll[u, u^2 + 1 > 0], Reals]}];
"""

function _wolfram_complex(x)
    x isa AbstractVector && length(x) == 2 && all(v -> v isa Real && !(v isa Bool), x) && return ComplexF64(x[1], x[2])
    x isa AbstractVector && return [_wolfram_complex(v) for v in x]
    x == "nan" && return NaN + 0im
    return x
end
_wolfram_data(d::AbstractDict) = Dict{String, Any}(k => (v isa AbstractVector ? _wolfram_complex(v) : v) for (k, v) in d)

function wolfram_attempts(problems)
    script = WL_PRELUDE * WL_WARM_UP * "LIMIT = $LIMIT;\n" *
             join(["attempt[Hold[$(p.wolfram)], $(wolfram_check(p.check))];" for p in problems], "\n")
    file = tempname() * ".wls"
    write(file, script)
    out = try wolfram("-file", file) catch e; sprint(showerror, e) end
    lines = [JSON.parse(l[3:end]) for l in split(out, '\n') if startswith(l, "@@")]
    map(enumerate(problems)) do (i, p)
        i <= length(lines) || return Attempt("no answer (the kernel stopped)", "", nothing, :unsolved, "error")
        r = lines[i]
        a = Attempt(r["answer"], r["tex"], r["time"], :unsolved, "")
        o = get(r, "outcome", nothing)
        o == "timeout" && (a.answer = "no answer within $LIMIT s"; a.note = "time limit"; return a)
        o == "unevaluated" && (a.note = "given back"; return a)
        chk = r["check"]
        if !(chk isa AbstractDict)
            a.outcome, a.note = :wrong, "cannot be checked"
            return a
        end
        a.outcome, a.note = try judge(p.check, _wolfram_data(chk)) catch e; (:wrong, "cannot be checked") end
        a
    end
end

"A contest in one area: its problems and each system's attempts."
struct Contest
    area::String
    problems::Vector{Problem}
    giac::Vector{Attempt}
    wolfram::Vector{Attempt}
end
# One area at a time, though Slate may run the cells side by side: GIAC, its global settings and
# the Mathematica kernel are shared, and a time limit must not include waiting for them
const RUNNING = ReentrantLock()
contest(area, problems) = lock(RUNNING) do
    Contest(area, problems, [giac_attempt(p) for p in problems], wolfram_attempts(problems))
end

# ---------------------------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------------------------

const _OUTCOME_STYLE = Dict(:solved => ("#7ec699", "solved"), :partial => ("#e5c07b", "partial"),
                            :unsolved => ("#8a8f98", "not solved"), :wrong => ("#e06c75", "wrong"))
_esc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
_seconds(t) = t === nothing ? "" : t < 1e-3 ? "$(round(t * 1e6; digits = 1)) μs" : t < 1 ? "$(round(t * 1e3; digits = 1)) ms" : "$(round(t; digits = 2)) s"
function _badge(a::Attempt)
    colour, word = _OUTCOME_STYLE[a.outcome]
    note = isempty(a.note) ? "" : " · " * _esc(a.note)
    "<div class=\"badge\" style=\"color:$colour\">$word$note<span class=\"t\">$(_seconds(a.seconds))</span></div>"
end
function _cell(a::Attempt)
    body = isempty(a.latex) || a.outcome === :unsolved && a.note in ("time limit", "error") ? "<code>" * _esc(a.answer) * "</code>" :
           "<div class=\"tex\">\$\\displaystyle " * _esc(a.latex) * "\$</div>"
    "<td>" * body * _badge(a) * "</td>"
end

const CONTEST_CSS = """<style>
  table.contest { border-collapse: collapse; margin: 6px 0; font-size: .92em; width: 100%; table-layout: fixed; }
  table.contest th, table.contest td { padding: 7px 10px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; vertical-align: top; }
  table.contest th:first-child { width: 30%; }
  table.contest .tex { max-height: 9em; overflow: auto; }
  table.contest .badge { font-size: .85em; margin-top: 4px; font-weight: 600; }
  table.contest .badge .t { opacity: .6; font-weight: 400; margin-left: .6em; }
  table.contest .title { font-weight: 600; margin-bottom: 4px; }
</style>"""

function Base.show(io::IO, ::MIME"text/html", c::Contest)
    print(io, CONTEST_CSS, "<table class=\"contest\"><tr><th>Problem</th><th>Giac.jl</th><th>Mathematica</th></tr>")
    for (p, g, w) in zip(c.problems, c.giac, c.wolfram)
        print(io, "<tr><td><div class=\"title\">", _esc(p.title), "</div>\$", _esc(p.tex), "\$</td>", _cell(g), _cell(w), "</tr>")
    end
    print(io, "</table>")
end
function Base.show(io::IO, ::MIME"text/markdown", c::Contest)
    println(io, "| Problem | Giac.jl | Mathematica |\n|:--|:--|:--|")
    for (p, g, w) in zip(c.problems, c.giac, c.wolfram)
        cell(a) = "**$(_OUTCOME_STYLE[a.outcome][2])**" * (isempty(a.note) ? "" : " ($(a.note))") * ": `" * replace(_shorten(a.answer, 120), "|" => "\\|") * "`"
        println(io, "| ", p.title, " | ", cell(g), " | ", cell(w), " |")
    end
end

#%% md id=environment_md
@md"""
## Environment

What the results were measured on, read when the notebook runs: each system's exact version and the libraries under it, the machine, and the contest's own settings. Giac.jl's commit is marked *modified* when its sources differ from that commit.
"""

#%% code id=environment
import Pkg, Dates
using Markdown
# `short_versions`: each system and its version, for the plot; `machine_summary`: the machine and
# Julia in a line
short_versions, machine_summary, environment = let
    sh(cmd) = try readchomp(cmd) catch; "" end
    git(dir, args...) = sh(`git -C $dir $args`)
    # the packages of the notebook's environment, by name
    deps = Dict(d.name => d for d in values(Pkg.dependencies()))
    ver(name) = haskey(deps, name) && deps[name].version !== nothing ? string(deps[name].version) : "—"
    giac_dir = pkgdir(Giac)
    giac_commit = git(giac_dir, "rev-parse", "--short", "HEAD") *
                  (isempty(git(giac_dir, "status", "--porcelain", "--", "src", "ext")) ? "" : " (modified)")
    wl = JSON.parse(wolfram("-code", "ExportString[<|\"version\" -> \$Version, \"system\" -> \$SystemID, \"processors\" -> \$ProcessorCount|>, \"RawJSON\", \"Compact\" -> True]"))
    cpu = Sys.cpu_info()
    software = [
        "Giac.jl" => "$(pkgversion(Giac)), $(git(giac_dir, "rev-parse", "--abbrev-ref", "HEAD")) at $giac_commit",
        "GIAC (engine)" => "$(Giac.GiacCxxBindings.giac_version()), GIAC_jll $(ver("GIAC_jll"))",
        "Giac's C++ wrapper" => "$giac_wrapper (libgiac_julia_jll $(ver("libgiac_julia_jll")), CxxWrap $(ver("CxxWrap")))",
        "PARI (through GIAC)" => "PARI_jll $(ver("PARI_jll"))",
        "Nemo / FLINT" => "Nemo $(ver("Nemo")), FLINT_jll $(ver("FLINT_jll"))",
        "msolve" => "AlgebraicSolving $(ver("AlgebraicSolving")), msolve_jll $(ver("msolve_jll"))",
        "Sum of squares" => "JuMP $(ver("JuMP")), Clarabel $(ver("Clarabel"))",
        "Julia" => "$(VERSION), $(Threads.nthreads()) threads, GIAC on its own thread (DedicatedThreads $(ver("DedicatedThreads")))",
        "Mathematica" => "$(wl["version"]) ($(wl["system"])), kernel sees $(wl["processors"]) processors",
    ]
    machine = [
        "Computer" => sh(`sysctl -n hw.model`),
        "CPU" => "$(cpu[1].model), $(sh(`sysctl -n hw.perflevel0.physicalcpu`)) performance and $(sh(`sysctl -n hw.perflevel1.physicalcpu`)) efficiency cores",
        "Memory" => "$(round(Int, Sys.total_memory() / 2^30)) GiB",
        "Operating system" => "macOS $(sh(`sw_vers -productVersion`)) ($(sh(`sw_vers -buildVersion`))), $(Sys.MACHINE)",
        "Power" => something(match(r"'(.*?)'", sh(`pmset -g batt`)), (; captures = ["unknown"])).captures[1],
        # other work on the machine shares its cores: the load average as the contest starts
        "Load when read" => join(round.(Sys.loadavg(); digits = 2), ", ") * " (1, 5, 15 min)",
    ]
    contest_settings = [
        "Time limit" => "$LIMIT s per problem and system",
        "Timing" => "Giac.jl: the second of two runs; Mathematica: one run, after a warm-up on unrelated problems",
        "Run" => string(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")),
    ]
    cell(v) = replace(v, "|" => "\\|", "_" => "\\_", "*" => "\\*")
    table(title, rows) = "| $title | |\n|:--|:--|\n" * join(["| $k | $(cell(v)) |" for (k, v) in rows], "\n")
    short = Dict("Giac.jl" => "Giac.jl $(pkgversion(Giac)) at $giac_commit (GIAC $(Giac.GiacCxxBindings.giac_version()))",
                 "Mathematica" => "Mathematica $(first(split(wl["version"])))")
    (short, "$(cpu[1].model), macOS $(sh(`sw_vers -productVersion`)), Julia $VERSION, $(Dates.format(Dates.now(), "yyyy-mm-dd"))",
     Markdown.parse(join([table("Software", software), table("Machine", machine), table("Contest", contest_settings)], "\n\n")))
end
environment

#%% code id=symbols
@giac_var x y z t s n k a b
∞, PI = giac_eval("+infinity"), giac_eval("pi")
x + y

#%% md id=integrals_md
@md"""
## Indefinite integrals

From integration test suites: an integral that is elementary only through a clever substitution, Chebyshev's pseudo-elliptic integral, an algebraic one whose answer is an arctangent of a radical, a rational function whose denominator has no roots in radicals, and an exponential-logarithmic integrand the Risch algorithm handles.
"""

#%% code id=integrals
integrals = contest("Indefinite integrals", [
    Problem("√tan x", raw"\int \sqrt{\tan x}\,dx",
            () -> C.integrate(sqrt(tan(x)), x), "Integrate[Sqrt[Tan[x]], x]",
            Antiderivative(sqrt(tan(x)), "Sqrt[Tan[x]]")),
    Problem("Chebyshev's pseudo-elliptic integral", raw"\int \frac{x}{\sqrt{x^4 + 10x^2 - 96x - 71}}\,dx",
            () -> C.integrate(x / sqrt(x^4 + 10x^2 - 96x - 71), x), "Integrate[x/Sqrt[x^4 + 10 x^2 - 96 x - 71], x]",
            Antiderivative(x / sqrt(x^4 + 10x^2 - 96x - 71), "x/Sqrt[x^4 + 10 x^2 - 96 x - 71]"; points = ["5", "6", "7"])),
    Problem("An arctangent of a radical", raw"\int \frac{x^2 - 1}{(x^2 + 1)\sqrt{x^4 + 1}}\,dx",
            () -> C.integrate((x^2 - 1) / ((x^2 + 1) * sqrt(x^4 + 1)), x), "Integrate[(x^2 - 1)/((x^2 + 1) Sqrt[x^4 + 1]), x]",
            Antiderivative((x^2 - 1) / ((x^2 + 1) * sqrt(x^4 + 1)), "(x^2 - 1)/((x^2 + 1) Sqrt[x^4 + 1])")),
    Problem("A quintic denominator", raw"\int \frac{dx}{x^5 - x + 1}",
            () -> C.integrate(1 / (x^5 - x + 1), x), "Integrate[1/(x^5 - x + 1), x]",
            Antiderivative(1 / (x^5 - x + 1), "1/(x^5 - x + 1)")),
    Problem("A fourth root", raw"\int \frac{dx}{\sqrt[4]{1 + x^4}}",
            () -> C.integrate(1 / (1 + x^4)^(1 // 4), x), "Integrate[1/(1 + x^4)^(1/4), x]",
            Antiderivative(1 / (1 + x^4)^(1 // 4), "1/(1 + x^4)^(1/4)")),
    Problem("Parts, twice", raw"\int x^2 \arcsin^2 x\,dx",
            () -> C.integrate(x^2 * asin(x)^2, x), "Integrate[x^2 ArcSin[x]^2, x]",
            Antiderivative(x^2 * asin(x)^2, "x^2 ArcSin[x]^2"; points = ["-1/2", "3/10", "7/10"])),
    Problem("Exponential and logarithm (Risch)", raw"\int \frac{e^x (1 + x \ln x)}{x}\,dx",
            () -> C.integrate(exp(x) * (1 + x * log(x)) / x, x), "Integrate[E^x (1 + x Log[x])/x, x]",
            Antiderivative(exp(x) * (1 + x * log(x)) / x, "E^x (1 + x Log[x])/x")),
])

#%% md id=definite_md
@md"""
## Definite integrals

Famous values: an integral of Bose–Einstein type, Serret's integral, a symmetry trick, the log-sine integral, Ahmed's integral (a hard one), Laplace's integral, a dilogarithm value, Euler's constant from a Gamma-function derivative, a Gaussian with a cosine, Frullani's integral, and an integral of the complementary error function.
"""

#%% code id=definite
definite = contest("Definite integrals", [
    Problem("Bose–Einstein type", raw"\int_0^\infty \frac{x}{e^x - 1}\,dx = \frac{\pi^2}{6}",
            () -> C.integrate(x / (exp(x) - 1), x, 0, ∞), "Integrate[x/(E^x - 1), {x, 0, Infinity}]", Value(big(π)^2 / 6)),
    Problem("Serret's integral", raw"\int_0^1 \frac{\ln(1 + x)}{1 + x^2}\,dx = \frac{\pi}{8}\ln 2",
            () -> C.integrate(log(1 + x) / (1 + x^2), x, 0, 1), "Integrate[Log[1 + x]/(1 + x^2), {x, 0, 1}]", Value(big(π) / 8 * log(big(2)))),
    Problem("A symmetry trick", raw"\int_0^\pi \frac{x \sin x}{1 + \cos^2 x}\,dx = \frac{\pi^2}{4}",
            () -> C.integrate(x * sin(x) / (1 + cos(x)^2), x, 0, PI), "Integrate[x Sin[x]/(1 + Cos[x]^2), {x, 0, Pi}]", Value(big(π)^2 / 4)),
    Problem("Log-sine", raw"\int_0^{\pi/2} \ln \sin x\,dx = -\frac{\pi}{2}\ln 2",
            () -> C.integrate(log(sin(x)), x, 0, PI / 2), "Integrate[Log[Sin[x]], {x, 0, Pi/2}]", Value(-big(π) / 2 * log(big(2)))),
    Problem("Ahmed's integral", raw"\int_0^1 \frac{\arctan\sqrt{x^2 + 2}}{(x^2 + 1)\sqrt{x^2 + 2}}\,dx = \frac{5\pi^2}{96}",
            () -> C.integrate(atan(sqrt(x^2 + 2)) / ((x^2 + 1) * sqrt(x^2 + 2)), x, 0, 1),
            "Integrate[ArcTan[Sqrt[x^2 + 2]]/((x^2 + 1) Sqrt[x^2 + 2]), {x, 0, 1}]", Value(5 * big(π)^2 / 96)),
    Problem("Laplace's integral", raw"\int_0^\infty \frac{\cos x}{1 + x^2}\,dx = \frac{\pi}{2e}",
            () -> C.integrate(cos(x) / (1 + x^2), x, 0, ∞), "Integrate[Cos[x]/(1 + x^2), {x, 0, Infinity}]", Value(big(π) / (2 * exp(big(1))))),
    Problem("A dilogarithm value", raw"\int_0^1 \ln x \ln(1 - x)\,dx = 2 - \frac{\pi^2}{6}",
            () -> C.integrate(log(x) * log(1 - x), x, 0, 1), "Integrate[Log[x] Log[1 - x], {x, 0, 1}]", Value(2 - big(π)^2 / 6)),
    Problem("Euler's constant", raw"\int_0^\infty e^{-x}\ln^2 x\,dx = \gamma^2 + \frac{\pi^2}{6}",
            () -> C.integrate(exp(-x) * log(x)^2, x, 0, ∞), "Integrate[E^-x Log[x]^2, {x, 0, Infinity}]",
            Value(big(Base.MathConstants.eulergamma)^2 + big(π)^2 / 6)),
    Problem("A Gaussian with a cosine", raw"\int_{-\infty}^{\infty} e^{-x^2}\cos 2x\,dx = \frac{\sqrt\pi}{e}",
            () -> C.integrate(exp(-x^2) * cos(2x), x, -∞, ∞), "Integrate[E^(-x^2) Cos[2 x], {x, -Infinity, Infinity}]", Value(sqrt(big(π)) / exp(big(1)))),
    Problem("Frullani's integral", raw"\int_0^\infty \frac{e^{-x} - e^{-2x}}{x}\,dx = \ln 2",
            () -> C.integrate((exp(-x) - exp(-2x)) / x, x, 0, ∞), "Integrate[(E^-x - E^(-2 x))/x, {x, 0, Infinity}]", Value(log(big(2)))),
    Problem("The error function", raw"\int_0^\infty \operatorname{erfc}^2 x\,dx = \frac{2 - \sqrt2}{\sqrt\pi}",
            () -> C.integrate(C.erfc(x)^2, x, 0, ∞), "Integrate[Erfc[x]^2, {x, 0, Infinity}]", Value((2 - sqrt(big(2))) / sqrt(big(π)))),
])

#%% md id=sums_md
@md"""
## Sums and products

Zeta values, a harmonic-number series, a Dirichlet beta value, a central binomial series, Vandermonde's identity in closed form, a closed form with a power of two, a polylogarithm value, and two infinite products.
"""

#%% code id=sums
sums = contest("Sums and products", [
    Problem("ζ(4)", raw"\sum_{n \ge 1} \frac{1}{n^4} = \frac{\pi^4}{90}",
            () -> invoke_cmd(:sum, 1 / n^4, n, 1, ∞), "Sum[1/n^4, {n, 1, Infinity}]", Value(big(π)^4 / 90)),
    Problem("Harmonic numbers", raw"\sum_{n \ge 1} \frac{H_n}{n^2} = 2\zeta(3)",
            () -> invoke_cmd(:sum, invoke_cmd(:sum, 1 / k, k, 1, n) / n^2, n, 1, ∞), "Sum[HarmonicNumber[n]/n^2, {n, 1, Infinity}]",
            Value(2 * big"1.20205690315959428539973816151144999076498629234049888179227155534")),
    Problem("Dirichlet β(3)", raw"\sum_{n \ge 0} \frac{(-1)^n}{(2n + 1)^3} = \frac{\pi^3}{32}",
            () -> invoke_cmd(:sum, (-1)^n / (2n + 1)^3, n, 0, ∞), "Sum[(-1)^n/(2 n + 1)^3, {n, 0, Infinity}]", Value(big(π)^3 / 32)),
    Problem("Arithmetic-geometric", raw"\sum_{n \ge 1} \frac{n^2}{2^n} = 6",
            () -> invoke_cmd(:sum, n^2 / 2^n, n, 1, ∞), "Sum[n^2/2^n, {n, 1, Infinity}]", Value(6)),
    Problem("A central binomial series", raw"\sum_{n \ge 1} \frac{1}{n^2 \binom{2n}{n}} = \frac{\pi^2}{18}",
            () -> invoke_cmd(:sum, 1 / (n^2 * C.binomial(2n, n)), n, 1, ∞), "Sum[1/(n^2 Binomial[2 n, n]), {n, 1, Infinity}]", Value(big(π)^2 / 18)),
    Problem("Vandermonde's identity", raw"\sum_{k=0}^{n} \binom{n}{k}^2 = \binom{2n}{n}",
            () -> invoke_cmd(:sum, C.binomial(n, k)^2, k, 0, n), "Sum[Binomial[n, k]^2, {k, 0, n}]", Value(3432; at = [["n" => "7"]])),
    Problem("A closed form in n", raw"\sum_{k=1}^{n} k\,2^k = (n - 1)\,2^{n+1} + 2",
            () -> invoke_cmd(:sum, k * 2^k, k, 1, n), "Sum[k 2^k, {k, 1, n}]", Value(18434; at = [["n" => "10"]])),
    Problem("A polylogarithm value", raw"\sum_{n \ge 1} \frac{x^n}{n^2}\Big|_{x = 1/2} = \frac{\pi^2}{12} - \frac{\ln^2 2}{2}",
            () -> invoke_cmd(:sum, x^n / n^2, n, 1, ∞), "Sum[x^n/n^2, {n, 1, Infinity}]", Value(big(π)^2 / 12 - log(big(2))^2 / 2; at = [["x" => "1/2"]])),
    Problem("A telescoping product", raw"\prod_{n \ge 2} \left(1 - \frac{1}{n^2}\right) = \frac{1}{2}",
            () -> invoke_cmd(:product, 1 - 1 / n^2, n, 2, ∞), "Product[1 - 1/n^2, {n, 2, Infinity}]", Value(1 / 2)),
    Problem("Wallis's product", raw"\prod_{n \ge 1} \frac{4n^2}{4n^2 - 1} = \frac{\pi}{2}",
            () -> invoke_cmd(:product, 4n^2 / (4n^2 - 1), n, 1, ∞), "Product[4 n^2/(4 n^2 - 1), {n, 1, Infinity}]", Value(big(π) / 2)),
])

#%% md id=limits_md
@md"""
## Limits

A limit where the first six orders cancel, Stirling's growth of $n!$, a second-order term of $e$, a nested-exponential limit of the kind the Gruntz algorithm was made for, a cancellation of poles, a ratio of Gamma functions, a power of the sinc function, and a bounded oscillation.
"""

#%% code id=limits
limits = contest("Limits", [
    Problem("tan sin − sin tan", raw"\lim_{x \to 0} \frac{\tan \sin x - \sin \tan x}{x^7} = \frac{1}{30}",
            () -> C.limit((tan(sin(x)) - sin(tan(x))) / x^7, x, 0), "Limit[(Tan[Sin[x]] - Sin[Tan[x]])/x^7, x -> 0]", Value(1 / 30)),
    Problem("Stirling", raw"\lim_{n \to \infty} \frac{\sqrt[n]{n!}}{n} = \frac{1}{e}",
            () -> C.limit(C.factorial(n)^(1 / n) / n, n, ∞), "Limit[n!^(1/n)/n, n -> Infinity]", Value(exp(-big(1)))),
    Problem("A second-order term of e", raw"\lim_{x \to \infty} x\left(\left(1 + \frac{1}{x}\right)^x - e\right) = -\frac{e}{2}",
            () -> C.limit(x * ((1 + 1 / x)^x - exp(convert(GiacExpr, 1))), x, ∞), "Limit[x ((1 + 1/x)^x - E), x -> Infinity]", Value(-exp(big(1)) / 2)),
    Problem("Nested exponentials (Gruntz)", raw"\lim_{x \to \infty} e^x\left(e^{1/x - e^{-x}} - e^{1/x}\right) = -1",
            () -> C.limit(exp(x) * (exp(1 / x - exp(-x)) - exp(1 / x)), x, ∞), "Limit[E^x (E^(1/x - E^-x) - E^(1/x)), x -> Infinity]", Value(-1)),
    Problem("Cancelling poles", raw"\lim_{x \to 0} \left(\frac{1}{x^2} - \frac{1}{\sin^2 x}\right) = -\frac{1}{3}",
            () -> C.limit(1 / x^2 - 1 / sin(x)^2, x, 0), "Limit[1/x^2 - 1/Sin[x]^2, x -> 0]", Value(-1 / 3)),
    Problem("A ratio of Gamma functions", raw"\lim_{x \to \infty} \frac{\Gamma(x + \frac12)}{\Gamma(x)\sqrt{x}} = 1",
            () -> C.limit(C.Gamma(x + 1 // 2) / (C.Gamma(x) * sqrt(x)), x, ∞), "Limit[Gamma[x + 1/2]/(Gamma[x] Sqrt[x]), x -> Infinity]", Value(1)),
    Problem("A power of sinc", raw"\lim_{x \to 0} \left(\frac{\sin x}{x}\right)^{1/x^2} = e^{-1/6}",
            () -> C.limit((sin(x) / x)^(1 / x^2), x, 0), "Limit[(Sin[x]/x)^(1/x^2), x -> 0]", Value(exp(-big(1) / 6))),
    Problem("A bounded oscillation", raw"\lim_{x \to \infty} \frac{x + \sin x}{x} = 1",
            () -> C.limit((x + sin(x)) / x, x, ∞), "Limit[(x + Sin[x])/x, x -> Infinity]", Value(1)),
])

#%% md id=equations_md
@md"""
## Equations and inequalities

Polynomial systems with all their complex solutions, a quintic that is solvable by radicals, every solution of a trigonometric equation, Lambert's $W$, inequalities with polynomials and absolute values, and a condition on a parameter that needs quantifier elimination. For complex solutions Giac.jl uses GIAC's `cSolve`, as its `solve` gives the real ones.
"""

#%% code id=equations
equations = contest("Equations and inequalities", [
    Problem("A sextic system", raw"x^2 + y^2 = 1,\ x^3 = y \quad \text{(all six solutions)}",
            () -> C.cSolve([x^2 + y^2 ~ 1, x^3 ~ y], [x, y]), "Solve[{x^2 + y^2 == 1, x^3 == y}, {x, y}]",
            Solutions([x^2 + y^2 - 1, x^3 - y], "{x^2 + y^2 - 1, x^3 - y}", ["x", "y"], 6)),
    Problem("A quintic solvable by radicals", raw"x^5 - 5x + 12 = 0",
            () -> C.cSolve(x^5 - 5x + 12 ~ 0, x), "Solve[x^5 - 5 x + 12 == 0, x]",
            Form(Solutions([x^5 - 5x + 12], "{x^5 - 5 x + 12}", ["x"], 5), (:free_of, "rootof", "Root"), "roots not in radicals")),
    Problem("Every solution of cos x = ½", raw"\cos x = \tfrac12 \quad \text{(all of them, } 2\pi k \pm \tfrac{\pi}{3})",
            () -> try
                giac_eval("all_trig_solutions(1)")
                C.solve(cos(x) ~ 1 // 2, x)
            finally
                giac_eval("all_trig_solutions(0)")
            end, "Solve[Cos[x] == 1/2, x]",
            Periodic(cos(x) - 1 // 2, "Cos[x] - 1/2", "x", (-10.0, 10.0), 6)),
    Problem("Lambert's W", raw"x e^x = 2",
            () -> C.solve(x * exp(x) ~ 2, x), "Solve[x E^x == 2, x]",
            Solutions([x * exp(x) - 2], "{x E^x - 2}", ["x"], 1)),
    Problem("A symmetric system", raw"x + y + z = 6,\ xy + yz + zx = 11,\ xyz = 6",
            () -> C.solve([x + y + z ~ 6, x * y + y * z + z * x ~ 11, x * y * z ~ 6], [x, y, z]),
            "Solve[{x + y + z == 6, x y + y z + z x == 11, x y z == 6}, {x, y, z}]",
            Solutions([x + y + z - 6, x * y + y * z + z * x - 11, x * y * z - 6], "{x + y + z - 6, x y + y z + z x - 11, x y z - 6}", ["x", "y", "z"], 6)),
    Problem("A quartic inequality", raw"x^4 - 5x^2 + 4 < 0",
            () -> giac_eval("solve(x^4 - 5*x^2 + 4 < 0, x)"), "Reduce[x^4 - 5 x^2 + 4 < 0, x, Reals]",
            Region("x", v -> v^4 - 5v^2 + 4 < 0, ["-5/2", "-3/2", "-1/2", "0", "1/2", "3/2", "5/2", "-21/10", "19/10"])),
    Problem("Absolute values", raw"|x - 1| + |x + 2| < 5",
            () -> giac_eval("solve(abs(x - 1) + abs(x + 2) < 5, x)"), "Reduce[Abs[x - 1] + Abs[x + 2] < 5, x, Reals]",
            Region("x", v -> abs(v - 1) + abs(v + 2) < 5, ["-4", "-31/10", "-29/10", "0", "19/10", "21/10", "3"])),
    Problem("Quantifier elimination", raw"\text{for which } b \text{ is } x^2 + bx + 1 > 0 \text{ for every real } x?",
            () -> begin
                # Giac.jl's route: where the polynomial has no real root (msolve's real root classification)
                rc = AlgebraicSolving.real_root_classification([x^2 + b * x + 1], [x]; params = [b])
                cases = [join(["(" * string(p) * (sg > 0 ? ">0" : "<0") * ")" for (p, sg) in zip(rc.polys, c.signs)], " and ")
                         for c in rc.cases if c.count == 0]
                giac_eval("[" * join(cases, ",") * "]")
            end, "Reduce[ForAll[x, x^2 + b x + 1 > 0], b, Reals]",
            Region("b", v -> -2 < v < 2, ["-3", "-21/10", "-19/10", "0", "19/10", "21/10", "3"])),
])

#%% md id=polynomials_md
@md raw"""
## Polynomial algebra

Factoring into cyclotomic polynomials, a symmetric multivariate factorization, factoring over $\mathbb{Q}(\sqrt2)$, counting the solutions of the cyclic-6 system, eliminating a parameter to get a curve's implicit equation, and decomposing a polynomial as a composition.
"""

#%% code id=polynomials
polynomials = contest("Polynomial algebra", [
    Problem("Cyclotomic factors", raw"\text{factor } x^{105} - 1",
            () -> C.factor(x^105 - 1), "Factor[x^105 - 1]",
            Form(Value(big(7 // 10)^105 - 1; at = [["x" => "7/10"]]), (:factors, 8), "not the factorization over ℚ")),
    Problem("A symmetric septic", raw"\text{factor } (x + y + z)^7 - x^7 - y^7 - z^7",
            () -> C.factor((x + y + z)^7 - x^7 - y^7 - z^7), "Factor[(x + y + z)^7 - x^7 - y^7 - z^7]",
            Form(Value(let (u, v, w) = big.((7 // 10, 2 // 5, 3 // 10)); (u + v + w)^7 - u^7 - v^7 - w^7 end; at = [["x" => "7/10", "y" => "2/5", "z" => "3/10"]]),
                 (:factors, 4), "not fully factored")),
    Problem("Factoring over ℚ(√2)", raw"\text{factor } x^4 + 1 \text{ over } \mathbb{Q}(\sqrt{2})",
            () -> C.factor(x^4 + 1, sqrt(convert(GiacExpr, 2))), "Factor[x^4 + 1, Extension -> Sqrt[2]]",
            Form(Value(big(7 // 10)^4 + 1; at = [["x" => "7/10"]]), (:factors, 2), "not factored over the extension")),
    Problem("The cyclic-6 system", raw"\text{the number of complex solutions of cyclic-6 } (156)",
            () -> begin
                v = [Giac.giac_identifier("c$i") for i in 1:6]
                eqs = [[sum(prod(v[mod1(i + j, 6)] for j in 0:m-1) for i in 1:6) for m in 1:5]; prod(v) - 1]
                AlgebraicSolving.rational_parametrization(eqs, v).degree
            end,
            "With[{v = Array[c, 6]}, Length[NSolve[Thread[Append[Table[Sum[Product[v[[Mod[i + j - 1, 6] + 1]], {j, 0, m - 1}], {i, 6}], {m, 5}], Times @@ v - 1] == 0], v]]]",
            Value(156)),
    Problem("Implicitization", raw"\text{eliminate } t:\ x = t^2 - 1,\ y = t^3 - t",
            () -> C.resultant(x - t^2 + 1, y - t^3 + t, t), "Resultant[x - t^2 + 1, y - t^3 + t, t]",
            Vanishing([["x" => "-91/100", "y" => "-273/1000"], ["x" => "-51/100", "y" => "-357/1000"]], ["x" => "1/2", "y" => "1/2"])),
    Problem("Polynomial decomposition", raw"x^6 + 6x^4 + x^3 + 9x^2 + 3x - 5 = g(h(x))",
            () -> Giac.polynomial_decomposition(x^6 + 6x^4 + x^3 + 9x^2 + 3x - 5, x), "Decompose[x^6 + 6 x^4 + x^3 + 9 x^2 + 3 x - 5, x]",
            Decomposition("x", "7/10", let u = 0.7; u^6 + 6u^4 + u^3 + 9u^2 + 3u - 5 end)),
])

#%% md id=simplification_md
@md raw"""
## Simplification and algebraic numbers

Denesting square and cube roots (Ramanujan's identity among them), a sum of cosines that is rational, a minimal polynomial, a trigonometric identity, Gauss's construction of $\cos(2\pi/17)$ in square roots, and an inverse cosine simplified under assumptions. These ask for a form, not only a value.
"""

#%% code id=simplification
simplification = contest("Simplification and algebraic numbers", [
    Problem("Denesting", raw"\sqrt{5 + 2\sqrt{6}} = \sqrt2 + \sqrt3",
            () -> C.simplify(sqrt(5 + 2sqrt(convert(GiacExpr, 6)))), "Simplify[Sqrt[5 + 2 Sqrt[6]]]",
            Form(Value(sqrt(big(2)) + sqrt(big(3))), :denested, "not denested")),
    Problem("Ramanujan's denesting", raw"\sqrt[3]{\sqrt[3]{2} - 1} = \sqrt[3]{\tfrac19} - \sqrt[3]{\tfrac29} + \sqrt[3]{\tfrac49}",
            () -> C.simplify((convert(GiacExpr, 2)^(1 // 3) - 1)^(1 // 3)), "FullSimplify[(2^(1/3) - 1)^(1/3)]",
            Form(Value(cbrt(cbrt(big(2)) - 1)), :denested, "not denested")),
    Problem("A heptagon sum", raw"\cos\tfrac{2\pi}{7} + \cos\tfrac{4\pi}{7} + \cos\tfrac{6\pi}{7} = -\tfrac12",
            # Giac.jl's exact simplifier for algebraic constants (Nemo's algebraic numbers)
            () -> Giac.simplify_algebraic(cos(2PI / 7) + cos(4PI / 7) + cos(6PI / 7)), "FullSimplify[Cos[2 Pi/7] + Cos[4 Pi/7] + Cos[6 Pi/7]]",
            Form(Value(-1 / 2), :no_trig, "cosines left")),
    Problem("A minimal polynomial", raw"\text{the minimal polynomial of } \sqrt2 + \sqrt3 + \sqrt5",
            () -> C.pmin(sqrt(convert(GiacExpr, 2)) + sqrt(convert(GiacExpr, 3)) + sqrt(convert(GiacExpr, 5)), x),
            "MinimalPolynomial[Sqrt[2] + Sqrt[3] + Sqrt[5], x]",
            Value(let u = big(7 // 10); u^8 - 40u^6 + 352u^4 - 960u^2 + 576 end; at = [["x" => "7/10"]])),
    Problem("A trigonometric identity", raw"\frac{\sin 3x}{\sin x} - \frac{\cos 3x}{\cos x} = 2",
            () -> C.simplify(sin(3x) / sin(x) - cos(3x) / cos(x)), "Simplify[Sin[3 x]/Sin[x] - Cos[3 x]/Cos[x]]",
            Form(Value(2; at = [["x" => "7/10"]]), (:free_of, "x", "x"), "not reduced to a constant")),
    Problem("Gauss's 17-gon", raw"\cos\tfrac{2\pi}{17} \text{ in square roots}",
            () -> Giac.in_radicals(cos(2PI / 17)), "FunctionExpand[Cos[2 Pi/17]]",
            Form(Value(cos(2 * big(π) / 17)), :no_trig, "no radicals")),
    Problem("An inverse cosine, with assumptions", raw"\arccos(\cos x) = 2\pi - x \quad (\pi < x < 2\pi)",
            () -> Giac.assuming(() -> C.simplify(acos(cos(x))), x > PI, x < 2PI), "Simplify[ArcCos[Cos[x]], Pi < x < 2 Pi]",
            Form(Value(2 * big(π) - 4; at = [["x" => "4"]]), :no_trig, "not simplified")),
])

#%% md id=odes_md
@md"""
## Differential equations

The Airy and Bessel equations, a Riccati equation whose solutions are Bessel functions, a nonlinear autonomous equation, a forced Euler–Cauchy equation, a Bernoulli equation, a linear system with a Jordan block, and two partial differential equations: the transport equation's general solution and the heat equation on an interval with boundary and initial conditions. Giac.jl's `desolve` handles ordinary equations only.
"""

#%% code id=odes
@giac_var Y(x)
odes = contest("Differential equations", [
    Problem("Airy", raw"y'' = x\,y",
            () -> C.desolve(C.diff(Y, x, 2) ~ x * Y, Y), "DSolve[y''[x] == x y[x], y[x], x]",
            DE(Y -> [C.diff(Y, x, 2) - x * Y], "Function[Y, D[Y, {x, 2}] - x Y]", ["x"], [["3/10"], ["7/10"], ["11/10"]], 2)),
    Problem("Bessel, order ½", raw"x^2 y'' + x y' + \left(x^2 - \tfrac14\right) y = 0",
            () -> C.desolve(x^2 * C.diff(Y, x, 2) + x * C.diff(Y, x) + (x^2 - 1 // 4) * Y ~ 0, Y),
            "DSolve[x^2 y''[x] + x y'[x] + (x^2 - 1/4) y[x] == 0, y[x], x]",
            DE(Y -> [x^2 * C.diff(Y, x, 2) + x * C.diff(Y, x) + (x^2 - 1 // 4) * Y], "Function[Y, x^2 D[Y, {x, 2}] + x D[Y, x] + (x^2 - 1/4) Y]",
               ["x"], [["3/10"], ["7/10"], ["11/10"]], 2)),
    Problem("Riccati", raw"y' = x^2 + y^2",
            () -> C.desolve(C.diff(Y, x) ~ x^2 + Y^2, Y), "DSolve[y'[x] == x^2 + y[x]^2, y[x], x]",
            DE(Y -> [C.diff(Y, x) - x^2 - Y^2], "Function[Y, D[Y, x] - x^2 - Y^2]", ["x"], [["3/10"], ["7/10"], ["11/10"]], 1)),
    Problem("A nonlinear autonomous equation", raw"y\,y'' = y'^2",
            () -> C.desolve(Y * C.diff(Y, x, 2) ~ C.diff(Y, x)^2, Y), "DSolve[y[x] y''[x] == y'[x]^2, y[x], x]",
            DE(Y -> [Y * C.diff(Y, x, 2) - C.diff(Y, x)^2], "Function[Y, Y D[Y, {x, 2}] - D[Y, x]^2]", ["x"], [["3/10"], ["7/10"], ["11/10"]], 2)),
    Problem("A forced Euler–Cauchy equation", raw"x^2 y'' - 3x y' + 4y = x^2 \ln x",
            () -> C.desolve(x^2 * C.diff(Y, x, 2) - 3x * C.diff(Y, x) + 4Y ~ x^2 * log(x), Y),
            "DSolve[x^2 y''[x] - 3 x y'[x] + 4 y[x] == x^2 Log[x], y[x], x]",
            DE(Y -> [x^2 * C.diff(Y, x, 2) - 3x * C.diff(Y, x) + 4Y - x^2 * log(x)], "Function[Y, x^2 D[Y, {x, 2}] - 3 x D[Y, x] + 4 Y - x^2 Log[x]]",
               ["x"], [["3/10"], ["7/10"], ["11/10"]], 2)),
    Problem("Bernoulli", raw"y' + y = x\,y^3",
            () -> C.desolve(C.diff(Y, x) + Y ~ x * Y^3, Y), "DSolve[y'[x] + y[x] == x y[x]^3, y[x], x]",
            DE(Y -> [C.diff(Y, x) + Y - x * Y^3], "Function[Y, D[Y, x] + Y - x Y^3]", ["x"], [["3/10"], ["7/10"], ["11/10"]], 1)),
    Problem("A Jordan block", raw"X' = \begin{bmatrix}1&1&0\\0&1&1\\0&0&1\end{bmatrix} X,\ X(0) = (1, 1, 1)",
            () -> exp(t * GiacExpr[1 1 0; 0 1 1; 0 0 1]) * GiacExpr[1, 1, 1], "MatrixExp[t {{1, 1, 0}, {0, 1, 1}, {0, 0, 1}}].{1, 1, 1}",
            Value(let u = big(7 // 10); exp(u) .* [1 + u + u^2 / 2, 1 + u, 1] end; at = [["t" => "7/10"]])),
    Problem("The transport equation", raw"u_x + 2u_y = 0 \quad \text{(general solution)}",
            () -> giac_eval("desolve(diff(u(x,y),x) + 2*diff(u(x,y),y) = 0, u(x,y))"),
            "DSolve[D[u[x, y], x] + 2 D[u[x, y], y] == 0, u[x, y], {x, y}]",
            DE(U -> [C.diff(U, x) + 2C.diff(U, y)], "Function[U, D[U, x] + 2 D[U, y]]", ["x", "y"], [["3/10", "1/2"], ["7/10", "-1"]], 1; unknown = "u[x, y]")),
    Problem("The heat equation", raw"u_t = u_{xx},\ u(0, t) = u(\pi, t) = 0,\ u(x, 0) = \sin x",
            () -> giac_eval("desolve([diff(u(x,t),t) = diff(u(x,t),x,2), u(0,t) = 0, u(pi,t) = 0, u(x,0) = sin(x)], u(x,t))"),
            "DSolve[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[0, t] == 0, u[Pi, t] == 0, u[x, 0] == Sin[x]}, u[x, t], {x, t}]",
            DE(U -> [C.diff(U, t) - C.diff(U, x, 2), C.subst(U, x, 0), C.subst(U, x, PI), C.subst(U, t, 0) - sin(x)],
               "Function[U, {D[U, t] - D[U, {x, 2}], U /. x -> 0, U /. x -> Pi, (U /. t -> 0) - Sin[x]}]",
               ["x", "t"], [["3/10", "1/2"], ["7/10", "1"], ["2", "3/2"]], 0; unknown = "u[x, t]")),
])

#%% md id=linear_md
@md"""
## Linear algebra

Eigenvalues of a matrix with a parameter, a Jordan form with a non-trivial block, a matrix exponential, a factored Vandermonde determinant, and a Smith normal form.
"""

#%% code id=linear
linear = contest("Linear algebra", [
    Problem("Symbolic eigenvalues", raw"\text{eigenvalues of } \begin{bmatrix}a&1&0\\1&a&1\\0&1&a\end{bmatrix}",
            () -> eigvals(GiacExpr[a 1 0; 1 a 1; 0 1 a]), "Eigenvalues[{{a, 1, 0}, {1, a, 1}, {0, 1, a}}]",
            Value(let u = big(13 // 10); [u - sqrt(big(2)), u, u + sqrt(big(2))] end; at = [["a" => "13/10"]], sorted = true)),
    Problem("A Jordan form", raw"J \text{ for } \begin{bmatrix}5&4&2&1\\0&1&-1&-1\\-1&-1&3&0\\1&1&-1&2\end{bmatrix}",
            () -> C.jordan(giac_eval("[[5,4,2,1],[0,1,-1,-1],[-1,-1,3,0],[1,1,-1,2]]")),
            "JordanDecomposition[{{5, 4, 2, 1}, {0, 1, -1, -1}, {-1, -1, 3, 0}, {1, 1, -1, 2}}]",
            # the eigenvalues on J's diagonal, sorted, and the number of ones above it
            Value([1, 2, 4, 4, 1]; giac = ans -> (J = _items(_items(ans)[2]); vcat(sort([_items(J[i])[i] for i in 1:4]; by = v -> real(_num(v))),
                                                                                    [round(Int, sum(real(_num(_items(J[i])[i + 1])) for i in 1:3))])),
                  wolfram = "Function[ans, Append[Sort[Diagonal[ans[[2]]]], Total[Diagonal[ans[[2]], 1]]]]")),
    Problem("A matrix exponential", raw"\exp\left(t \begin{bmatrix}0&1\\-1&0\end{bmatrix}\right)",
            () -> exp(t * GiacExpr[0 1; -1 0]), "MatrixExp[t {{0, 1}, {-1, 0}}]",
            Value(let u = big(7 // 10); [cos(u) sin(u); -sin(u) cos(u)] end; at = [["t" => "7/10"]])),
    Problem("A Vandermonde determinant", raw"\text{factor } \det\left[v_i^{\,j-1}\right]_{5\times5},\ v = (a, b, x, y, z)",
            () -> C.factor(det(GiacExpr[v^j for v in (a, b, x, y, z), j in 0:4])), "Factor[Det[Table[v^j, {v, {a, b, x, y, z}}, {j, 0, 4}]]]",
            Form(Value(let v = big.([13 // 10, 2 // 5, 7 // 10, -1 // 2, 3]); prod(v[j] - v[i] for i in 1:5 for j in i+1:5) end;
                       at = [["a" => "13/10", "b" => "2/5", "x" => "7/10", "y" => "-1/2", "z" => "3"]]), (:factors, 10), "not factored")),
    Problem("A Smith normal form", raw"\text{Smith form of } \begin{bmatrix}2&4&4\\-6&6&12\\10&-4&-16\end{bmatrix}",
            () -> C.ismith(giac_eval("[[2,4,4],[-6,6,12],[10,-4,-16]]")), "SmithDecomposition[{{2, 4, 4}, {-6, 6, 12}, {10, -4, -16}}]",
            Value([2 0 0; 0 6 0; 0 0 12]; giac = ans -> [x for r in _items(_items(ans)[2]) for x in _items(r)], wolfram = "Function[ans, ans[[2]]]")),
])

#%% md id=numbers_md
@md"""
## Number theory

Factoring the Fermat number $F_7$, proving a 157-digit Mersenne number prime, a discrete logarithm modulo $10^9 + 7$, Pell's equation, a periodic continued fraction, and the partition number $p(200)$. Giac.jl reaches PARI through GIAC's `pari` command where GIAC has no command of its own.
"""

#%% code id=numbers
# the integers of a product like `p1*p2^3`, each as often as its power
_product_integers(e) = [parse(BigInt, first(f)) for t in split(string(e), '*') for f in [split(t, '^')] for _ in 1:(length(f) == 2 ? parse(Int, f[2]) : 1)]
const PRIME = 1_000_000_007
numbers = contest("Number theory", [
    Problem("The Fermat number F₇", raw"\text{factor } 2^{128} + 1",
            () -> giac_eval("ifactor(2^128 + 1)"), "FactorInteger[2^128 + 1]",
            Integers(v -> sort(v) == [59649589127497217, 5704689200685129054721], _product_integers, "Function[a, First /@ a]")),
    Problem("A Mersenne prime", raw"2^{521} - 1 \text{ is prime}",
            () -> giac_eval("isprime(2^521 - 1)"), "PrimeQ[2^521 - 1]", Truth()),
    Problem("A discrete logarithm", raw"5^x \equiv 123456789 \pmod{10^9 + 7}",
            () -> giac_eval("pari(\"znlog\", Mod(123456789, $PRIME), Mod(5, $PRIME))"), "MultiplicativeOrder[5, $PRIME, {123456789}]",
            Integers(v -> powermod(5, v[1], PRIME) == 123456789, a -> [parse(BigInt, string(a))], "Identity")),
    Problem("Pell's equation", raw"x^2 - 61y^2 = 1 \text{ in positive integers}",
            () -> giac_eval("isolve(x^2 - 61*y^2 = 1)"), "FindInstance[x^2 - 61 y^2 == 1 && x > 1 && y > 0, {x, y}, Integers]",
            # GIAC's family of solutions, at its parameter 1
            Integers(v -> length(v) == 2 && v[1] > 1 && v[2] > 0 && v[1]^2 - 61v[2]^2 == 1,
                     a -> [string(C.simplify(giac_eval(replace(string(Giac.rhs(eq)), r"_N\d+" => "1")))) for eq in _items(a)],
                     "Function[a, {x, y} /. First[a]]")),
    Problem("A periodic continued fraction", raw"\sqrt{61} = [7; \overline{1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14}]",
            () -> giac_eval("dfc(sqrt(61))"), "ContinuedFraction[Sqrt[61]]",
            Integers(v -> length(v) >= 12 && v[1:12] == [7, 1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14], a -> string.(_items(a)), "Identity")),
    Problem("Partitions", raw"p(200) = 3972999029388",
            () -> giac_eval("pari(\"numbpart\", 200)"), "PartitionsP[200]",
            Integers(v -> v == [3972999029388], a -> [string(a)], "Identity")),
])

#%% md id=transforms_md
@md"""
## Transforms and special functions

Laplace transforms forward and back, a delayed step, the series of Lambert's $W$, a Gauss hypergeometric function in closed form, and the reflection formula of the Gamma function.
"""

#%% code id=transforms
transforms = contest("Transforms and special functions", [
    Problem("A Laplace transform", raw"\mathcal{L}\{t^2 e^{-t} \sin t\}(s)",
            () -> C.laplace(t^2 * exp(-t) * sin(t), t, s), "LaplaceTransform[t^2 E^-t Sin[t], t, s]",
            Value(52 // 1000; at = [["s" => "2"]])),
    Problem("An inverse Laplace transform", raw"\mathcal{L}^{-1}\left\{\frac{1}{(s^2 + 1)^2}\right\}(t) = \frac{\sin t - t\cos t}{2}",
            () -> C.ilaplace(1 / (s^2 + 1)^2, s, t), "InverseLaplaceTransform[1/(s^2 + 1)^2, s, t]",
            Value(let u = big(7 // 10); (sin(u) - u * cos(u)) / 2 end; at = [["t" => "7/10"]])),
    Problem("A delayed step", raw"\mathcal{L}^{-1}\left\{\frac{e^{-s}}{s}\right\}(t) = \theta(t - 1)",
            () -> C.ilaplace(exp(-s) / s, s, t), "InverseLaplaceTransform[E^-s/s, s, t]",
            Value([0, 1]; at = [["t" => "1/2"], ["t" => "3/2"]])),
    Problem("The series of Lambert's W", raw"W(x) = x - x^2 + \tfrac32 x^3 - \tfrac83 x^4 + \tfrac{125}{24} x^5 + O(x^6)",
            () -> C.series(C.LambertW(x), x, 0, 5), "Series[ProductLog[x], {x, 0, 5}]",
            Value(let u = big(1 // 10); u - u^2 + 3u^3 / 2 - 8u^4 / 3 + 125u^5 / 24 end; at = [["x" => "1/10"]],
                  giac = a -> C.convert(a, giac_eval("polynom")), wolfram = "Normal")),
    Problem("A hypergeometric function", raw"{}_2F_1(1, 1; 2; x) = -\frac{\ln(1 - x)}{x}",
            () -> Giac.hypergeometric([1, 1], [2], x), "FunctionExpand[Hypergeometric2F1[1, 1, 2, x]]",
            Form(Value(2log(big(2)); at = [["x" => "1/2"]]), (:free_of, "hypergeom", "Hypergeometric2F1"), "not in closed form")),
    Problem("Gamma's reflection formula", raw"\Gamma\left(\tfrac14\right)\Gamma\left(\tfrac34\right) = \pi\sqrt2",
            () -> C.simplify(C.Gamma(giac_eval("1/4")) * C.Gamma(giac_eval("3/4"))), "FullSimplify[Gamma[1/4] Gamma[3/4]]",
            Form(Value(big(π) * sqrt(big(2))), (:free_of, "Gamma", "Gamma"), "Gamma left")),
])

#%% md id=control_md
@md"""
## Control

From a nonlinear control course: the gains that keep a cubic characteristic polynomial stable (Routh–Hurwitz), a Lyapunov equation, the algebraic Riccati equation of an LQR problem, the Lie algebra rank condition for the unicycle, a center manifold, and two polynomial inequalities, one with a sum-of-squares certificate and Motzkin's polynomial, which is nonnegative but not a sum of squares. Mathematica has Lyapunov and Riccati solvers; for the rest its user writes a few lines, as here.
"""

#%% code id=control
control = contest("Control", [
    Problem("A stability region", raw"\text{for which } k \text{ is } s^3 + 3s^2 + 3s + 1 + k \text{ Hurwitz?}",
            () -> giac_eval("solve([" * join(string.(Giac.Control.hurwitz_conditions(s^3 + 3s^2 + 3s + 1 + k, s)) .* ">0", ",") * "], k)"),
            "Reduce[ForAll[{u, w}, Implies[ComplexExpand[Re[(s^3 + 3 s^2 + 3 s + 1 + k) /. s -> u + I w]] == 0 && ComplexExpand[Im[(s^3 + 3 s^2 + 3 s + 1 + k) /. s -> u + I w]] == 0, u < 0]], k, Reals]",
            Region("k", v -> -1 < v < 8, ["-3/2", "-1/2", "0", "7", "15/2", "17/2", "10"])),
    Problem("A Lyapunov equation", raw"A^{\mathsf T} P + P A = -I,\ A = \begin{bmatrix}0&1\\-2&-3\end{bmatrix}",
            () -> let A = GiacExpr[0 1; -2 -3], p = [Giac.giac_identifier("p$i") for i in 1:3], P = GiacExpr[p[1] p[2]; p[2] p[3]]
                C.solve(transpose(A) * P + P * A + I ~ zeros(GiacExpr, 2, 2), P)
            end, "LyapunovSolve[Transpose[{{0, 1}, {-2, -3}}], -IdentityMatrix[2]]",
            Value([5//4 1//4; 1//4 1//4])),
    Problem("A Riccati equation (LQR)", raw"A^{\mathsf T} P + P A - P B B^{\mathsf T} P + I = 0,\ A = \begin{bmatrix}0&1\\0&0\end{bmatrix},\ B = \begin{bmatrix}0\\1\end{bmatrix}",
            () -> let A = GiacExpr[0 1; 0 0], B = GiacExpr[0; 1;;], p = [Giac.giac_identifier("p$i") for i in 1:3], P = GiacExpr[p[1] p[2]; p[2] p[3]]
                R = transpose(A) * P + P * A - P * B * transpose(B) * P + I
                # the real solutions, and of them the positive definite one, which stabilizes
                sols = C.solve([R[1, 1] ~ 0, R[1, 2] ~ 0, R[2, 2] ~ 0], p)
                only(filter(v -> isposdef(real.(_num.([v[1] v[2]; v[2] v[3]]))), [_items(v) for v in _items(sols)])) |> v -> GiacExpr[v[1] v[2]; v[2] v[3]]
            end, "RiccatiSolve[{{{0, 1}, {0, 0}}, {{0}, {1}}}, {IdentityMatrix[2], {{1}}}]",
            Value([sqrt(big(3)) 1; 1 sqrt(big(3))])),
    Problem("The unicycle's Lie algebra rank", raw"\operatorname{rank}\,\{f_1, f_2, [f_1, f_2]\},\ f_1 = (\cos\theta, \sin\theta, 0),\ f_2 = (0, 0, 1)",
            () -> let X = [Giac.giac_identifier(v) for v in ("px", "py", "th")], f1 = [cos(X[3]), sin(X[3]), 0], f2 = GiacExpr[0, 0, 1]
                rank(hcat(f1, f2, Giac.Control.lie_bracket(f1, f2, X)))
            end, "With[{f1 = {Cos[th], Sin[th], 0}, f2 = {0, 0, 1}, X = {px, py, th}}, MatrixRank[Transpose[{f1, f2, D[f2, {X}].f1 - D[f1, {X}].f2}]]]",
            Value(3)),
    Problem("A center manifold", raw"\dot x = xy,\ \dot y = -y - x^2: \quad y = h(x) = -x^2 - 2x^4 + O(x^6)",
            () -> last(only(Giac.Control.center_manifold([x * y, -y - x^2], [x, y]; center = [x], order = 4).h)),
            "Module[{h = c2 x^2 + c3 x^3 + c4 x^4}, h /. First[Solve[Thread[Take[CoefficientList[D[h, x] x h + h + x^2, x], {3, 5}] == 0], {c2, c3, c4}]]]",
            Value(-big(1 // 100) - 2big(1 // 10000); at = [["x" => "1/10"]])),
    Problem("A sum-of-squares certificate", raw"x^4 - 2x^2 y + y^2 + x^2 - 2x + 1 \ge 0",
            () -> Giac.sign_of(x^4 - 2x^2 * y + y^2 + x^2 - 2x + 1) in (:nonnegative, :positive),
            "Resolve[ForAll[{x, y}, x^4 - 2 x^2 y + y^2 + x^2 - 2 x + 1 >= 0], Reals]", Truth()),
    Problem("Motzkin's polynomial", raw"x^4 y^2 + x^2 y^4 - 3x^2 y^2 + 1 \ge 0 \quad \text{(not a sum of squares)}",
            () -> Giac.sign_of(x^4 * y^2 + x^2 * y^4 - 3x^2 * y^2 + 1) in (:nonnegative, :positive),
            "Resolve[ForAll[{x, y}, x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1 >= 0], Reals]", Truth()),
])

#%% md id=summary_md
@md"""
## Summary

Each area's problems by outcome, for both systems, and the problems one system solved and the other didn't.
"""

#%% code id=summary
"The outcome counts of each area, and the problems only one system solved."
struct Scoreboard
    areas::Vector{Contest}
end
const _ORDER = (:solved, :partial, :unsolved, :wrong)
_counts(as) = [count(a -> a.outcome === o, as) for o in _ORDER]

function Base.show(io::IO, ::MIME"text/html", b::Scoreboard)
    print(io, CONTEST_CSS, "<table class=\"contest\"><tr><th>Area</th><th>Problems</th>")
    for sys in ("Giac.jl", "Mathematica")
        print(io, "<th>", sys, "</th>")
    end
    print(io, "</tr>")
    total_g, total_w = zeros(Int, 4), zeros(Int, 4)
    cell(c) = join(["<span style=\"color:$(_OUTCOME_STYLE[o][1])\">$n $(_OUTCOME_STYLE[o][2])</span>" for (o, n) in zip(_ORDER, c) if n > 0], " · ")
    for c in b.areas
        g, w = _counts(c.giac), _counts(c.wolfram)
        total_g .+= g; total_w .+= w
        print(io, "<tr><td>", _esc(c.area), "</td><td>", length(c.problems), "</td><td>", cell(g), "</td><td>", cell(w), "</td></tr>")
    end
    print(io, "<tr><td><b>All</b></td><td><b>", sum(length(c.problems) for c in b.areas), "</b></td><td><b>", cell(total_g), "</b></td><td><b>", cell(total_w), "</b></td></tr></table>")
    for (who, mine, other) in (("Giac.jl", :giac, :wolfram), ("Mathematica", :wolfram, :giac))
        alone = [(c.area, p.title) for c in b.areas for (p, x, y) in zip(c.problems, getfield(c, mine), getfield(c, other))
                if x.outcome === :solved && y.outcome !== :solved]
        print(io, "<p><b>Solved by ", who, " only</b> (", length(alone), "): ", isempty(alone) ? "none" : join([_esc(t) * " <span style=\"opacity:.6\">(" * _esc(a) * ")</span>" for (a, t) in alone], ", "), "</p>")
    end
end
function Base.show(io::IO, ::MIME"text/markdown", b::Scoreboard)
    println(io, "| Area | Problems | Giac.jl (solved/partial/not/wrong) | Mathematica (solved/partial/not/wrong) |\n|:--|:--|:--|:--|")
    for c in b.areas
        println(io, "| ", c.area, " | ", length(c.problems), " | ", join(_counts(c.giac), "/"), " | ", join(_counts(c.wolfram), "/"), " |")
    end
end
scoreboard = Scoreboard([integrals, definite, sums, limits, equations, polynomials, simplification, odes, linear, numbers, transforms, control])

#%% md id=performance_md
@md"""
## Performance

For each area: how many problems each system solved, and on the problems both solved, how many each answered faster and the typical ratio of their times, the geometric mean of one system's time over the other's. "Giac.jl 3.0× faster" means its answers took a third of Mathematica's time, typically, on the problems both solved. A problem only one system solved is not timed against anything.

Each time is of a system computing an answer it has not seen, with its code ready, so neither Julia compiling Giac.jl's code nor Mathematica loading its own on first use is counted. Giac.jl computes a problem twice and the second run is timed: neither it nor GIAC keeps results, and a new problem of a kind already run takes as long as a repeat. Mathematica cannot be timed that way, as `Sum` and `Product` remember their results through `ClearSystemCache[]` and answer a repeat in a tenth of a millisecond; it is warmed up on unrelated problems of the same kinds (integrals, sums, limits, solving, factoring, differential equations, transforms, matrices) and then times each problem once. Mathematica's time is its kernel's `AbsoluteTiming`. Giac.jl's is the call's time in Julia, through its Julia layers to GIAC and, for the problems that use them, Nemo, msolve and Clarabel. The plot shows every time.
"""

#%% code id=performance
"Per area: problems each system solved, and on those both solved, how often each was faster and the typical ratio of their times."
struct Performance
    rows::Vector{Pair{String, NamedTuple}}
end
const SYSTEMS = ("Giac.jl", "Mathematica")

_solved_time(a::Attempt) = a.outcome === :solved && a.seconds !== nothing && a.seconds > 0 ? a.seconds : nothing

function _performance(pairs)
    solved = [count(p -> _solved_time(p[i]) !== nothing, pairs) for i in 1:2]
    both = [(_solved_time(g), _solved_time(w)) for (g, w) in pairs if _solved_time(g) !== nothing && _solved_time(w) !== nothing]
    faster = (count(((g, w),) -> g < w, both), count(((g, w),) -> w < g, both))
    # Mathematica's time over Giac.jl's, the geometric mean: above 1, Giac.jl is the faster
    ratio = isempty(both) ? nothing : exp(sum(log(w / g) for (g, w) in both) / length(both))
    (; problems = length(pairs), solved, both = length(both), faster, ratio)
end

_pairs(c::Contest) = [(g, w) for (g, w) in zip(c.giac, c.wolfram)]
performance = Performance(vcat([c.area => _performance(_pairs(c)) for c in scoreboard.areas],
                               ["All problems" => _performance(reduce(vcat, _pairs.(scoreboard.areas)))]))

_ratio_text(r) = r === nothing ? "—" : r >= 1 ? "Giac.jl $(_times(r)) faster" : "Mathematica $(_times(1 / r)) faster"
_times(r) = (r < 10 ? string(round(r; digits = 1)) : string(round(Int, r))) * "×"
# orange toward Giac.jl, blue toward Mathematica, stronger with the ratio (full at 100×)
function _ratio_colour(r)
    r === nothing && return "transparent"
    a = round(clamp(abs(log10(r)) / 2, 0.05, 1) * 0.45; digits = 2)
    r >= 1 ? "rgba(255,165,0,$a)" : "rgba(0,191,255,$a)"
end

function Base.show(io::IO, ::MIME"text/html", p::Performance)
    print(io, """<style>
      table.perf { border-collapse: collapse; margin: 6px 0; font-size: .92em; }
      table.perf th, table.perf td { padding: 7px 12px; border-bottom: 1px solid rgba(128,128,128,.22); text-align: left; }
      table.perf td.n { font-variant-numeric: tabular-nums; }
      table.perf td.r { font-weight: 600; }
      table.perf tr.all td { border-top: 2px solid rgba(128,128,128,.5); font-weight: 600; }
    </style><table class="perf"><tr><th></th><th>Solved: Giac.jl</th><th>Solved: Mathematica</th><th>Both solved</th>
    <th>Faster: Giac.jl</th><th>Faster: Mathematica</th><th>Typical time ratio</th></tr>""")
    for (name, r) in p.rows
        print(io, name == "All problems" ? "<tr class=\"all\">" : "<tr>", "<td>", _esc(name), "</td>",
              "<td class=\"n\">$(r.solved[1]) of $(r.problems)</td><td class=\"n\">$(r.solved[2]) of $(r.problems)</td>",
              "<td class=\"n\">$(r.both)</td><td class=\"n\">$(r.faster[1])</td><td class=\"n\">$(r.faster[2])</td>",
              "<td class=\"r\" style=\"background:$(_ratio_colour(r.ratio))\">", _ratio_text(r.ratio), "</td></tr>")
    end
    print(io, "</table>")
end

function Base.show(io::IO, ::MIME"text/markdown", p::Performance)
    println(io, "| | Solved: Giac.jl | Solved: Mathematica | Both solved | Faster: Giac.jl | Faster: Mathematica | Typical time ratio |\n|:--|:--|:--|:--|:--|:--|:--|")
    for (name, r) in p.rows
        println(io, "| $name | $(r.solved[1])/$(r.problems) | $(r.solved[2])/$(r.problems) | $(r.both) | $(r.faster[1]) | $(r.faster[2]) | $(_ratio_text(r.ratio)) |")
    end
end
performance

#%% code id=performance_plot
using CairoMakie
set_theme!(theme_dark())
let rows = [(c.area, p.title, (g, w)) for c in scoreboard.areas for (p, g, w) in zip(c.problems, c.giac, c.wolfram)]
    colours, markers = [:orange, :deepskyblue], [:circle, :rect]
    n = length(rows)
    times = [a.seconds for r in rows for a in r[3] if a.seconds !== nothing && a.seconds > 0]
    decades = floor(Int, log10(minimum(times))):ceil(Int, log10(maximum(times)))
    fig = Figure(size = (1100, 18n + 120))
    ax = Axis(fig[2, 1]; xscale = log10, xlabel = "seconds",
              xticks = (10.0 .^ decades, [d >= 0 ? string(10^d) : "10^$d" for d in decades]),
              yticks = (1:n, [first(t, 46) for (_, t, _) in rows]), yreversed = true,
              title = "every problem's time; hollow: not solved", subtitle = machine_summary, yticklabelsize = 10)
    for (i, (col, mk)) in enumerate(zip(colours, markers))
        good = [(r[3][i].seconds, j) for (j, r) in enumerate(rows) if r[3][i].seconds !== nothing && r[3][i].seconds > 0 && r[3][i].outcome === :solved]
        poor = [(r[3][i].seconds, j) for (j, r) in enumerate(rows) if r[3][i].seconds !== nothing && r[3][i].seconds > 0 && r[3][i].outcome !== :solved]
        isempty(good) || scatter!(ax, first.(good), last.(good); color = col, marker = mk, markersize = 9, label = short_versions[SYSTEMS[i]])
        isempty(poor) || scatter!(ax, first.(poor), last.(poor); color = :transparent, strokecolor = col, strokewidth = 1.5, marker = mk, markersize = 9)
    end
    # the areas: marked off, and named on the right at the middle of each
    starts = [j for j in 1:n if j == 1 || rows[j][1] != rows[j - 1][1]]
    hlines!(ax, starts[2:end] .- 0.5; color = (:white, 0.15))
    ends = [starts[2:end] .- 1; n]
    areas = Axis(fig[2, 1]; yaxisposition = :right, yreversed = true, backgroundcolor = :transparent,
                 yticks = ((starts .+ ends) ./ 2, [rows[s][1] for s in starts]), yticklabelsize = 12, yticksvisible = false, ygridvisible = false)
    hidexdecorations!(areas); hidespines!(areas)
    linkyaxes!(ax, areas)
    Legend(fig[1, 1], ax; orientation = :horizontal, framevisible = false)
    fig
end

#%% md id=findings_md
@md raw"""
## What the contest shows

Mathematica answers almost everything here. It misses Ahmed's integral, which Giac.jl misses too, and the discrete logarithm, which takes it a little over the minute. Of the problems asking for a form, it leaves Ramanujan's denesting, a square root it does not denest and an inverse cosine its `Simplify` leaves under the assumptions; Giac.jl does the last two. Giac.jl is level with it on limits, linear algebra, control, the transforms and special functions, and number theory, nearly so on equations, polynomial algebra and definite integrals, and it has no wrong answers. Where both solve a problem Giac.jl is usually the faster, by a wide margin on definite integrals and limits; Mathematica is faster on sums and on its own matrix and control routines.

Many of Giac.jl's answers come from its own layers rather than from GIAC alone: definite integrals as derivatives of Beta and Gamma integrals, with sines and cosines as exponentials, or averaged over an involution of the interval (Serret's integral, the symmetry trick); infinite sums and products as limits of GIAC's partial ones, alternating sums in pairs of terms with polygamma values reflected, and Chu–Vandermonde for finite hypergeometric sums; `hypergeometric`; `polynomial_decomposition`; exact complex solutions where GIAC's `cSolve` gives decimals; Euler–Cauchy equations in `desolve`; Nemo's exact algebraic numbers (the heptagon sum); Gauss's periods for constructible angles (`in_radicals`); and a sum-of-squares certificate with a positive multiplier (Motzkin's polynomial).

Where it falls behind, the gaps are in a few places:

- **Integration**: GIAC's integrator gives back algebraic integrands that need a clever substitution (Chebyshev's pseudo-elliptic integral, the arctangent of a radical, the fourth root) and a rational function whose denominator has no roots in radicals, which Mathematica writes as a `RootSum`; $\int_0^\infty \operatorname{erfc}^2 x\,dx$ stays unevaluated.
- **Symbolic sums**: harmonic-number and central binomial series, and polylogarithm values; GIAC has no polylogarithm.
- **Differential equations**: no Airy functions (GIAC has them but not their derivatives), no Riccati equation with Bessel solutions, and no partial differential equations.
- **Exact forms**: the solvable quintic's roots are exact but not in radicals, and GIAC's `factor` splits $\Phi_5$ with $\sqrt5$, so $x^{105} - 1$ is not factored over $\mathbb{Q}$.

One failure is a defect rather than missing mathematics: `isprime(2^521 - 1)` fails inside PARI, whose APRCL test errs on this number though it proves other primes of the same size. A PARI error also leaves PARI locked for the rest of the session, and `pari_unlock` locks a PARI that is free; the harness works around both.
"""

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   AlgebraicSolving 0.13.0 66b61cbe-0446-4d5d-9090-1ff510639f9d
#   CairoMakie 0.15.17 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   Clarabel 0.11.1 61c947e1-3e6d-4ee4-985a-eec8c727bd6e
#   DedicatedThreads 0.1.0 9d6577c3-9664-4440-a53f-cec765ebc045
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   JSON 1.13.4 682c06a0-de6a-54ab-a142-c8b1cf79cde6
#   JuMP 1.32.1 4076af6c-e467-56ae-b986-b466b2749572
#   Nemo 0.57.2 2edaba10-b0f1-5616-af89-8c11ac63239a
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = f328007c-4c53-4cb8-9a60-3174d7f3bc07
#   format = 2
# ╚═╡
