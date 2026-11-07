# The independent CAS integration test suite (Abbasi's, built on Rubi's test files): a catalogue of
# its integrals and the 2024 results of nine systems, a runner that integrates a selection of them
# with Giac.jl in worker processes, and the grading of each answer.
#
# The catalogue is made once from the suite's SQLite database; a run's results are kept, one line
# of JSON per integral, under `~/.cache/giac/integration_suite/runs/<name>/`.

module IntegrationSuite

using Giac, JSON, Dates

const C = Giac.Commands

const DATA = joinpath(homedir(), ".cache", "giac", "integration_suite")
const DATABASE_URL = "https://www.12000.org/my_notes/CAS_integration_tests/reports/summer_2024/DATA_BASE/cas_integration_tests.zip"
catalogue_file() = joinpath(DATA, "catalogue.jsonl")
runs_dir() = joinpath(DATA, "runs")

# the systems of the 2024 results, by their prefix in the database
const SYSTEMS_2024 = ["rubi", "mma", "maple", "fricas", "maxima", "giac", "mupad", "reduce", "sympy"]
const SYSTEM_NAMES = Dict("rubi" => "Rubi 4.17.3", "mma" => "Mathematica 14", "maple" => "Maple 2024",
                          "fricas" => "FriCAS 1.3.10", "maxima" => "Maxima 5.47", "giac" => "GIAC 1.9.0-99",
                          "mupad" => "MuPAD (Matlab 2021a)", "reduce" => "Reduce", "sympy" => "SymPy 1.12")

# ---------------------------------------------------------------------------------------------
# The integrands in GIAC's syntax
# ---------------------------------------------------------------------------------------------

# The suite writes its integrands for Sage. Its parameters `e`, `i` and `epsilon` are constants
# in GIAC, so they are renamed; `I` is the imaginary unit.
const RENAMED = Dict("e" => "e_", "i" => "i_", "epsilon" => "epsilon_", "I" => "i")

# Sage's functions by GIAC's names, or as GIAC expressions of their arguments
const FUNCTIONS = Dict{String, Any}(
    "arctan" => "atan", "arcsin" => "asin", "arccos" => "acos", "arccot" => "acot", "arcsec" => "asec",
    "arccsc" => "acsc", "arcsinh" => "asinh", "arccosh" => "acosh", "arctanh" => "atanh",
    "arccoth" => "acoth", "gamma" => "Gamma", "sin_integral" => "Si",
    "cos_integral" => "Ci", "log_integral" => "Li", "lngamma" => "lgamma",
    "arcsech" => a -> "acosh(1/($(a[1])))",
    "arccsch" => a -> "asinh(1/($(a[1])))",
    "erfi" => a -> "(-i*erf(i*($(a[1]))))",
    "Shi" => a -> "(-i*Si(i*($(a[1]))))",
    # Chi z = Ci(i z) - ln(i z) + ln z
    "Chi" => a -> "(Ci(i*($(a[1])))-ln(i*($(a[1])))+ln($(a[1])))",
    # ψ⁽ⁿ⁾(z) is Psi(z, n) in GIAC
    "Psi" => a -> length(a) == 2 ? "Psi($(a[2]),$(a[1]))" : "Psi($(a[1]))",
    # LambertW's branch k comes second in GIAC
    "lambert_w" => a -> length(a) == 2 ? "LambertW($(a[2]),$(a[1]))" : "LambertW($(a[1]))",
    # the Hurwitz zeta function, which GIAC has not
    "Zeta" => a -> length(a) == 1 ? "Zeta($(a[1]))" : nothing,
)
# functions GIAC has no form of
const UNSUPPORTED = Set(["polylog", "fresnel_sin", "fresnel_cos", "exp_integral_e"])

# the top-level arguments of the call whose opening parenthesis is at `open`, and the index of its
# closing parenthesis
function _arguments(s::String, open::Int)
    depth, start, args = 0, open + 1, String[]
    i = open
    while i <= lastindex(s)
        c = s[i]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            if depth == 0
                push!(args, s[start:prevind(s, i)])
                return args, i
            end
        elseif c == ',' && depth == 1
            push!(args, s[start:prevind(s, i)])
            start = nextind(s, i)
        end
        i = nextind(s, i)
    end
    error("unbalanced parentheses in $s")
end

"""
    to_giac(s) -> (text, unsupported)

The Sage expression `s` in GIAC's syntax, and the functions in it GIAC has no form of.
"""
function to_giac(s::AbstractString)
    s = String(s)
    out, unsupported = IOBuffer(), String[]
    i = firstindex(s)
    while i <= lastindex(s)
        m = match(r"[A-Za-z_][A-Za-z_0-9]*", s, i)
        if m === nothing || m.offset != i
            print(out, s[i]); i = nextind(s, i); continue
        end
        name = m.match
        after = i + ncodeunits(name)
        if after <= lastindex(s) && s[after] == '('
            name in UNSUPPORTED && push!(unsupported, name)
            f = get(FUNCTIONS, name, name)
            if f isa Function
                args, close = _arguments(s, after)
                inner = [to_giac(a) for a in args]
                foreach(a -> append!(unsupported, a[2]), inner)
                g = f(first.(inner))
                g === nothing && push!(unsupported, name)
                print(out, something(g, s[i:close]))
                i = close + 1
            else
                print(out, f, "(")
                i = after + 1
            end
        else
            print(out, get(RENAMED, name, name))
            i = after
        end
    end
    String(take!(out)), unique(unsupported)
end

# `integrate(f,x, algorithm="giac")`: the integrand and the variable
function parse_command(cmd::AbstractString)
    m = match(r"^integrate\((.*),\s*([A-Za-z_][A-Za-z_0-9]*)\s*,\s*algorithm=\"giac\"\)$"s, cmd)
    m === nothing && return nothing
    String(m[1]), String(m[2])
end

# ---------------------------------------------------------------------------------------------
# Grading an answer
# ---------------------------------------------------------------------------------------------

"""
    leaf_count(e)

The size of `e` counted as Mathematica's `LeafCount` counts it, so that it compares with the
suite's optimal sizes: every head and atom is a leaf, a fraction is `Rational[p, q]`, a
difference is a sum with a factor −1, a quotient a power −1, a square root the power 1/2, and
`exp(u)` the power `E^u`.
"""
leaf_count(e::GiacExpr) = _leaves(Giac._node(e))

_leaves(v::Integer) = 1
_leaves(v::Rational) = isinteger(v) ? 1 : 3
_leaves(v::Complex) = 3
_leaves(v::Number) = 1
_leaves(n::Giac.NumNode) = _leaves(n.v)
_leaves(n::Giac.SymNode) = 1
_leaves(n::Giac.GenNode) = 1
function _leaves(n::Giac.CallNode)
    n.f == "exp" && return 2 + sum(_leaves, n.args)                    # Power[E, u]
    n.f == "sqrt" && return 4 + sum(_leaves, n.args)                   # Power[u, Rational[1, 2]]
    1 + sum(_leaves, n.args; init = 0)
end
# a term c·m, as Times[c, m] with Times flattened
function _term_leaves(m::Giac.Node, c::Number)
    isone(c) && return _leaves(m)
    m === Giac._UNIT && return _leaves(c)
    m isa Giac.MulNode ? _leaves(m) + _leaves(c) : 1 + _leaves(c) + _leaves(m)
end
_leaves(n::Giac.AddNode) = (length(n.terms) == 1 ? 0 : 1) + sum(_term_leaves(m, c) for (m, c) in n.terms)
_power_leaves(b, e) = isequal(e, Giac._ONE) ? _leaves(b) : 1 + _leaves(b) + _leaves(e)
function _leaves(n::Giac.MulNode)
    parts = sum(_power_leaves(b, e) for (b, e) in n.factors; init = 0)
    k = length(n.factors) + (isone(n.coef) ? 0 : 1)
    k == 1 && isone(n.coef) ? parts : 1 + parts + (isone(n.coef) ? 0 : _leaves(n.coef))
end

const ELEMENTARY = Set(["exp", "ln", "log", "sqrt", "sin", "cos", "tan", "sec", "csc", "cot",
                        "asin", "acos", "atan", "acot", "asec", "acsc", "sinh", "cosh", "tanh",
                        "sech", "csch", "coth", "asinh", "acosh", "atanh", "acoth", "abs", "sign"])
# the special functions written in a Mathematica expression, by their GIAC-like names
const MMA_SPECIAL = r"\b(Erf|Erfi|Erfc|ExpIntegralEi|ExpIntegralE|SinIntegral|CosIntegral|SinhIntegral|CoshIntegral|LogIntegral|PolyLog|Gamma|LogGamma|PolyGamma|Zeta|ProductLog|FresnelS|FresnelC|Elliptic[A-Za-z]*|Hypergeometric[A-Za-z0-9]*|AppellF1|BesselJ|BesselY|BesselI|BesselK|AiryAi|AiryBi|Erfc|HurwitzZeta|LerchPhi|Beta|RootSum|Root)\b"

# functions of optimal answers, in Mathematica's form, that GIAC has no form of; an optimal answer
# with one can't be reached in GIAC
const BEYOND_GIAC = r"\b(PolyLog|FresnelS|FresnelC|Hypergeometric\w*|AppellF1|Elliptic\w*|ExpIntegralE|SinhIntegral|CoshIntegral|LerchPhi|HurwitzZeta)\["

_functions(n::Giac.CallNode, acc) = (push!(acc, n.f); foreach(a -> _functions(a, acc), n.args); acc)
_functions(n::Giac.AddNode, acc) = (foreach(t -> _functions(t[1], acc), n.terms); acc)
_functions(n::Giac.MulNode, acc) = (foreach(f -> (_functions(f[1], acc); _functions(f[2], acc)), n.factors); acc)
_functions(n::Giac.Node, acc) = acc
special_functions(e::GiacExpr) = sort!(collect(setdiff(_functions(Giac._node(e), Set{String}()), ELEMENTARY)))

_has_i(n::Giac.NumNode) = n.v isa Complex
_has_i(n::Giac.SymNode) = false
_has_i(n::Giac.CallNode) = any(_has_i, n.args)
_has_i(n::Giac.AddNode) = any(t -> _has_i(t[1]) || t[2] isa Complex, n.terms)
_has_i(n::Giac.MulNode) = n.coef isa Complex || any(f -> _has_i(f[1]) || _has_i(f[2]), n.factors)
_has_i(n::Giac.Node) = occursin(r"\bi\b", string(n))
has_complex(e::GiacExpr) = _has_i(Giac._node(e))

# values for the parameters, distinct and away from special points
_parameter_value(name) = 0.5 + mod(hash(name) % 1000, 997) / 664.0
const POINTS = [0.37, 0.83, 1.61, 2.29]

# |d/dx answer − integrand| relative to the integrand at the points where it is finite, in
# 20-digit arithmetic: `true` when it is small at every such point (two at least), `false` when
# it is not, `nothing` when there are too few points to tell
function verify(answer::GiacExpr, integrand::GiacExpr, x::GiacExpr)
    d = C.diff(answer, x) - integrand
    names = setdiff(string.(Giac.to_julia(C.lname(integrand + answer))), [string(x)])
    params = Dict(Giac.giac_identifier(n) => _parameter_value(n) for n in names)
    checked = 0
    for p in POINTS
        values = merge(params, Dict(x => p))
        f = _number(Giac.substitute(integrand, values))
        (f === nothing || !isfinite(f) || abs(f) < 1e-12) && continue
        g = _number(Giac.substitute(d, values))
        g === nothing && continue
        abs(g) <= 1e-8 * max(abs(f), 1.0) || return false
        checked += 1
    end
    checked >= 2 ? true : nothing
end
function _number(e)
    try
        v = Giac.to_julia(C.evalf(e, 20))
        v isa Number ? ComplexF64(v) : nothing
    catch
        nothing
    end
end

"""
    grade(answer, integrand, x, optimal_size, optimal_special, optimal_complex) -> (grade, note)

The suite's grade of `answer`: A, correct and at most twice the optimal size; B, correct and
larger; C, correct but with complex numbers or special functions the optimal answer does not
have (`optimal_special`, `optimal_complex` say whether it has them); F, unevaluated or wrong. `?`
is an answer that could not be checked.
"""
function grade(answer::GiacExpr, integrand::GiacExpr, x::GiacExpr, optimal_size, optimal_special, optimal_complex)
    occursin(r"\bintegrate\(", string(answer)) && return ("F", "unevaluated")
    ok = verify(answer, integrand, x)
    ok === nothing && return ("?", "not checked: too few finite points")
    ok || return ("F", "wrong: the derivative differs from the integrand")
    special = special_functions(answer)
    !isempty(special) && !optimal_special && return ("C", "special functions the optimal has not: " * join(special, ", "))
    has_complex(answer) && !optimal_complex && !has_complex(integrand) && return ("C", "complex numbers the optimal has not")
    (optimal_size === nothing || optimal_size <= 0) && return ("A", "")
    size = leaf_count(answer)
    size <= 2optimal_size ? ("A", "") : ("B", "size $size, over twice the optimal $optimal_size")
end

# ---------------------------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------------------------

_time(s) = (v = tryparse(Float64, something(s, "")); v === nothing || v <= 0 ? nothing : v)
_text(s) = s === missing || s === nothing ? nothing : String(s)
_int(s) = s === missing || s === nothing ? nothing : Int(s)
_name(part) = replace(replace(part, r"^[0-9][0-9.]*[a-z]?_" => ""), "_" => " ")

# the optimal antiderivative in GIAC's syntax, from its Maple form, or none if it has none
function _optimal_giac(maple)
    (maple === nothing || occursin(r"Unintegrable|\bInt\(|\bint\(", maple)) && return nothing
    first(to_giac(replace(maple, r"\bln\(" => "log(")))
end

"""
    optimal_size(optimal_giac, optimal_mma_size)

The size of an integral's optimal antiderivative by `leaf_count`, so that it compares with an
answer's; the suite's own size (Mathematica's) when its GIAC form can't be read.
"""
function optimal_size(optimal_giac, optimal_mma_size)
    optimal_giac === nothing && return optimal_mma_size
    try
        leaf_count(giac_eval(optimal_giac))
    catch
        optimal_mma_size
    end
end

"A system's 2024 result for an integral: its grade, its time in seconds and its answer's size."
const Reference = @NamedTuple{grade::Union{Nothing, String}, seconds::Union{Nothing, Float64}, size::Union{Nothing, Int}}

"""
An integral of the suite: where it is in Rubi's taxonomy, the integrand in GIAC's syntax (none if
it has a function GIAC lacks, which `unsupported` names) and in Mathematica's, the optimal
antiderivative in GIAC's syntax with Mathematica's size of it, whether it has special functions
or complex numbers, whether it can be written in GIAC (`expressible`: it has no function of
`BEYOND_GIAC`), and the 2024 results of the systems of `SYSTEMS_2024`.
"""
struct Integral
    id::Int
    file::Int
    number::Int
    chapter::String
    section::String
    integrand::Union{Nothing, String}
    variable::Union{Nothing, String}
    unsupported::Vector{String}
    mma::Union{Nothing, String}
    latex::Union{Nothing, String}
    integrand_size::Union{Nothing, Int}
    rubi_steps::Union{Nothing, Int}
    has_optimal::Bool
    optimal_giac::Union{Nothing, String}
    optimal_mma_size::Union{Nothing, Int}
    optimal_special::Bool
    optimal_complex::Bool
    expressible::Bool
    reference::Vector{Reference}
end

"An integral's 2024 result from `system`, one of `SYSTEMS_2024`."
reference(e::Integral, system) = e.reference[findfirst(==(system), SYSTEMS_2024)]

function Integral(d::AbstractDict)
    Integral((d[string(f)] for f in fieldnames(Integral)[1:end-1])...,
             [Reference((r["grade"], r["seconds"], r["size"])) for r in d["reference"]])
end

"""
    build_catalogue(rows) -> Vector{Integral}

The catalogue of the suite's database rows (anything with its columns as properties, such as an
SQLite query's result), written to `catalogue_file()` one integral a line.
"""
function build_catalogue(rows)
    entries = Integral[]
    for r in rows
        command = parse_command(r.command_in_giac)
        integrand, variable = command === nothing ? (nothing, nothing) : command
        text, unsupported = integrand === nothing ? (nothing, String[]) : to_giac(integrand)
        # Rubi's chapters, and the extra tests
        parts = split(replace(r.output_folder, r"^test_cases/(rubi_tests/)?" => ""), '/')
        optimal_maple, optimal_mma = _text(r.optimal_in_maple), _text(r.optimal_in_mma)
        # Rubi's `Defer[Int]` (or an older `Unintegrable`) is an integral with no known closed form
        known = optimal_mma !== nothing && r.has_known_anti == 1 && !occursin(r"Unintegrable|\bInt\[|Defer\[Int\]", optimal_mma)
        push!(entries, Integral(
            Int(r.global_integral_number), Int(r.input_file_number), Int(r.integral_number_in_file),
            uppercasefirst(_name(parts[1])), length(parts) > 1 ? _name(parts[2]) : "",
            isempty(unsupported) ? text : nothing, variable, unsupported,
            _text(r.command_in_mma), _text(r.integral_in_latex), _int(r.integrand_size), _int(r.rubi_number_of_steps),
            known, known ? _optimal_giac(optimal_maple) : nothing, known ? _int(r.optimal_leafsize) : nothing,
            # an unknown optimal is taken to have whatever an answer has
            !known || occursin(MMA_SPECIAL, optimal_mma), !known || occursin(r"\bI\b", optimal_mma),
            known && !occursin(BEYOND_GIAC, optimal_mma),
            [Reference((_text(getproperty(r, Symbol(s, "_grade"))), _time(_text(getproperty(r, Symbol(s, "_time")))),
                        _int(getproperty(r, Symbol(s, "_leafsize"))))) for s in SYSTEMS_2024]))
    end
    mkpath(DATA)
    open(catalogue_file(), "w") do io
        foreach(e -> println(io, JSON.json(e)), entries)
    end
    entries
end

load_catalogue() = [Integral(JSON.parse(l)) for l in eachline(catalogue_file())]

"""
    details(query, ids)

What the catalogue leaves in the suite's database, for the integrals `ids`: the optimal answer
(in Mathematica's form and typeset), and each 2024 system's answer. `query` runs SQL on the
database and returns its rows.
"""
function details(query, ids)
    isempty(ids) && return Dict{Int, Dict{String, Any}}()
    columns = join(["optimal_in_mma", "optimal_anti_in_latex", (s * "_anti" for s in SYSTEMS_2024)...], ", ")
    rows = query("select global_integral_number, $columns from main where global_integral_number in ($(join(ids, ",")))")
    Dict(Int(r.global_integral_number) => Dict{String, Any}(
             "optimal" => _text(r.optimal_in_mma), "optimal_latex" => _text(r.optimal_anti_in_latex),
             (s => _text(getproperty(r, Symbol(s, "_anti"))) for s in SYSTEMS_2024)...) for r in rows)
end

# ---------------------------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------------------------

"""
    sample(catalogue; per_file = 8)

A fixed sample: every integral of the independent test suites, and from each other test file
`per_file` integrals evenly spaced through it (all of a shorter file).
"""
function sample(catalogue; per_file = 8)
    files = Dict{Int, Vector{Integral}}()
    foreach(e -> push!(get!(files, e.file, Integral[]), e), catalogue)
    chosen = Integral[]
    for f in sort!(collect(keys(files)))
        es = sort!(files[f]; by = e -> e.number)
        if es[1].chapter == "Independent test suites" || length(es) <= per_file
            append!(chosen, es)
        else
            append!(chosen, es[unique(round.(Int, range(1, length(es); length = per_file)))])
        end
    end
    chosen
end

# ---------------------------------------------------------------------------------------------
# Running: integrals in worker processes, each with a time limit
# ---------------------------------------------------------------------------------------------

const WORKER = joinpath(@__DIR__, "worker.jl")

# the versions and the machine a run is made on
function environment()
    sh(cmd) = try readchomp(cmd) catch; "" end
    dir = pkgdir(Giac)
    commit = sh(`git -C $dir rev-parse --short HEAD`) *
             (isempty(sh(`git -C $dir status --porcelain -- src ext`)) ? "" : " (modified)")
    Dict("giac_jl" => string(pkgversion(Giac)), "commit" => commit,
         "branch" => sh(`git -C $dir rev-parse --abbrev-ref HEAD`),
         "giac" => Giac.GiacCxxBindings.giac_version(), "julia" => string(VERSION),
         "cpu" => Sys.cpu_info()[1].model, "machine" => sh(`sysctl -n hw.model`),
         "os" => "macOS " * sh(`sw_vers -productVersion`))
end

# a worker's environment, with the commit of its wrapper library when that is a local build
function _with_wrapper_commit(env)
    dir = dirname(expanduser(env["wrapper"]))
    commit = try readchomp(`git -C $dir rev-parse --short HEAD`) catch; "" end
    isempty(commit) ? env : merge(env, Dict("wrapper_commit" => commit))
end

mutable struct _Worker
    process::Base.Process
    lines::Channel{String}
    environment::Dict{String, Any}     # what it runs: Giac.jl, GIAC and its wrapper, Julia
end

function _start_worker(project, log)
    cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no -t 1 $WORKER`
    p = open(pipeline(cmd; stderr = log), "r+")
    lines = Channel{String}(64)
    @async (try
        for l in eachline(p)
            startswith(l, "@@") && put!(lines, l[3:end])
        end
    catch
    end; close(lines))
    w = _Worker(p, lines, Dict{String, Any}())
    ready = _next(w, 600)
    ready isa String && startswith(ready, "READY ") || error("a worker did not start")
    w.environment = JSON.parse(ready[7:end]; dicttype = Dict{String, Any})
    w
end

# the worker's next message within `limit` seconds, or `:timeout`, or `:exited`
function _next(w::_Worker, limit)
    timedwait(() -> isready(w.lines) || !isopen(w.lines), limit; pollint = 0.01) === :ok || return :timeout
    isready(w.lines) ? take!(w.lines) : :exited
end
# a worker that has answered ends when its input does; one that is late is killed
function _stop(w::_Worker; late = false)
    late ? kill(w.process) : close(w.process.in)
    try close(w.process) catch end
end

function _attempt(w, e::Integral, limit, check_limit)
    println(w.process, JSON.json(e))
    flush(w.process)
    r = _next(w, limit)
    r === :timeout && return (Dict("grade" => "F(-1)", "note" => "no answer within $limit s"), true)
    r === :exited && return (Dict("grade" => "F(-2)", "note" => "the worker stopped"), true)
    result = JSON.parse(r[3:end])        # "R {...}"
    haskey(result, "error") && return (merge(result, Dict("grade" => "F(-2)", "note" => result["error"])), false)
    v = _next(w, check_limit)
    v isa Symbol && return (merge(result, Dict("grade" => "?", "note" => "not checked within $check_limit s")), true)
    (merge(result, JSON.parse(v[3:end])), false)
end

"""
    run!(name, entries; limit = 30, check_limit = 60, workers = …, progress)

Integrate `entries` with Giac.jl, `workers` at a time in processes of their own, and keep each
result as a line of `runs/<name>/results.jsonl`. An integral without an answer within `limit`
seconds has its worker stopped and replaced. Integrals already in the file are skipped, so a run
that is stopped can be continued.
"""
function run!(name, entries; limit = 30, check_limit = 60, workers = max(1, Sys.CPU_THREADS ÷ 2),
              project = Base.active_project(), selection = "", progress = (done, total) -> nothing)
    dir = joinpath(runs_dir(), name)
    mkpath(dir)
    file = joinpath(dir, "results.jsonl")
    done = Set(r.id for r in load_results(name))
    todo = [e for e in entries if !(e.id in done)]
    meta_file = joinpath(dir, "meta.json")
    meta = isfile(meta_file) ? JSON.parsefile(meta_file; dicttype = Dict{String, Any}) :
           Dict{String, Any}("name" => name, "selection" => selection, "started" => string(now()),
                             "limit" => limit, "check_limit" => check_limit, "workers" => workers,
                             "environment" => environment())
    meta["count"] = length(entries)
    write(meta_file, JSON.json(meta))
    out, lock_ = open(file, "a"), ReentrantLock()
    queue = Channel{Integral}(length(todo))
    foreach(e -> put!(queue, e), todo); close(queue)
    finished = Threads.Atomic{Int}(length(done))
    keep(record) = lock(lock_) do
        println(out, JSON.json(record)); flush(out)
        finished[] += 1
        progress(finished[], length(entries))
    end
    log = open(joinpath(dir, "workers.log"), "a")
    try
        @sync for _ in 1:workers
            @async begin
                w = nothing
                for e in queue
                    record = Dict{String, Any}("id" => e.id)
                    if e.integrand === nothing
                        keep(merge(record, Dict("grade" => "N/A", "note" => "not in GIAC: " * join(e.unsupported, ", "))))
                        continue
                    end
                    w === nothing && (w = _start_worker(project, log))
                    result, lost = _attempt(w, e, limit, check_limit)
                    haskey(meta, "worker") || (meta["worker"] = _with_wrapper_commit(w.environment); write(meta_file, JSON.json(meta)))
                    lost && (_stop(w; late = true); w = nothing)
                    keep(merge(record, result))
                end
                w === nothing || _stop(w)
            end
        end
    finally
        close(out)
        close(log)
        meta["finished"] = string(now())
        write(meta_file, JSON.json(meta))
    end
    name
end

"Giac.jl's result for an integral: its grade, a note on it, the time, the answer and its size."
struct Result
    id::Int
    grade::String
    note::String
    seconds::Union{Nothing, Float64}
    answer::Union{Nothing, String}
    size::Union{Nothing, Int}
end
Result(d::AbstractDict) = Result(d["id"], d["grade"], get(d, "note", ""), get(d, "seconds", nothing),
                                 get(d, "answer", nothing), get(d, "size", nothing))

"A run's results."
function load_results(name)
    file = joinpath(runs_dir(), name, "results.jsonl")
    isfile(file) || return Result[]
    [Result(JSON.parse(l)) for l in eachline(file) if !isempty(strip(l))]
end

"The runs kept, newest first, each as its `meta.json`."
function runs()
    isdir(runs_dir()) || return Dict{String, Any}[]
    metas = [JSON.parsefile(joinpath(runs_dir(), d, "meta.json"); dicttype = Dict{String, Any})
             for d in readdir(runs_dir()) if isfile(joinpath(runs_dir(), d, "meta.json"))]
    sort!(metas; by = m -> m["started"], rev = true)
end

end
