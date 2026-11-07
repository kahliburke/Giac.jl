# The independent CAS integration test suite (Abbasi's, built on Rubi's test files): a catalogue of
# its integrals and the 2024 results of nine systems, the units a sweep integrates them in, and the
# runs kept. The grading, which the sweep's processes load, is in grading.jl.
#
# The catalogue is made once from the suite's SQLite database; a run's results are kept, one line
# of JSON per integral, under `~/.cache/giac/integration_suite/runs/<name>/`.

module IntegrationSuite

using Giac, JSON, Dates

include("grading.jl")
using .IntegrationGrading

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

# the special functions written in a Mathematica expression, by their GIAC-like names
const MMA_SPECIAL = r"\b(Erf|Erfi|Erfc|ExpIntegralEi|ExpIntegralE|SinIntegral|CosIntegral|SinhIntegral|CoshIntegral|LogIntegral|PolyLog|Gamma|LogGamma|PolyGamma|Zeta|ProductLog|FresnelS|FresnelC|Elliptic[A-Za-z]*|Hypergeometric[A-Za-z0-9]*|AppellF1|BesselJ|BesselY|BesselI|BesselK|AiryAi|AiryBi|Erfc|HurwitzZeta|LerchPhi|Beta|RootSum|Root)\b"

# functions of optimal answers, in Mathematica's form, that GIAC has no form of; an optimal answer
# with one can't be reached in GIAC
const BEYOND_GIAC = r"\b(PolyLog|FresnelS|FresnelC|Hypergeometric\w*|AppellF1|Elliptic\w*|ExpIntegralE|SinhIntegral|CoshIntegral|LerchPhi|HurwitzZeta)\["

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
# Runs: an integral as a sweep's unit, the code that integrates it, and the results kept
# ---------------------------------------------------------------------------------------------

"""
    unit(e::Integral)

What a sweep's unit needs of an integral: its integrand and variable, and what is known of its
optimal antiderivative ("" and 0 when nothing is).
"""
unit(e::Integral) = (id = e.id, integrand = e.integrand, variable = e.variable,
                     optimal_giac = something(e.optimal_giac, ""), optimal_mma_size = something(e.optimal_mma_size, 0),
                     optimal_special = e.optimal_special, optimal_complex = e.optimal_complex)

_sh(cmd) = try readchomp(cmd) catch; "" end
_short_hash(s) = string(hash(s); base = 16, pad = 16)[1:8]

# the last commit of a git checkout that changed `paths`, with a hash of their uncommitted changes
# after a +: commits that change nothing under them leave it as it was
function _checkout_version(dir, paths...)
    top = _sh(`git -C $dir rev-parse --show-toplevel`)
    isempty(top) && return ""
    changes = _sh(`git -C $top diff HEAD -- $paths`)
    _sh(`git -C $top log -1 --format=%h -- $paths`) * (isempty(changes) ? "" : "+" * _short_hash(changes))
end

"""
    code_version()

What integrates and grades the integrals: the last commit that changed Giac.jl's source, the
C++ wrapper library's (a local build's, or the released version), and a hash of the grading
code, each with a hash of uncommitted changes after a +. A sweep that captures it starts again
when any of them changes, and only then.
"""
function code_version()
    wrapper_dir = dirname(Giac.libgiac_julia_jll.libgiac_wrapper_path)
    local_build = _checkout_version(wrapper_dir, "src")
    (giac_jl = _checkout_version(pkgdir(Giac), "src", "ext"),
     wrapper = isempty(local_build) ? "released " * Giac.GiacCxxBindings.wrapper_version() : local_build,
     wrapper_path = Giac.libgiac_julia_jll.libgiac_wrapper_path,
     grading = _short_hash(read(joinpath(@__DIR__, "grading.jl"), String)),
     giac = Giac.GiacCxxBindings.giac_version())
end

# the machine a run is made on
environment() = Dict("julia" => string(VERSION), "cpu" => Sys.cpu_info()[1].model,
                     "machine" => _sh(`sysctl -n hw.model`), "os" => "macOS " * _sh(`sw_vers -productVersion`))

"""
    keep_run!(name, results; selection, version, limit, started, finished)

Keep a finished sweep's results, `Result`s, as the run `name` under `runs_dir()`, with what made
them, so that runs of different code can be compared. A run already kept is left as it is.
"""
function keep_run!(name, results; selection, version, limit, started, finished)
    dir = joinpath(runs_dir(), name)
    isfile(joinpath(dir, "meta.json")) && return name
    mkpath(dir)
    open(joinpath(dir, "results.jsonl"), "w") do io
        foreach(r -> println(io, JSON.json(r)), results)
    end
    write(joinpath(dir, "meta.json"), JSON.json(Dict(
        "name" => name, "selection" => selection, "count" => length(results), "limit" => limit,
        "version" => Dict(pairs(version)), "environment" => environment(),
        "started" => string(started), "finished" => string(finished))))
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
# from a sweep's row: the integral's id among its parameters, and what the unit returned
# (a unit that has no answer stores no `answer` or `size`)
Result(id::Integer, r) = Result(id, r.grade, r.note, r.seconds, coalesce(get(r, :answer, nothing), nothing),
                                coalesce(get(r, :size, nothing), nothing))

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
