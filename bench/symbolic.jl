# Timings for the symbolic layer built on Giac: arithmetic, reading expressions, typesetting,
# rule rewriting, search and form matching. Each case reports its median time per call.
#
# Results are saved by label under ~/.cache/giac/benchmarks/, and compared with a saved label:
#
#   julia --project bench/symbolic.jl baseline            # save as "baseline"
#   julia --project bench/symbolic.jl tree baseline       # save as "tree", compare with "baseline"
#
# or from a session: include("bench/symbolic.jl"); bench_symbolic("tree"; against = "baseline")

using Giac, TOML, Printf
using Giac.Rules

const BENCH_DIR = joinpath(homedir(), ".cache", "giac", "benchmarks")

function symbolic_cases()
    @giac_var x y a b c s t K ζ ω h k A B
    @giac_var q(t) q̇ q̈
    poly30 = sum(k * a^k * x^k + b * y^k for k in 1:15)
    trig_log = sum(sin(k * x)^2 * a^k + cos(k * x)^2 * a^k + log(x * y^k) for k in 1:10)
    trig_rules = vcat(trig_simplify(), log_expand())
    pyth = @rule sin(~u)^2 + cos(~u)^2 => 1
    G = 5 / (s^2 + 3s + 4)
    second_order = K * ω^2 / (s^2 + 2ζ * ω * s + ω^2)
    return [
        "add" => () -> x + y,
        "add, like terms" => () -> 3x + 2x,
        "multiply" => () -> a * x,
        "sum of 30 terms" => () -> sum(k * a^k * x^k + b * y^k for k in 1:15),
        "arguments, 30 terms" => () -> Giac.arguments(poly30),
        "latex, small" => () -> Giac.giac_latex(c + b * x + a * x^2),
        "latex, 30 terms" => () -> Giac.giac_latex(poly30),
        "latex, dynamics" => () -> Giac.giac_latex(q̈ + 2ζ * ω * q̇ + ω^2 * q),
        "rule, one match" => () -> pyth(sin(x)^2 + cos(x)^2),
        "rewrite, 30 terms" => () -> rewrite(trig_log, trig_rules),
        "rewrite, polynomial" => () -> rewrite(poly30, trig_rules),
        "collect_terms" => () -> collect_terms(poly30, x),
        "simplest, small" => () -> simplest(sin(x)^2 + cos(x)^2 + log(x * y)),
        "simplest, medium" => () -> simplest(exp(log(x) + log(y)) + sin(x)^2 * a + cos(x)^2 * a),
        "in_form" => () -> in_form(G, second_order, (K, ζ, ω)),
    ]
end

# median seconds per call: one warm-up call, then samples for about `seconds`
function time_case(f; seconds = 0.5, min_samples = 5)
    f()
    samples = Float64[]
    start = time()
    while length(samples) < min_samples || time() - start < seconds
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) / 1e9)
    end
    sort!(samples)
    return samples[cld(length(samples), 2)]
end

_format(t) = t < 1e-3 ? @sprintf("%8.1f µs", t * 1e6) : t < 1 ? @sprintf("%8.2f ms", t * 1e3) : @sprintf("%8.2f s ", t)

function bench_symbolic(label = nothing; against = nothing, seconds = 0.5)
    previous = against === nothing ? nothing : TOML.parsefile(joinpath(BENCH_DIR, "$against.toml"))
    results = Dict{String, Float64}()
    for (name, f) in symbolic_cases()
        t = time_case(f; seconds)
        results[name] = t
        line = rpad(name, 24) * _format(t)
        if previous !== nothing && haskey(previous, name)
            line *= @sprintf("   %s %6.2fx", _format(previous[name]), previous[name] / t)
        end
        println(line)
    end
    if label !== nothing
        mkpath(BENCH_DIR)
        open(io -> TOML.print(io, results), joinpath(BENCH_DIR, "$label.toml"), "w")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench_symbolic(get(ARGS, 1, nothing); against = get(ARGS, 2, nothing))
end
