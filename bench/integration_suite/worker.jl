# A worker of the integration suite's runner: integrals in on stdin, one JSON object a line, and for
# each, two lines out, marked `@@` to tell them from anything GIAC prints: `@@R` with the answer and
# its time, then `@@V` with its grade. The runner stops the process when an answer is late.

include(joinpath(@__DIR__, "IntegrationSuite.jl"))
using .IntegrationSuite: Giac, JSON, grade, leaf_count, optimal_size
const C = Giac.Commands

say(tag, d) = (println(stdout, "@@", tag, " ", JSON.json(d)); flush(stdout))

# integrals of each kind first, so that no timed one includes compiling the code they share
let x = Giac.giac_identifier("x")
    for f in ("x^2*sin(x)", "1/(x^2+1)", "sqrt(1+x^2)", "exp(x)*cos(x)", "log(x)/x", "(a+b*x)^m")
        g = Giac.giac_eval(f)
        grade(C.integrate(g, x), g, x, 10, false, false)
    end
end
# what this process runs, for the run's record
say("READY", Dict("giac_jl" => string(pkgversion(Giac)), "giac" => Giac.GiacCxxBindings.giac_version(),
                  "wrapper" => replace(Giac.libgiac_julia_jll.libgiac_wrapper_path, homedir() => "~"),
                  "julia" => string(VERSION), "threads" => Threads.nthreads()))

for line in eachline(stdin)
    task = JSON.parse(line)
    integrand, x = try
        Giac.giac_eval(task["integrand"]), Giac.giac_identifier(task["variable"])
    catch e
        say("R", Dict("seconds" => 0.0, "error" => "the integrand: " * first(sprint(showerror, e), 300)))
        continue
    end
    answer = nothing
    seconds = @elapsed answer = try
        C.integrate(integrand, x)
    catch e
        e
    end
    if answer isa Exception
        say("R", Dict("seconds" => seconds, "error" => first(sprint(showerror, answer), 300)))
        continue
    end
    say("R", Dict("seconds" => seconds, "answer" => string(answer)))
    g, note = try
        grade(answer, integrand, x, optimal_size(task["optimal_giac"], task["optimal_mma_size"]),
              task["optimal_special"], task["optimal_complex"])
    catch e
        ("?", "not checked: " * first(sprint(showerror, e), 200))
    end
    size = try leaf_count(answer) catch; nothing end
    say("V", Dict("grade" => g, "note" => note, "size" => size))
end
