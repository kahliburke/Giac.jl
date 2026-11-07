try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Giac.jl on the independent CAS integration tests

Nasser Abbasi's [CAS integration tests](https://www.12000.org/my_notes/CAS_integration_tests/) run Rubi's test suite, 106,812 indefinite integrals from textbooks, papers and Rubi's own rule tests, through nine computer algebra systems, and grade each answer against an optimal antiderivative. This notebook runs Giac.jl on them and compares its grades with the nine systems' results from the suite's latest full run, Summer 2024, which used GIAC 1.9.0-99.

The suite's results come as one SQLite database. From it the notebook makes a catalogue: each integral in GIAC's syntax, its optimal antiderivative and that answer's size, its place in Rubi's taxonomy, and the 2024 grade, time and answer size of each system. A run integrates a selection of the catalogue with Giac.jl and keeps the results; the analysis reads the runs kept. Nothing is recomputed when the notebook opens: the run cell is locked, and runs only when run deliberately.

**Grades**, as the suite gives them: **A**, a correct answer at most twice the optimal's size; **B**, correct but larger; **C**, correct but with complex numbers or special functions the optimal answer does not need; **F**, unevaluated or wrong; **F(-1)**, no answer within the time limit; **F(-2)**, an error. **N/A** is an integrand GIAC can't state, one with a function it doesn't have (polylogarithms, Fresnel and generalized exponential integrals, the Hurwitz zeta function). Giac.jl's answers are graded here, in Julia: an answer is correct when its derivative matches the integrand, in 20-digit arithmetic, at points where the integrand is finite, with the parameters given fixed values; sizes are counted the way Mathematica's `LeafCount` counts them, for the answer and the optimal alike. **?** is an answer that could not be checked that way.
"""

#%% code id=setup eval_barrier
using Giac, JSON, SQLite, CairoMakie, Dates
using Markdown
include(joinpath(pkgdir(Giac), "bench", "integration_suite", "IntegrationSuite.jl"))
const S = IntegrationSuite
set_theme!(theme_dark())
(; data = S.DATA, giac = Giac.GiacCxxBindings.giac_version())

#%% md id=data_md
@md"""
## The catalogue

Downloaded and made once: the database (1.5 GB) is kept in `~/.cache/giac/integration_suite/download/`, the catalogue made from it beside it.
"""

#%% code id=catalogue
suite_db = let file = joinpath(S.DATA, "download", "cas_integration_tests.db")
    if !isfile(file)
        mkpath(dirname(file))
        zip = joinpath(dirname(file), "cas_integration_tests.zip")
        download(S.DATABASE_URL, zip)
        run(`unzip -o -q $zip -d $(dirname(file))`)
        rm(zip)
    end
    SQLite.DB(file)
end
query(sql) = SQLite.DBInterface.execute(suite_db, sql)
catalogue = isfile(S.catalogue_file()) ? S.load_catalogue() :
            S.build_catalogue(query("select * from main order by global_integral_number"))
by_id = Dict(e.id => e for e in catalogue)
chapters = unique(e.chapter for e in catalogue)
(integrals = length(catalogue), chapters = length(chapters), files = length(unique(e.file for e in catalogue)),
 not_in_giac = count(e -> e.integrand === nothing, catalogue))

#%% code id=chapters_table
# `k` of `n` as a percentage, and whether a grade is of a correct answer
percent(k, n) = n == 0 ? "—" : "$(round(Int, 100k / n))%"
solved(g) = g in ("A", "B", "C")
# each chapter's integrals with no known closed form, with an optimal answer GIAC has no functions
# for, and with one it can write; and the share each system solved in 2024
let rows = [(c, [e for e in catalogue if e.chapter == c]) for c in chapters]
    Markdown.parse("| Chapter | Integrals | No closed form | Beyond GIAC | Expressible | " * join([S.SYSTEM_NAMES[s] for s in S.SYSTEMS_2024], " | ") * " |\n" *
                   "|:--|--:|--:|--:|--:|" * repeat("--:|", length(S.SYSTEMS_2024)) * "\n" *
                   join(["| $c | $(length(es)) | $(percent(count(e -> !e.has_optimal, es), length(es))) | " *
                         "$(percent(count(e -> e.has_optimal && !e.expressible, es), length(es))) | $(percent(count(e -> e.expressible, es), length(es))) | " *
                         join([percent(count(e -> solved(S.reference(e, s).grade), es), length(es)) for s in S.SYSTEMS_2024], " | ") * " |"
                         for (c, es) in rows], "\n"))
end

#%% md id=selections_md
@md"""
## Selections

A run integrates a selection of the catalogue. **sample** is fixed: every integral of the independent test suites (textbook and paper problems), and eight from each of Rubi's other test files, evenly spaced through each. The others pick out kinds of integral.
"""

#%% code id=selections
selections = Dict(
    "sample" => S.sample(catalogue),
    "independent" => [e for e in catalogue if e.chapter == "Independent test suites"],
    "GIAC 1.9 failed, Rubi solved" => [e for e in catalogue if startswith(something(S.reference(e, "giac").grade, "F"), "F") &&
                                                               S.reference(e, "rubi").grade in ("A", "B")],
    "all" => catalogue)
Dict(k => length(v) for (k, v) in selections)

#%% md id=run_md
@md"""
## A run

Locked: it runs only when run deliberately. The integrals are shared among several worker processes, each a Julia with Giac.jl that integrates one integral after another, as a session would, after a few integrals of each kind that compile the code they share, so that no timed integral includes compiling it. An integral without an answer in 30 seconds has its worker stopped and replaced: GIAC's own timeout leaves the process unusable once it fires. Each result is kept as it comes, so a run that is stopped continues where it left off when run again under the same name.
"""

#%% code id=units
# the selection to integrate, as the sweep's units: the integrals GIAC can state (the others are
# N/A without a run), and what integrates them
run_selection = "sample"
units = [S.unit(e) for e in selections[run_selection] if e.integrand !== nothing]
version = S.code_version()
(selection = run_selection, units = length(units), version)

#%% job id=integration cluster=local eval_barrier
# Each integral is a unit: integrated by Giac.jl with a 30 s limit, then graded. A process takes a
# chunk of units in turn, as one session would. `version` names the code: a change to Giac.jl,
# its wrapper library or the grading starts a new sweep. GIAC runs on a thread of its own
# (DedicatedThreads), which lets `with_timeout` stop an integration that runs over.
integration = @sweep(paramgrid(units); plot = false) do p
    using DedicatedThreads, Giac
    isdefined(Main, :IntegrationGrading) ||
        Base.include(Main, joinpath(pkgdir(Giac), "bench", "integration_suite", "grading.jl"))
    Giac.libgiac_julia_jll.libgiac_wrapper_path == version.wrapper_path ||
        error("the task process loads another wrapper library: ", Giac.libgiac_julia_jll.libgiac_wrapper_path)
    # the grading code was loaded during this call, so it is reached in the newest world
    Base.invokelatest(() -> Main.IntegrationGrading.integrate_unit(p; limit = 30))
end

#%% md id=results_md
@md"""
## Results

The sweep's results so far, against the 2024 results on the same integrals; the cells run again as units land. The 2024 runs were made on another machine, and GIAC through Sage, so their times compare only roughly.

| Grade | Meaning |
|:--|:--|
| **A** | correct, and at most twice the size of the optimal antiderivative (leaves counted as Mathematica's `LeafCount` counts them) |
| **B** | correct, but more than twice the optimal size |
| **C** | correct, but with complex numbers or special functions the optimal answer does not need |
| **?** | an answer that could not be checked (too few points where the integrand is finite) |
| **F** | not solved: given back unevaluated, or wrong (its derivative is not the integrand) |
| **F(-1)** | no answer within the time limit |
| **F(-2)** | an error |
| **N/A** | an integrand GIAC can't state: it has a function GIAC lacks |

*Solved* in the tables means A, B or C. The 2024 grader gave MuPAD and Reduce at most B.
"""

#%% code id=results
# Giac.jl's results so far: the sweep's units that have landed, and the selection's integrals GIAC
# can't state. The cells below run again as units land.
landed = [S.Result(row.params.id, row.record) for row in integration if row.record !== nothing]
not_stated = [S.Result(e.id, "N/A", "not in GIAC: " * join(e.unsupported, ", "), nothing, nothing, nothing)
              for e in selections[run_selection] if e.integrand === nothing]
results = [landed; not_stated]
graded = [(e = by_id[r.id], r = r) for r in results]
GRADES = ["A", "B", "C", "?", "F", "F(-1)", "F(-2)", "N/A"]
grade_2024(e, s) = something(S.reference(e, s).grade, "N/A")
tally(gs) = Dict(g => count(==(g), gs) for g in GRADES)
ours = tally([x.r.grade for x in graded])
theirs = Dict(s => tally([grade_2024(x.e, s) for x in graded]) for s in S.SYSTEMS_2024)
# what made them, for the plots
machine = S.environment()
made_by = "Giac.jl $(pkgversion(Giac)) at $(version.giac_jl), wrapper $(version.wrapper), GIAC $(version.giac)"
(landed = length(landed), of = length(units), grades = ours)

#%% code id=kept_runs
# Once every unit has landed, the sweep's results are kept as a run, to compare with runs of
# other code; the runs kept so far
run_name = string(Dates.today(), "_", version.giac_jl, "_", replace(run_selection, r"\W+" => "_"))
if integration.total > 0 && integration.pending == 0
    at = [row.at for row in integration if row.record !== nothing]
    S.keep_run!(run_name, results; selection = run_selection, version, limit = 30,
                started = minimum(at), finished = maximum(at))
end
kept = S.runs()
md_text(s) = replace(string(s), "_" => "\\_", "|" => "\\|")
Markdown.parse("| Run | Selection | Integrals | Giac.jl | Wrapper | Grading | GIAC | Finished |\n|:--|:--|--:|:--|:--|:--|:--|:--|\n" *
               join(["| $(md_text(m["name"])) | $(md_text(m["selection"])) | $(m["count"]) | $(m["version"]["giac_jl"]) | " *
                     "$(m["version"]["wrapper"]) | $(m["version"]["grading"]) | $(m["version"]["giac"]) | $(first(m["finished"], 16)) |"
                     for m in kept], "\n"))

#%% code id=grades_plot
let systems = ["Giac.jl"; [S.SYSTEM_NAMES[s] for s in S.SYSTEMS_2024]]
    counts = [ours; [theirs[s] for s in S.SYSTEMS_2024]]
    n = length(graded)
    colours = Dict("A" => :seagreen3, "B" => :yellowgreen, "C" => :goldenrod, "?" => :gray60, "F" => :tomato,
                   "F(-1)" => :orchid, "F(-2)" => :mediumpurple, "N/A" => :gray35)
    meaning = Dict("A" => "A  correct, at most twice the optimal size", "B" => "B  correct, larger",
                   "C" => "C  correct, needless complex numbers or special functions", "?" => "?  not checked",
                   "F" => "F  unevaluated or wrong", "F(-1)" => "F(-1)  over the time limit", "F(-2)" => "F(-2)  an error",
                   "N/A" => "N/A  not expressible in GIAC")
    fig = Figure(size = (1000, 500))
    ax = Axis(fig[1, 1]; xlabel = "share of the $n integrals", yticks = (1:length(systems), systems), yreversed = true,
              title = "grades: $made_by, and the 2024 results",
              subtitle = "$(machine["cpu"]), $(machine["os"]), Julia $(machine["julia"]); selection $run_selection")
    left = zeros(length(systems))
    for g in GRADES
        w = [c[g] / n for c in counts]
        barplot!(ax, 1:length(systems), w; offset = left, direction = :x, color = colours[g], label = meaning[g])
        left .+= w
    end
    Legend(fig[2, 1], ax; orientation = :horizontal, nbanks = 3, framevisible = false, labelsize = 12)
    fig
end

#%% code id=by_chapter
# correct answers by chapter, of all the integrals and of those whose optimal answer GIAC can write
let systems = ["giac", "mma", "rubi", "maple", "fricas", "sympy"]
    rows = filter(r -> !isempty(r[2]), [(c, [x for x in graded if x.e.chapter == c]) for c in chapters])
    rate(xs, s) = (g(x) = s == "ours" ? x.r.grade : grade_2024(x.e, s);
                   ex = filter(x -> x.e.expressible, xs);
                   "$(percent(count(x -> solved(g(x)), xs), length(xs))) · $(percent(count(x -> solved(g(x)), ex), length(ex)))")
    Markdown.parse("Each cell: correct of all · of those whose optimal answer GIAC can express.\n\n" *
                   "| Chapter | Integrals | Expressible | **Giac.jl** | " * join([S.SYSTEM_NAMES[s] for s in systems], " | ") * " |\n" *
                   "|:--|--:|--:|--:|" * repeat("--:|", length(systems)) * "\n" *
                   join(["| $c | $(length(xs)) | $(count(x -> x.e.expressible, xs)) | **$(rate(xs, "ours"))** | " *
                         join([rate(xs, s) for s in systems], " | ") * " |" for (c, xs) in rows], "\n"))
end

#%% md id=transitions_md
@md"""
### From GIAC 1.9 to Giac.jl

Each integral's 2024 GIAC grade against its grade now: the diagonal is unchanged, left of it better, right of it worse.
"""

#%% code id=transitions
let k = length(GRADES)
    M = zeros(Int, k, k)
    for x in graded
        M[findfirst(==(grade_2024(x.e, "giac")), GRADES), something(findfirst(==(x.r.grade), GRADES), k)] += 1
    end
    fig = Figure(size = (640, 560))
    ax = Axis(fig[1, 1]; xlabel = made_by, ylabel = "GIAC 1.9.0-99 (2024)",
              xticks = (1:k, GRADES), yticks = (1:k, GRADES), yreversed = true, aspect = DataAspect())
    heatmap!(ax, 1:k, 1:k, log10.(M' .+ 1); colormap = :viridis)
    top = maximum(log10.(M .+ 1))
    for i in 1:k, j in 1:k
        M[i, j] > 0 && text!(ax, j, i; text = string(M[i, j]), align = (:center, :center), fontsize = 12,
                             color = log10(M[i, j] + 1) > 0.6top ? :black : :white)
    end
    fig
end

#%% code id=times_plot
let times(xs, f) = sort([t for x in xs for t in (f(x),) if t !== nothing])
    ours_t = times([x for x in graded if solved(x.r.grade)], x -> x.r.seconds)
    old = times([x for x in graded if solved(grade_2024(x.e, "giac"))], x -> S.reference(x.e, "giac").seconds)
    mma = times([x for x in graded if solved(grade_2024(x.e, "mma"))], x -> S.reference(x.e, "mma").seconds)
    fig = Figure(size = (900, 420))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "seconds", ylabel = "integrals answered correctly within",
              title = "times of the correct answers", subtitle = "Giac.jl on $(machine["cpu"]); the 2024 times on the suite's machine")
    stairs!(ax, ours_t, 1:length(ours_t); color = :orange, linewidth = 2, label = made_by)
    isempty(old) || stairs!(ax, old, 1:length(old); color = :deepskyblue, label = "GIAC 1.9.0-99 via Sage (2024)")
    isempty(mma) || stairs!(ax, mma, 1:length(mma); color = :gray70, label = "Mathematica 14 (2024)")
    axislegend(ax; position = :lt, framevisible = false)
    fig
end

#%% md id=cases_md
@md"""
### Cases

Integrals worth looking at: answers that are wrong, integrals GIAC 1.9 did and Giac.jl does not, and the smallest integrals Giac.jl misses that both Rubi and Mathematica do. The optimal answers come from the suite's database.
"""

#%% code id=cases
let wrong = [x for x in graded if startswith(x.r.note, "wrong")]
    lost = [x for x in graded if solved(grade_2024(x.e, "giac")) && !(x.r.grade in ("A", "B", "C", "?"))]
    missed = sort([x for x in graded if !(x.r.grade in ("A", "B", "C", "?", "N/A")) &&
                   S.reference(x.e, "rubi").grade in ("A", "B") && S.reference(x.e, "mma").grade in ("A", "B")];
                  by = x -> something(x.e.integrand_size, 0))
    shown = [first(wrong, 10); first(lost, 10); first(missed, 10)]
    more = S.details(query, [x.e.id for x in shown])
    cell(s) = replace(first(something(s, ""), 160), "|" => "\\|")
    row(x) = "| $(x.e.id) | \$$(x.e.latex)\$ | $(grade_2024(x.e, "giac")) → **$(x.r.grade)** | " *
             "$(cell(something(x.r.answer, x.r.note))) | \$$(something(get(get(more, x.e.id, Dict()), "optimal_latex", nothing), ""))\$ |"
    table(title, xs) = "#### $title ($(length(xs)))\n\n| Id | Integral | GIAC 1.9 → Giac.jl | Answer | Optimal |\n|:--|:--|:--|:--|:--|\n" *
                       join(row.(first(xs, 10)), "\n")
    Markdown.parse(join([table("Wrong answers", wrong), table("GIAC 1.9 solved, Giac.jl does not", lost),
                         table("Missed, solved by Rubi and Mathematica, smallest first", missed)], "\n\n"))
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   CairoMakie 0.15.17 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   DedicatedThreads 0.1.0 9d6577c3-9664-4440-a53f-cec765ebc045
#   JSON 1.13.4 682c06a0-de6a-54ab-a142-c8b1cf79cde6
#   SQLite 1.11.0 0aa819cd-b072-5ff4-a722-6bc24af294d9
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 4e2749e3-58a7-438b-97a3-04843237f7b6
#   format = 2
# ╚═╡
