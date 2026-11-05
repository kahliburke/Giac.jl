try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md raw"""
# Simulating Giac models with SciML

A model written as Giac expressions becomes a SciML problem: `ODEProblem(F, u0, tspan, p)` for $\dot x = F(x, p, t)$, and `NonlinearProblem(F, u0, p)` for $F(x, p) = 0$. The extension `GiacSciMLBaseExt` turns the right-hand side and its exact derivatives into Julia code and compiles it, so a solver calls functions as fast as hand-written ones, which allocate nothing:

| The problem's function | Is |
|--|--|
| `f(du, u, p, t)` | the right-hand side $F$ |
| `f.jac(J, u, p, t)` | the exact Jacobian $\partial F / \partial x$, dense or sparse, for implicit solvers |
| `f.tgrad(dT, u, p, t)` | $\partial F / \partial t$, for a model with a time variable |
| `f.paramjac(Jp, u, p, t)` | $\partial F / \partial p$, for sensitivities |

The states and parameters are the variables of the assignments `u0` and `p`, written `[x1 => 1, x2 => 0]` or `x => values`. The problem keeps them, so a solution is indexed by the model's own variables, and by expressions in them: `sol[x1]`, `sol[x1^2 + x2^2]`, `remake(prob; p = [g => 3.7])`.
"""

#%% code id=setup
using Giac, GiacSlate, LinearAlgebra
using SciMLBase, OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, NonlinearSolve, SparseArrays
using CairoMakie, BenchmarkTools
set_theme!(theme_dark())
const C = Giac.Commands
(; giac = pkgdir(Giac), scimlbase = pkgversion(SciMLBase), rosenbrock = pkgversion(OrdinaryDiffEqRosenbrock))

#%% code id=examples_helper collapsed
"Lines of code beside the values they give: rows of `code => value`, or a heading string."
struct Examples
    rows::Vector{Any}
end

# a string followed by a line parses as a docstring for it: split them again
_unwrap_doc(st) = st isa Expr && st.head === :macrocall && st.args[1] == GlobalRef(Core, Symbol("@doc")) ?
                  Any[st.args[3], st.args[4]] : Any[st]

# code as one readable line: no line-number comments, `() -> f(x)` rather than a `begin` block
# (a `do` block keeps its lines)
function _collapse(ex)
    ex isa Expr || return ex
    ex.head === :do && return ex
    ex.head === :macrocall && return Expr(:macrocall, ex.args[1], nothing, map(_collapse, ex.args[3:end])...)
    ex.head === :block && length(ex.args) == 1 && return _collapse(ex.args[1])
    return Expr(ex.head, map(_collapse, ex.args)...)
end
_code(st) = replace(string(_collapse(Base.remove_linenums!(deepcopy(st)))), " ^ " => "^")

# `x = …` or `a, b = …` (not a short function definition `f(x) = …`)
_is_assignment(st) = st isa Expr && st.head === :(=) && !(st.args[1] isa Expr && st.args[1].head in (:call, :where))

"`@examples begin … end`: show each line of code next to its result; a string line is a heading."
macro examples(block)
    rows = gensym(:rows)
    out = Any[:($rows = Any[])]
    for st in mapreduce(_unwrap_doc, vcat, block.args; init = Any[])
        st isa LineNumberNode && continue
        if st isa String
            push!(out, :(push!($rows, $st)))
        elseif _is_assignment(st)
            lhs, rhs = st.args
            value = gensym(:value)
            # the right side runs inside a `try`; the binding (or destructuring) happens outside
            # it, so later lines can use the names
            push!(out, :($value = try $(esc(rhs)) catch err; err end),
                       :($value isa Exception || ($(esc(lhs)) = $value)),
                       :(push!($rows, $(_code(st)) => $value)))
        else
            push!(out, :(push!($rows, $(_code(st)) => try $(esc(st)) catch err; err end)))
        end
    end
    push!(out, :(Examples($rows)))
    return Expr(:block, out...)
end

_esc_html(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
_value_tex(v) = strip(sprint(show, MIME("text/latex"), v), ['$', '\n', ' '])
_has_tex(v) = showable(MIME("text/latex"), v) && !(v isa AbstractString)
# a vector of numbers on one line; anything else as the REPL shows it
_plain(v::AbstractVector{<:Number}) = repr(v; context = :compact => true)
_plain(v) = repr(MIME("text/plain"), v; context = :compact => true)
function _example_value(v)
    v isa Exception && return "<span class=\"ex-err\">" * _esc_html(first(sprint(showerror, v), 160)) * "</span>"
    v isa Tuple && return join(map(_example_value, v), " , ")
    _has_tex(v) && return "\$\\displaystyle " * _value_tex(v) * "\$"
    s = _plain(v)
    occursin('\n', s) && return "<pre class=\"ex-pre\">" * _esc_html(s) * "</pre>"
    return "<code>" * _esc_html(s) * "</code>"
end

function Base.show(io::IO, ::MIME"text/html", ex::Examples)
    print(io, """<style>
      .disp.html .examples-scroll { overflow-x: auto; max-width: 100%; }
      .disp.html table.examples { border-collapse: collapse; margin: 4px 0; }
      .disp.html table.examples td { padding: 9px 16px; vertical-align: middle; border-bottom: 1px solid rgba(128,128,128,.25); }
      .disp.html table.examples td.code { white-space: pre; font-family: var(--mono, monospace); font-size: .92em; opacity: .9; }
      .disp.html table.examples td.head { padding: 14px 16px 4px; font-weight: 600; border-bottom: none; }
      .disp.html table.examples .ex-err { color: #e06c75; font-family: monospace; font-size: .9em; }
      .disp.html table.examples pre.ex-pre { margin: 0; font-size: .9em; }
    </style><div class="examples-scroll"><table class="examples">""")
    for r in ex.rows
        if r isa String
            print(io, "<tr><td class=\"head\" colspan=\"2\">", _esc_html(r), "</td></tr>")
        else
            print(io, "<tr><td class=\"code\">", _esc_html(first(r)), "</td><td>", _example_value(last(r)), "</td></tr>")
        end
    end
    print(io, "</table></div>")
end

_md_cell(s) = replace(s, "|" => "\\|")
_md_code(s) = (t = _md_cell(s); occursin('`', t) ? "`` " * t * " ``" : "`" * t * "`")
function _example_md(v)
    v isa Exception && return _md_code("error: " * first(sprint(showerror, v), 120))
    v isa Tuple && return join(map(_example_md, v), ", ")
    _has_tex(v) && return "\$\$" * replace(_value_tex(v), "|" => "\\vert ") * "\$\$"
    return _md_code(join(strip.(split(_plain(v), '\n')), "; "))
end
function Base.show(io::IO, ::MIME"text/markdown", ex::Examples)
    println(io, "| code | result |\n|:--|:--|")
    for r in ex.rows
        r isa String ? println(io, "| **", _md_cell(r), "** | |") :
                       println(io, "| ", _md_code(first(r)), " | ", _example_md(last(r)), " |")
    end
end

#%% md id=pendulum_md
@md raw"""
## A damped pendulum

$\dot x_1 = x_2, \quad \dot x_2 = -\tfrac{g}{\ell}\sin x_1 - c\,x_2$. The model is a vector of expressions; the problem takes its states and parameters from the assignments. Its Jacobian is Giac's, compiled, and an implicit solver (`Rodas5P`) uses it in place of automatic differentiation. The energy $E = \tfrac12 x_2^2 + \tfrac{g}{\ell}(1 - \cos x_1)$ is an expression in the states and parameters, so the solution evaluates it directly.
"""

#%% code id=pendulum
@giac_var x[1:2] g ℓ c
F = [x2, -g / ℓ * sin(x1) - c * x2]
E = x2^2 / 2 + g / ℓ * (1 - cos(x1))
J = zeros(2, 2)
@examples begin
    "the problem"
    prob = ODEProblem(F, [x1 => π / 2, x2 => 0], (0, 20), [g => 9.81, ℓ => 1, c => 0.25])
    prob.f.sys
    prob.u0, prob.p
    "the exact Jacobian, and its compiled form"
    jacobian(F, x)
    (prob.f.jac(J, [0.1, 0.0], prob.p, 0.0); J)
    "the solution, indexed by the model's variables"
    sol = solve(prob, Rodas5P(); abstol = 1e-9, reltol = 1e-9)
    sol.stats.njacs
    last(sol[x1])
    sol(2.5; idxs = [x1, x2])
    sol(0.0; idxs = E), sol(20.0; idxs = E)
    "parameters and values by name"
    prob.ps[g]
    moon = remake(prob; p = [g => 1.62])
    moon.p
    solve(moon, Rodas5P())(20.0; idxs = x1)
end

#%% code id=portrait
let
    fig = Figure(size = (1000, 420))
    ax = Axis(fig[1, 1]; xlabel = "x₁", ylabel = "x₂", title = "trajectories, coloured by energy")
    starts = [(a, b) for a in range(-2π, 2π; length = 9) for b in (-6.0, -3.0, 3.0, 6.0)]
    for (a, b) in starts
        s = solve(remake(prob; u0 = [x1 => a, x2 => b], tspan = (0.0, 12.0)), Rodas5P())
        ts = range(0, 12; length = 600)
        lines!(ax, s(ts; idxs = x1).u, s(ts; idxs = x2).u; color = s(ts; idxs = E).u,
               colormap = :plasma, colorrange = (0, 40), linewidth = 1.2)
    end
    xlims!(ax, -2π, 2π); ylims!(ax, -7, 7)
    Colorbar(fig[1, 2]; colormap = :plasma, limits = (0, 40), label = "E")
    ax2 = Axis(fig[1, 3]; xlabel = "t", ylabel = "E", title = "energy along sol, from x = (π/2, 0)")
    ts = range(0, 20; length = 800)
    lines!(ax2, ts, sol(ts; idxs = E).u; color = :orange)
    fig
end

#%% md id=forced_md
@md raw"""
## Forcing in time

With a time variable, given as `t = …`, the function also carries the exact $\partial F / \partial t$, which Rosenbrock methods use. The forced Duffing oscillator $\ddot y + \delta \dot y - y + y^3 = \gamma \cos \omega t$ is chaotic at these values. Its Poincaré section, the state once every forcing period, is the strange attractor; `sol(ts; idxs = [y1, y2])` reads it off the dense solution.
"""

#%% code id=forced
@giac_var y[1:2] δ γ ω t
duffing = [y2, -δ * y2 + y1 - y1^3 + γ * cos(ω * t)]
T = 2π / 1.2
dprob = ODEProblem(duffing, [y1 => 1, y2 => 0], (0, 3000T), [δ => 0.3, γ => 0.5, ω => 1.2]; t)
dsol = solve(dprob, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
let
    fig = Figure(size = (1000, 420))
    ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "y₁", title = "the first 30 periods")
    ts = range(0, 30T; length = 4000)
    lines!(ax, ts, dsol(ts; idxs = y1).u; color = :deepskyblue, linewidth = 1)
    pts = dsol((100:3000) .* T; idxs = [y1, y2]).u
    ax2 = Axis(fig[1, 2]; xlabel = "y₁", ylabel = "y₂", title = "Poincaré section, $(length(pts)) periods")
    scatter!(ax2, first.(pts), last.(pts); markersize = 2.5, color = :orange)
    fig
end

#%% md id=stiff_md
@md raw"""
## A large stiff model, with a sparse Jacobian

The Brusselator reaction–diffusion system on a ring of $N$ cells has $2N$ states, and each right-hand side involves only a cell and its two neighbours:

$\dot u_i = A + u_i^2 v_i - (B + 1) u_i + \alpha (u_{i-1} - 2u_i + u_{i+1}), \qquad \dot v_i = B u_i - u_i^2 v_i + \alpha (v_{i-1} - 2v_i + v_{i+1})$

Diffusion makes it stiff, so an implicit solver needs the Jacobian at every step. The model is built with ordinary Julia comprehensions over Giac variables. Giac differentiates each right-hand side only by the variables it contains, so building the problem costs time in proportion to the nonzero entries. `sparse = true` gives the solver a sparse matrix with the Jacobian's own structure, and the solver factorizes it as a sparse matrix.
"""

#%% code id=stiff
function brusselator(N)
    @giac_var u[1:N] v[1:N] A B α
    lap(w, i) = w[mod1(i - 1, N)] - 2w[i] + w[mod1(i + 1, N)]
    F = vcat([A + u[i]^2 * v[i] - (B + 1) * u[i] + α * lap(u, i) for i in 1:N],
             [B * u[i] - u[i]^2 * v[i] + α * lap(v, i) for i in 1:N])
    s = range(0, 1; length = N)
    u0 = vcat([u[i] => 1 + sin(2π * s[i]) for i in 1:N], [v[i] => 3 for i in 1:N])
    return F, u0, [A => 1, B => 3, α => (N - 1)^2 / 500]
end
cells = 100
Fb, u0b, pb = brusselator(cells)
bt = @elapsed bprob = ODEProblem(Fb, u0b, (0, 10), pb; sparse = true)
ms(p) = round(1000 * @belapsed(solve($p, Rodas5P()), samples = 5, evals = 1); digits = 1)
@examples begin
    "building the problem"
    length(Fb)
    bt
    nnz(bprob.f.jac_prototype), length(Fb)^2
    "solving with Rodas5P, in ms"
    ms(bprob)
    ms(ODEProblem(Fb, u0b, (0, 10), pb))
    ms(ODEProblem(Fb, u0b, (0, 10), pb; jac = false))
end

#%% code id=stiff_plot
let
    bsol = solve(remake(bprob; tspan = (0.0, 30.0)), Rodas5P())
    ts = range(0, 30; length = 400)
    @giac_var u[1:cells]
    U = reduce(hcat, bsol(ts; idxs = u).u)          # cells × times
    fig = Figure(size = (1000, 360))
    ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "cell", title = "u, the activator, on the ring")
    hm = heatmap!(ax, ts, 1:cells, U'; colormap = :inferno)
    Colorbar(fig[1, 2], hm)
    fig
end

#%% md id=equilibria_md
@md raw"""
## Equilibria, exactly and numerically

A predator $z_2$ eats a prey $z_1$ that grows logistically to a carrying capacity $K$, with a saturating appetite:

$\dot z_1 = z_1\left(1 - \tfrac{z_1}{K}\right) - \dfrac{a z_1 z_2}{1 + z_1}, \qquad \dot z_2 = \dfrac{a b z_1 z_2}{1 + z_1} - d z_2$

The same vector of expressions gives Giac's exact equilibria, a `NonlinearProblem` for NonlinearSolve, and an `ODEProblem`. At the coexistence equilibrium the Jacobian's trace changes sign at $K_H$. Above it the equilibrium is unstable and the populations cycle: enriching the prey's environment destabilizes the system.
"""

#%% code id=equilibria
@giac_var z[1:2] K a b d
P = [z1 * (1 - z1 / K) - a * z1 * z2 / (1 + z1), a * b * z1 * z2 / (1 + z1) - d * z2]
pvals = [K => 2, a => 1, b => 1//2, d => 1//5]
@examples begin
    "the equilibria, exactly"
    zstar = C.factor.(C.solve(P, z)[3])
    C.simplify.(substitute(zstar, pvals...))
    "the coexistence equilibrium, by NonlinearSolve"
    nprob = NonlinearProblem(P, [z1 => 1, z2 => 1], pvals)
    solve(nprob).u
    "where it loses stability"
    Jstar = map(e -> substitute(e, z1 => zstar[1], z2 => zstar[2]), jacobian(P, z))
    trace = C.factor(C.simplify(tr(Jstar)))
    KH = only(C.solve(trace ~ 0, K))
    C.simplify(substitute(KH, a => 1, b => 1//2, d => 1//5))
end

#%% code id=enrichment
let
    oprob = ODEProblem(P, [z1 => 1, z2 => 1], (0, 300), pvals)
    fig = Figure(size = (1000, 400))
    for (k, Kv) in enumerate((2.0, 3.0))
        s = solve(remake(oprob; p = [K => Kv]), Rodas5P(); abstol = 1e-9, reltol = 1e-9)
        ts = range(0, 300; length = 6000)
        ax = Axis(fig[1, k]; xlabel = "prey z₁", ylabel = "predator z₂",
                  title = "K = $Kv, " * (Kv < 7 / 3 ? "below" : "above") * " K_H = 7/3")
        lines!(ax, s(ts; idxs = z1).u, s(ts; idxs = z2).u; color = ts, colormap = :viridis, linewidth = 1.2)
        eqv = to_julia.(C.simplify.(substitute(zstar, K => Kv, a => 1, b => 1//2, d => 1//5)))
        scatter!(ax, [eqv[1]], [eqv[2]]; color = :orangered, markersize = 12, marker = :xcross)
    end
    fig
end

#%% md id=generated_md
@md"""
## The generated code

Each function is ordinary Julia, generated from Giac's expressions by `Giac.to_expr` and compiled with RuntimeGeneratedFunctions. The states and parameters are read into local variables of the same names, and a Jacobian writes only its nonzero entries. The functions are compiled as ordinary Julia, so a problem built and solved in the same cell needs no `invokelatest`, and they allocate nothing. Here are the pendulum's Jacobian and the Duffing oscillator's right-hand side and time gradient.

A function from `@giac_function` makes a problem too, with its arguments read as SciML's `(u, p, t)`.
"""

#%% code id=generated
code(f) = Base.Text(string(SciMLBase.RuntimeGeneratedFunctions.get_expression(f)))
allocations(f, args...) = (f(args...); @allocated f(args...))
@giac_function pendulum(q[1:2], k[1:2]) = [q2, -k1 * sin(q1) - k2 * q2]
du, u = zeros(2), [0.1, 0.2]
@examples begin
    "the generated functions"
    code(prob.f.jac)
    code(dprob.f.f)
    code(dprob.f.tgrad)
    allocations(prob.f, du, u, prob.p, 0.0), allocations(prob.f.jac, J, u, prob.p, 0.0)
    "from @giac_function"
    pendulum
    gprob = ODEProblem(pendulum, [π / 2, 0], (0, 20), [9.81, 0.25])
    gprob.f.sys
    solve(gprob, Rodas5P(); abstol = 1e-9, reltol = 1e-9)(20.0) ≈ sol(20.0)
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   BenchmarkTools 1.8.0 6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf
#   CairoMakie 0.15.17 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   NonlinearSolve 4.32.0 8913a72c-1f9b-4ce2-8d82-65094dcecaec
#   OrdinaryDiffEqRosenbrock 2.7.5 43230ef6-c299-4910-a778-202eb28ce4ce
#   OrdinaryDiffEqTsit5 2.1.5 b1df2697-797e-41e3-8120-5422d3b24e4a
#   SciMLBase 3.58.0 0bca4576-84f4-4d90-8ffe-ffa030f20462
#   SparseArrays 1.13.0 2f01184e-e22b-5df5-ae63-d93ebab69eaf
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = e3a4f5a1-22e7-4f29-b4c1-0af2f0ac56fc
# ╚═╡
