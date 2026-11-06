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
using SciMLBase, OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, StochasticDiffEq, NonlinearSolve
using SparseArrays, StaticArrays
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
prob = ODEProblem(F, [x1 => π / 2, x2 => 0], (0, 20), [g => 9.81, ℓ => 1, c => 0.25])
sol = solve(prob, Rodas5P(); abstol = 1e-9, reltol = 1e-9)
moon = remake(prob; p = [g => 1.62])
J = zeros(2, 2)
jacobian_at(u) = (prob.f.jac(J, u, prob.p, 0.0); J)
@examples begin
    "the problem"
    prob.f.sys
    prob.u0, prob.p
    "the exact Jacobian, and its compiled form"
    jacobian(F, x)
    jacobian_at([0.1, 0.0])
    "the solution, indexed by the model's variables"
    string(sol.retcode), length(sol.t), sol.stats.njacs
    sol(2.5; idxs = [x1, x2])
    sol(0.0; idxs = E), sol(20.0; idxs = E)
    "parameters and values by name"
    prob.ps[g]
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
# a small model first, so the time below doesn't include compiling the builder itself
let (F3, u03, p3) = brusselator(3)
    ODEProblem(F3, u03, (0, 1), p3; sparse = true)
end
bt = @elapsed bprob = ODEProblem(Fb, u0b, (0, 10), pb; sparse = true)
ms(p) = round(1000 * @belapsed(solve($p, Rodas5P()), samples = 5, evals = 1); digits = 1)
@examples begin
    "building the problem, in seconds"
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
zstar = C.factor.(C.solve(P, z)[3])
nprob = NonlinearProblem(P, [z1 => 1, z2 => 1], pvals)
Jstar = map(e -> substitute(e, z1 => zstar[1], z2 => zstar[2]), jacobian(P, z))
@examples begin
    "the coexistence equilibrium, exactly"
    zstar
    C.simplify.(substitute(zstar, pvals...))
    "and by NonlinearSolve, from (1, 1)"
    solve(nprob).u
    "where it loses stability: the trace of the Jacobian there"
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

#%% md id=cartpole_md
@md raw"""
## Equations of motion, feedback and inputs

A model can also be written as its equations, in unknown functions of time (`@giac_var r(t) θ(t)`) and their derivatives, to any order. `first_order` brings them to a first-order system: each function and its derivatives below the highest are states, written $r, r'$, and Giac solves the equations for the highest derivatives. A problem takes the equations directly, with initial values given by the functions and derivatives themselves: `[θ => 0.3, D(θ) => 0]`.

The cart–pole below is driven by a force $u$ on the cart:

$(M + m)\ddot r + m\ell\cos\theta\,\ddot\theta - m\ell\sin\theta\,\dot\theta^2 = u, \qquad \cos\theta\,\ddot r + \ell\ddot\theta - g\sin\theta = 0$

Linearized about the upright position, it is $\dot s = A s + B u$. Giac places every pole of $A - BK$ at $-2$ by matching the characteristic polynomial with $(\sigma + 2)^4$, which gives the gains exactly in terms of the physical parameters.

An `inputs` entry gives a model's input variable a value. An expression, such as the feedback law $u = -K s + w$, is substituted into the model, so the exact Jacobian includes the closed loop. A Julia function of time, here a push $w(t)$ on the cart, is called by the compiled code.
"""

#%% code id=cartpole
@giac_var r(t) θ(t) M m u w kg[1:4] σ
D = Differential(t)
cartpole = [(M + m) * D(D(r)) + m * ℓ * cos(θ) * D(D(θ)) - m * ℓ * sin(θ) * D(θ)^2 ~ u,
            cos(θ) * D(D(r)) + ℓ * D(D(θ)) - g * sin(θ) ~ 0]
upright(e) = C.simplify(substitute(e, (v => 0 for v in first_order(cartpole).states)..., u => 0))
@examples begin
    "the equations, to first order"
    fo = first_order(cartpole)
    "linearized upright, exactly"
    A = upright.(jacobian(fo.rhs, fo.states))
    B = upright.(jacobian(fo.rhs, [u]))
    "the gains that put every pole at −2"
    charpoly = C.expand(det(σ * Matrix{GiacExpr}(I, 4, 4) - (A - B * permutedims(kg))))
    gains = collect(GiacExpr, C.solve([C.coeff(charpoly - (σ + 2)^4, σ, j) for j in 0:3], collect(kg))[1])
end

#%% code id=cartpole_sim
feedback = -(gains[1] * r + gains[2] * D(r) + gains[3] * θ + gains[4] * D(θ))
push(τ) = 3exp(-((τ - 4) / 0.05)^2)
cprob = ODEProblem(cartpole, [r => 0, D(r) => 0, θ => 0.3, D(θ) => 0], (0, 8),
                   [M => 1, m => 0.2, ℓ => 0.5, g => 9.81]; inputs = [u => feedback + w, w => push])
csol = solve(cprob, Rodas5P(); abstol = 1e-9, reltol = 1e-9, tstops = [4.0])
let ts = range(0, 8; length = 1500)
    fig = Figure(size = (1000, 520))
    ax1 = Axis(fig[1, 1]; ylabel = "θ, rad", title = "the pole, from θ = 0.3 and pushed at t = 4")
    lines!(ax1, ts, csol(ts; idxs = θ).u; color = :orange)
    ax2 = Axis(fig[1, 2]; ylabel = "r, m", title = "the cart")
    lines!(ax2, ts, csol(ts; idxs = r).u; color = :deepskyblue)
    ax3 = Axis(fig[2, 1:2]; xlabel = "t", ylabel = "N", title = "the force u = −Ks + w, and the push w")
    lines!(ax3, ts, csol(ts; idxs = feedback + w).u; color = :mediumpurple, label = "u")
    lines!(ax3, ts, csol(ts; idxs = w).u; color = :gray70, linestyle = :dash, label = "w")
    axislegend(ax3; position = :rb)
    fig
end

#%% md id=noise_md
@md raw"""
## Noise

`SDEProblem(F, G, u0, tspan, p)` is the stochastic model $ds = F\,dt + G\,dW$, built from the same drift: equations, inputs and events work as for an `ODEProblem`. The noise $G$ is compiled with the drift. A vector gives a separate noise on each state. A matrix gives a column for each noise process.

Below, the push becomes a random force on the cart, $u = -Ks + \sigma_w \dot W$. A force enters the first-order system along $\partial F/\partial u$, which Giac differentiates exactly from the solved equations, so $G = \sigma_w\,\partial F/\partial u$ is a single column that depends on $\theta$. The same feedback holds the pole up through the noise.
"""

#%% code id=noise
@giac_var σw
cart_ps = [M => 1, m => 0.2, ℓ => 0.5, g => 9.81, σw => 0.4]
sprob = SDEProblem(cartpole, σw .* jacobian(fo.rhs, [u]), [r => 0, D(r) => 0, θ => 0.3, D(θ) => 0], (0, 8),
                   cart_ps; inputs = [u => feedback])
paths = solve(EnsembleProblem(sprob), LambaEM(), EnsembleThreads(); trajectories = 40, saveat = 0.02).u
noise_at(s) = (G = zeros(4, 1); sprob.g(G, s, sprob.p, 0.0); G)
spread(xs) = sqrt(sum(abs2, xs .- sum(xs) / length(xs)) / (length(xs) - 1))
@examples begin
    "the noise: a random force, along the exact ∂F/∂u"
    σw .* jacobian(fo.rhs, [u])
    "compiled, at the initial state: one noise process"
    noise_at(sprob.u0)
    "the spread at t = 8 of θ and of r, over 40 paths"
    [spread([p[θ][end] for p in paths]), spread([p[r][end] for p in paths])]
end

#%% code id=noise_plot
let fig = Figure(size = (1000, 360))
    ax1 = Axis(fig[1, 1]; xlabel = "t", ylabel = "θ, rad", title = "the pole under a random force, 40 paths")
    ax2 = Axis(fig[1, 2]; xlabel = "t", ylabel = "r, m", title = "the cart")
    for p in paths
        lines!(ax1, p.t, p[θ]; color = (:orange, 0.25))
        lines!(ax2, p.t, p[r]; color = (:deepskyblue, 0.25))
    end
    ts = range(0, 8; length = 800)
    quiet = solve(ODEProblem(cartpole, [r => 0, D(r) => 0, θ => 0.3, D(θ) => 0], (0, 8), cart_ps[1:4];
                             inputs = [u => feedback]), Rodas5P())
    lines!(ax1, ts, quiet(ts; idxs = θ).u; color = :white, linewidth = 2, label = "without noise")
    lines!(ax2, ts, quiet(ts; idxs = r).u; color = :white, linewidth = 2)
    axislegend(ax1; position = :rt)
    fig
end

#%% md id=events_md
@md raw"""
## Events

`events` lists events `condition => affect`, which the solver locates as zero crossings. The condition `a ~ b` occurs when $a - b$ crosses zero in either direction, and an inequality `a < b` when it becomes true. Write a `~` condition in parentheses, because `~` binds more loosely than `=>`. An affect assigns states and parameters, all computed from the values just before the event, or is a Julia function of the integrator, such as `terminate!`. Both the condition and the assignments are compiled like the right-hand side.

A ball bounces with restitution $\kappa$. A relay with hysteresis drives an oscillator, $\ddot\xi + c\,\dot\xi + \xi = v$, where the parameter $v$ switches to $-1$ when $\xi$ rises above $\eta$ and to $+1$ when it falls below $-\eta$. From any start the loop settles into the same limit cycle. An event's parameter changes stay within the solve, so the problem keeps $v = 1$ for the next one.
"""

#%% code id=events
@giac_var b[1:2] κ ξ(t) v η
ball = ODEProblem([b2, -g], [b1 => 1, b2 => 0], (0, 4), [g => 9.81, κ => 0.8];
                  events = [b1 < 0 => [b2 => -κ * b2]])
bsol = solve(ball, Tsit5(); abstol = 1e-10, reltol = 1e-10)
oscillator = [D(D(ξ)) + c * D(ξ) + ξ ~ v]
relay = [ξ > η => [v => -1], ξ < -η => [v => 1]]
rsols = [solve(ODEProblem(oscillator, [ξ => ξ0, D(ξ) => 0], (0, 40), [c => 0.4, v => 1, η => 0.2]; events = relay),
               Tsit5(); abstol = 1e-9, reltol = 1e-9) for ξ0 in (0.0, -2.5)]
# an event saves the state just before and just after it, at the same time
event_times(s) = unique(s.t[i] for i in 1:length(s.t)-1 if s.t[i] == s.t[i+1])
@examples begin
    "the first impacts"
    event_times(bsol)[1:3]
    "exactly: (1, 1 + 2κ, 1 + 2κ + 2κ²) √(2/g)"
    cumsum([1, 2 * 0.8, 2 * 0.8^2]) * sqrt(2 / 9.81)
    "bounces before t = 4, approaching the limit (1 + 2κ/(1 − κ)) √(2/g) ≈ 4.06"
    length(event_times(bsol))
    "the time between relay switches, settling to half the cycle"
    diff(event_times(rsols[1]))[[1, 2, 3, end]]
    "the problem keeps its value of v"
    rsols[1].prob.ps[v]
end

#%% code id=events_plot
let fig = Figure(size = (1000, 380))
    ax1 = Axis(fig[1, 1]; xlabel = "t", ylabel = "height", title = "a bouncing ball, κ = 0.8")
    ts = range(0, 4; length = 2000)
    lines!(ax1, ts, bsol(ts; idxs = b1).u; color = :deepskyblue)
    ax2 = Axis(fig[1, 2]; xlabel = "ξ", ylabel = "ξ′", title = "relay feedback: two starts, one limit cycle")
    vlines!(ax2, [-0.2, 0.2]; color = :gray50, linestyle = :dash)
    for (s, col) in zip(rsols, (:orange, :mediumpurple))
        τ = range(0, 40; length = 8000)
        lines!(ax2, s(τ; idxs = ξ).u, s(τ; idxs = D(ξ)).u; color = (col, 0.8))
    end
    fig
end

#%% md id=generated_md
@md"""
## The generated code

Each function is ordinary Julia, generated from Giac's expressions by `Giac.to_expr` and compiled as a runtime-generated function. The states and parameters are read into local variables of the same names, a subexpression that several entries share is computed once into a local `s1, s2, …`, and a Jacobian writes only its nonzero entries. The functions are compiled as ordinary Julia, so a problem built and solved in the same cell needs no `invokelatest`, and they allocate nothing. Here are the pendulum's Jacobian, the Duffing oscillator's right-hand side and time gradient, and one cell of the Brusselator, where $u_1^2 v_1$ is shared by two rows.

A function from `@giac_function` makes a problem too, with its arguments read as SciML's `(u, p, t)`.
"""

#%% code id=generated
code(f) = Base.Text(string(SciMLBase.RuntimeGeneratedFunctions.get_expression(f)))
allocations(f, args...) = (f(args...); @allocated f(args...))
@giac_function pendulum(q[1:2], k[1:2]) = [q2, -k1 * sin(q1) - k2 * q2]
gprob = ODEProblem(pendulum, [π / 2, 0], (0, 20), [9.81, 0.25])
du, u_ = zeros(2), [0.1, 0.2]
small = let (F3, u03, p3) = brusselator(3)
    ODEProblem(F3, u03, (0, 1), p3)
end
@examples begin
    "the generated functions"
    code(prob.f.jac)
    code(dprob.f.f)
    code(dprob.f.tgrad)
    code(small.f.f)
    allocations(prob.f, du, u_, prob.p, 0.0), allocations(prob.f.jac, J, u_, prob.p, 0.0)
    "from @giac_function"
    pendulum
    gprob.f.sys
    solve(gprob, Rodas5P(); abstol = 1e-9, reltol = 1e-9)(20.0) ≈ sol(20.0)
end

#%% md id=static_md
@md raw"""
## Static arrays, for many small solves

With StaticArrays loaded, `static = true` gives a problem static vectors for its state and parameters, with out-of-place functions that return static arrays. A small model then lives in registers rather than on the heap, which is the fast path for batches of solves.

An undamped pendulum released from rest at $\theta_0$ comes back to rest at half its period, so the event `x2 > 0 => terminate!` ends each solve there. An ensemble of 200 amplitudes measures the period, which grows without bound towards $\theta_0 = \pi$. The exact period is $2\pi\sqrt{\ell/g}\,/\,\mathrm{agm}(1, \cos\tfrac{\theta_0}{2})$.
"""

#%% code id=static
undamped = [x2, -g / ℓ * sin(x1)]
swing = ODEProblem(undamped, [x1 => 1, x2 => 0], (0, 100), [g => 9.81, ℓ => 1];
                   static = true, events = [x2 > 0 => terminate!])
amplitudes = range(0.05, 3.1; length = 200)
swings = solve(EnsembleProblem(swing; prob_func = (q, ctx) -> remake(q; u0 = SVector(amplitudes[ctx.sim_id], 0.0))),
               Tsit5(), EnsembleThreads(); trajectories = length(amplitudes), abstol = 1e-10, reltol = 1e-10,
               save_everystep = false).u
periods = [2 * s.t[end] for s in swings]
agm(a, b) = (while abs(a - b) > 1e-15 * a; a, b = (a + b) / 2, sqrt(a * b); end; a)
exact_periods = [2π * sqrt(1 / 9.81) / agm(1, cos(θ0 / 2)) for θ0 in amplitudes]
heap, regs = (ODEProblem(undamped, [x1 => 1, x2 => 0], (0, 10), [g => 9.81, ℓ => 1]; static) for static in (false, true))
μs(q) = round(1e6 * @belapsed(solve($q, Tsit5(); abstol = 1e-8, reltol = 1e-8)); digits = 1)
@examples begin
    "the static right-hand side"
    code(swing.f.f)
    "the largest error in the period, against the exact one"
    maximum(abs, periods .- exact_periods)
    "ten seconds of swinging with Tsit5, on the heap and static, in μs"
    μs(heap), μs(regs)
end

#%% code id=static_plot
let fig = Figure(size = (900, 360))
    T0 = 2π * sqrt(1 / 9.81)
    ax = Axis(fig[1, 1]; xlabel = "amplitude θ₀", ylabel = "T / 2π√(ℓ/g)", title = "the pendulum's period, from 200 solves ended by an event")
    lines!(ax, amplitudes, exact_periods ./ T0; color = :gray70, linewidth = 2, label = "exact, through the AGM")
    scatter!(ax, amplitudes[1:4:end], periods[1:4:end] ./ T0; color = :orange, markersize = 7, label = "measured")
    axislegend(ax; position = :lt)
    fig
end

#%% md id=exact_md
@md raw"""
## Exact solutions of linear models

A model affine in its states, $\dot x = A x + b$ with $A$ and $b$ constant in time, has the exact solution $\begin{bmatrix} x \\ 1 \end{bmatrix} = e^{t \left[\begin{smallmatrix} A & b \\ 0 & 0 \end{smallmatrix}\right]} \begin{bmatrix} x_0 \\ 1 \end{bmatrix}$. With `analytic = true`, Giac checks that the model is affine, compiles the exact $A$ and $b$, and attaches the solution. SciML then reports each solution's error against it in `sol.errors`. The matrix exponential is computed numerically, so the solution holds at every parameter value, including a repeated eigenvalue, where the closed form in the parameters is singular.

Initial values may be written in the parameters: here the oscillator starts at rest at twice its equilibrium deflection $f/k$. Below, the error of two solvers is measured against the requested tolerance.
"""

#%% code id=exact
@giac_var kf cf ff
oscillator_lin = [x2, -kf * x1 - cf * x2 + ff]
lprob = ODEProblem(oscillator_lin, [x1 => 2ff / kf, x2 => 0], (0, 20), [kf => 4, cf => 0.3, ff => 1]; analytic = true)
tols = 10.0 .^ (-3:-1:-12)
final_error(alg, tol) = solve(lprob, alg; abstol = tol, reltol = tol).errors[:final]
lerrors = Dict(name => [final_error(alg, tol) for tol in tols] for (name, alg) in (("Tsit5", Tsit5()), ("Rodas5P", Rodas5P())))
@examples begin
    "the start, from the parameters: twice the equilibrium deflection f/k"
    lprob.u0
    "the errors of one solve, against the exact solution"
    solve(lprob, Tsit5(); abstol = 1e-8, reltol = 1e-8).errors
    "critically damped (c² = 4k) from x₁ = 1 at rest: (1 + 2t) e^{−2t}, its difference at t = 1.5"
    lprob.f.analytic([1.0, 0.0], [4.0, 4.0, 0.0], 1.5)[1] - 4exp(-3)
end

#%% code id=exact_plot
let fig = Figure(size = (900, 380))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "tolerance", ylabel = "error at t = 20",
              title = "error against the exact solution", xreversed = true)
    lines!(ax, tols, tols; color = :gray50, linestyle = :dash, label = "error = tolerance")
    for (name, col) in (("Tsit5", :orange), ("Rodas5P", :deepskyblue))
        scatterlines!(ax, tols, lerrors[name]; color = col, label = name)
    end
    axislegend(ax; position = :rt)
    fig
end

#%% md id=sensitivity_md
@md"""
## Gradients with respect to the parameters

Fitting a model to data needs the gradient of a loss with respect to the parameters, through the solve. SciMLSensitivity computes it by forward sensitivities or by adjoints. With `autojacvec = false` they use the problem's own Jacobian and parameter Jacobian, which here are Giac's exact ones, compiled. Forward-mode AD runs dual numbers through the same generated code, since it is ordinary generic Julia.

The loss is the sum of squares of the pendulum's states, saved every 0.1 s over 10 s, as a function of $(g, \ell, c)$. Every method agrees with finite differences. With three parameters forward mode is the fastest. An adjoint's cost does not grow with the number of parameters, so it wins for large models.

A forward-sensitivity or interpolating-adjoint solve with an implicit solver such as `Rodas5P` needs that solver's own automatic differentiation turned off, `Rodas5P(autodiff = AutoFiniteDiff())`. The solver differentiates SciMLSensitivity's augmented system, whose work arrays hold plain `Float64`s. The quadrature adjoint has no such restriction.
"""

#%% code id=sensitivity
using SciMLSensitivity, Zygote, ForwardDiff
sprob_p = remake(prob; tspan = (0.0, 10.0))
loss(q, sensealg = nothing; alg = Tsit5()) =
    sum(abs2, Array(solve(remake(sprob_p; p = q), alg; abstol = 1e-10, reltol = 1e-10, saveat = 0.1, sensealg)))
p0 = collect(sprob_p.p)
e_(i) = 1:3 .== i
fd = [(loss(p0 .+ 1e-5e_(i)) - loss(p0 .- 1e-5e_(i))) / 2e-5 for i in 1:3]
adjoint_gradient(sensealg; kw...) = Zygote.gradient(q -> loss(q, sensealg; kw...), p0)[1]
ms_(f) = round(1e3 * @belapsed($f()); digits = 2)
@examples begin
    "∂loss/∂(g, ℓ, c): finite differences, the reference"
    fd
    "forward-mode AD through the solver"
    ForwardDiff.gradient(loss, p0)
    "forward sensitivities, interpolating and quadrature adjoints, with the exact jac and paramjac"
    adjoint_gradient(ForwardSensitivity(autojacvec = false))
    adjoint_gradient(InterpolatingAdjoint(autojacvec = false))
    adjoint_gradient(QuadratureAdjoint(autojacvec = false))
    "an implicit solver, with its own differentiation by finite differences"
    adjoint_gradient(InterpolatingAdjoint(autojacvec = false); alg = Rodas5P(autodiff = SciMLBase.ADTypes.AutoFiniteDiff()))
    "time per gradient, in ms: forward AD, forward sensitivities, interpolating, quadrature"
    ms_(() -> ForwardDiff.gradient(loss, p0)), ms_(() -> adjoint_gradient(ForwardSensitivity(autojacvec = false))),
        ms_(() -> adjoint_gradient(InterpolatingAdjoint(autojacvec = false))), ms_(() -> adjoint_gradient(QuadratureAdjoint(autojacvec = false)))
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   BenchmarkTools 1.8.0 6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf
#   CairoMakie 0.15.17 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   ForwardDiff 1.4.7 f6369f11-7733-5829-9624-2563aa707210
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   NonlinearSolve 4.32.0 8913a72c-1f9b-4ce2-8d82-65094dcecaec
#   OrdinaryDiffEqRosenbrock 2.7.5 43230ef6-c299-4910-a778-202eb28ce4ce
#   OrdinaryDiffEqTsit5 2.1.5 b1df2697-797e-41e3-8120-5422d3b24e4a
#   SciMLBase 3.58.0 0bca4576-84f4-4d90-8ffe-ffa030f20462
#   SciMLSensitivity 7.119.14 1ed8b502-d754-442c-8d5d-10ac956f44a1
#   SparseArrays 1.13.0 2f01184e-e22b-5df5-ae63-d93ebab69eaf
#   StaticArrays 1.9.24 90137ffa-7385-5640-81b9-e52037218182
#   StochasticDiffEq 7.2.0 789caeaf-c7a9-5a7d-9973-96adeb23e2a0
#   Zygote 0.7.15 e88e6eb3-aa80-5325-afca-941959d7151f
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = e3a4f5a1-22e7-4f29-b4c1-0af2f0ac56fc
# ╚═╡
