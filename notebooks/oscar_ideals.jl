try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Ideals of Giac expressions, in Oscar

A system of polynomial equations has a solution set, and the ideal of the system says how that set comes apart: into irreducible components (branches of equilibria, say), each with a multiplicity. [Oscar](https://www.oscar-system.org/) computes these decompositions in Singular. With Oscar loaded, Giac's extension `GiacOscarExt` adds methods to Oscar's own functions for a vector of Giac expressions or equations, and gives the generators back as `GiacExpr`s:

| Function | Gives |
|----------|-------|
| `minimal_primes(eqs)` | the irreducible components of the set where `eqs` vanish |
| `primary_decomposition(eqs)` | the primary components, `(primary, prime)`, with multiplicity |
| `radical(eqs)` | the polynomials vanishing on the solution set |
| `saturation(eqs, g)` | the components on which `g` is not identically zero |

Every name in the system is a variable, parameters included, so a component can be a relation between a parameter and the state.

Oscar comes from its `master` branch here. The released Oscar 1.8 pins AlgebraicSolving 0.10 and Nemo 0.56, while `master` uses AlgebraicSolving 0.13 and Nemo 0.57, so Oscar and the msolve extension load together.
"""

#%% code id=setup
using Giac, GiacSlate, LinearAlgebra
# Oscar and AlgebraicSolving export names Giac also uses (`solve`, `factor`, `degree`), so they
# are imported, and only the functions used here are brought in
import Oscar, Nemo, AlgebraicSolving
using Oscar: minimal_primes, primary_decomposition, radical, saturation
using AlgebraicSolving: real_solutions

const C = Giac.Commands
(; giac = pkgdir(Giac), oscar = pkgversion(Oscar), algebraic_solving = pkgversion(AlgebraicSolving),
   extensions = [e => Base.get_extension(Giac, e) !== nothing for e in (:GiacOscarExt, :GiacAlgebraicSolvingExt)])

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
# the results here are ideals, written ⟨g₁, g₂⟩: one, a list of components, or primary
# components over their primes
_gens_tex(gs) = "\\left\\langle " * join(Giac.giac_latex.(gs), ",\\ ") * " \\right\\rangle"
_ideal_tex(v::AbstractVector{GiacExpr}) = _gens_tex(v)
_ideal_tex(v::AbstractVector{<:AbstractVector{GiacExpr}}) = join(_gens_tex.(v), ",\\quad ")
_ideal_tex(v::AbstractVector{<:NamedTuple{(:primary, :prime)}}) =
    join([_gens_tex(c.primary) * " \\text{ over } " * _gens_tex(c.prime) for c in v], ",\\quad ")
_ideal_tex(v) = nothing
_value_tex(v) = (t = _ideal_tex(v); t !== nothing ? t : strip(sprint(show, MIME("text/latex"), v), ['$', '\n', ' ']))
_has_tex(v) = _ideal_tex(v) !== nothing || (showable(MIME("text/latex"), v) && !(v isa AbstractString))
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
    _has_tex(v) && return "\$" * replace(_value_tex(v), "|" => "\\vert ") * "\$"
    return _md_code(join(strip.(split(_plain(v), '\n')), "; "))
end
function Base.show(io::IO, ::MIME"text/markdown", ex::Examples)
    println(io, "| code | result |\n|:--|:--|")
    for r in ex.rows
        r isa String ? println(io, "| **", _md_cell(r), "** | |") :
                       println(io, "| ", _md_code(first(r)), " | ", _example_md(last(r)), " |")
    end
end

#%% md id=branches_md
@md"""
## Branches of equilibria

The equilibria of a system with parameters form a set in the space of states and parameters. Its irreducible components are the branches, and `minimal_primes` finds them, each as a reduced Gröbner basis: a system solved as far as it goes.

For the pitchfork `ẋ = μx − x³`, the branches `x = 0` and `x² = μ` cross at `μ = 0`. For the Lorenz system the components also include degenerate cases, `σ = 0` or `β = 0`. `saturation(eqs, σβ)` removes the components on which `σβ` vanishes, which leaves the origin and the pair of equilibria `x = y = ±√(β(ρ − 1))`, `z = ρ − 1`.
"""

#%% code id=branches
let
    @giac_var x y z μ σ ρ β
    lorenz = [σ * (y - x), x * (ρ - z) - y, x * y - β * z]
    @examples begin
        "the pitchfork: two branches through μ = 0"
        minimal_primes([μ * x - x^3])
        "the Lorenz system: every component, degenerate ones included"
        minimal_primes(lorenz)
        "away from σβ = 0: the origin, and the pair at z = ρ - 1"
        minimal_primes(saturation(lorenz, σ * β))
    end
end

#%% md id=multiplicity_md
@md"""
## Multiplicity, and the radical

Two systems can vanish on the same set and still differ: `x² = 0` has the root `x = 0` twice. A primary decomposition keeps that. Each primary component sits over a prime (the component of the solution set it lives on), and a primary component larger than its prime is a solution of multiplicity above one. The radical forgets multiplicity: it is every polynomial that vanishes wherever the system does.
"""

#%% code id=multiplicity
let
    @giac_var x y
    @examples begin
        "the line x = 0, and the origin embedded in it"
        primary_decomposition([x^2, x * y])
        "the line y = 0, and the point (1, 0) embedded in it, of multiplicity 2·3 = 6"
        primary_decomposition([(x - 1)^2 * y, y^3])
        "the radical: multiplicities dropped"
        radical([x^3 - x^2 * y])
        radical([(x - 1)^2 * y, y^3])
    end
end

#%% md id=msolve_md
@md"""
## From a component to its points

A component is a smaller system, so the other engines can take it from there. With the parameters given values, msolve (`real_solutions`, from the AlgebraicSolving extension) gives the real points of the Lorenz pair, here at the classical `ρ = 28`, `β = 8/3`: `x = y = ±6√2 ≈ ±8.485`, `z = 27`. Both extensions load in the same session, because this Oscar is the development version.
"""

#%% code id=msolve
let
    @giac_var x y z σ ρ β
    lorenz = [σ * (y - x), x * (ρ - z) - y, x * y - β * z]
    @examples begin
        pair = first(minimal_primes(saturation(lorenz, σ * β)))
        classical = [C.simplify(substitute(g, ρ => 28, β => 8 // 3)) for g in pair]
        real_solutions(classical, [x, y, z])
    end
end

#%% md id=threads_md
@md"""
## From several tasks at once

Singular is not safe to call from two threads at once, and Slate runs independent cells in parallel. So the extension's calls take turns behind a lock, and decompositions requested from many tasks all come back right. Oscar's functions called on its own ideals are not covered by the lock.
"""

#%% code id=threads
let
    @giac_var x μ
    @examples begin
        Threads.nthreads()
        "sixteen tasks, each finding the pitchfork's two branches"
        counts = fetch.([Threads.@spawn length(minimal_primes([μ * x - x^3])) for _ in 1:16])
        all(==(2), counts)
    end
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   AlgebraicSolving 0.13.0 66b61cbe-0446-4d5d-9090-1ff510639f9d
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   Nemo 0.57.1 2edaba10-b0f1-5616-af89-8c11ac63239a
#   Oscar 1.9.0-DEV f1435218-dba5-11e9-1e4d-f1a5fab5fc13 git=https://github.com/oscar-system/Oscar.jl#master
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = d674f7b7-de7b-4a40-87ba-392bb6d6634c
# ╚═╡
