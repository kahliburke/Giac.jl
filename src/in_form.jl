# Writing an expression in a given form: solving for the parameters of a template.

"""
    in_form(e, template, params) -> InForm

Write `e` in the form of `template` by solving for its parameters `params` (a variable or a
collection of them). The other variables of the template are the form's variables; `e` and the
template must agree as rational functions of them, so their coefficients are matched and the
resulting equations solved. Parameters can be expressions in the other symbols of `e`.

The result is a dictionary of `parameter => value` and shows as the template followed by the
values. When several choices fit, the one with fewest negative values is kept and the rest are
in `f.alternatives`. A sign declared for a parameter (`@giac_var ω::positive`) rules out choices
that contradict it. Throws an `ArgumentError` when `e` cannot be written in the form.

```julia
@giac_var s K ζ ω x a h k A B
f = in_form(5 / (s^2 + 3s + 4), K * ω^2 / (s^2 + 2ζ * ω * s + ω^2), (K, ζ, ω))
f[ω]                                    # 2
f.alternatives                          # the choice with ζ = -3/4, ω = -2
in_form(2x^2 - 4x + 7, a * (x - h)^2 + k, (a, h, k))
g = in_form((3s + 5) / (s^2 + 3s + 2), A / (s + 1) + B / (s + 2), (A, B))
substitute(A / (s + 1) + B / (s + 2), g)  # 2/(s+1) + 1/(s+2)
```
"""
function in_form(e, template, params)
    e, template = convert(GiacExpr, e), convert(GiacExpr, template)
    ps = params isa GiacExpr ? [params] : GiacExpr[p for p in params]
    names = Set(string.(ps))
    vars = GiacExpr[v for v in free_symbols(template) if !(string(v) in names)]
    isempty(vars) && throw(ArgumentError("the template has no variables besides the parameters"))
    difference = invoke_cmd(:numer, Commands.normal(e - template))
    equations = unique(string, GiacExpr[c for (_, c) in _collected(difference, vars)])
    fits = _ranked_fits(isempty(equations) ? [ps] : Commands.solve(equations, ps), ps)
    isempty(fits) && throw(ArgumentError("$e cannot be written in the form $template for $(join(string.(ps), ", "))"))
    return InForm(template, vars, fits[1], fits[2:end])
end

# The solutions as `parameter => value` lists, those contradicting a declared sign left out,
# the fewest negative values first
function _ranked_fits(solutions, ps)
    fits = Vector{Pair{GiacExpr, GiacExpr}}[]
    for sol in solutions
        values = GiacExpr[Commands.simplify(v) for v in sol]
        all(_admissible(p, v) for (p, v) in zip(ps, values)) && push!(fits, [p => v for (p, v) in zip(ps, values)])
    end
    negatives(fit) = count(((_, v),) -> Rules.is_negative(v) || startswith(string(v), "-"), fit)
    return sort!(fits; by = negatives)
end

function _admissible(p, v)
    Rules.is_positive(p) && (Rules.is_negative(v) || string(v) == "0") && return false
    Rules.is_negative(p) && (Rules.is_positive(v) || string(v) == "0") && return false
    Rules.is_nonnegative(p) && Rules.is_negative(v) && return false
    return true
end

"""
    InForm

An expression written in the form of a template, from [`in_form`](@ref): a dictionary of
`parameter => value`, with the `template`, its variables `vars`, and the other choices of values
that also fit, `alternatives`.
"""
struct InForm <: AbstractDict{GiacExpr, GiacExpr}
    template::GiacExpr
    vars::Vector{GiacExpr}
    values::Vector{Pair{GiacExpr, GiacExpr}}
    alternatives::Vector{Vector{Pair{GiacExpr, GiacExpr}}}
end

Base.length(f::InForm) = length(f.values)
Base.iterate(f::InForm, i = 1) = i > length(f.values) ? nothing : (f.values[i], i + 1)
function Base.get(f::InForm, p, default)
    k = findfirst(((q, _),) -> string(q) == string(p), f.values)
    return k === nothing ? default : last(f.values[k])
end
Base.haskey(f::InForm, p) = any(((q, _),) -> string(q) == string(p), f.values)

function Base.show(io::IO, f::InForm)
    print(io, f.template, " with ", join(("$p = $v" for (p, v) in f.values), ", "))
end
Base.show(io::IO, ::MIME"text/plain", f::InForm) = show(io, f)

function Base.show(io::IO, ::MIME"text/latex", f::InForm)
    print(io, "\$\$", _in_form_latex(f), "\$\$")
end

function _in_form_latex(f::InForm)
    form = giac_latex(typeset(f.template; by = f.vars))
    values = join((giac_latex(p) * " = " * giac_latex(v) for (p, v) in f.values), ",\\; ")
    return form * ", \\qquad " * values
end
