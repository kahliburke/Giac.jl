# Named rewrites of whole expressions, for `simplest`: Giac's own transformations (normal form,
# factoring, trigonometric linearisation, …) beside the rule sets.

"""
    Transform(name, f)

A rewrite of a whole expression, `f(e)`, labelled `name` in traces. For [`simplest`](@ref),
alongside rules.
"""
struct Transform
    name::String
    f::Any
end
(t::Transform)(e::GiacExpr) = t.f(e)
Base.show(io::IO, t::Transform) = print(io, "Transform(\"", t.name, "\")")

_label_latex(t::Transform) = "\\text{" * t.name * "}"

_giac(cmd::Symbol, name) = Transform(name, e -> Giac.invoke_cmd(cmd, e))

# `cmd` applied only to the calls of the functions in `heads`, not across the whole expression
function _at_calls(cmd::Symbol, name, heads)
    f(e) = !iscall(e) ? e : _head(e) in heads ? Giac.invoke_cmd(cmd, e) : _rebuild_args(f, e)
    return Transform(name, f)
end

const _TRIG = ("sin", "cos", "tan")

"""
    standard_transforms()

The transformations [`simplest`](@ref) uses by default, each valid for every value of its
variables (or applied only under the conditions where it is): rational normal form, factoring,
expansion, partial fractions; trigonometric linearisation (`tlin`), collection (`tcollect`),
expansion of multiple angles, and rewriting in `sin`, `cos` or `tan` only; `exp(n log x)` as
`x^n`; and the simplifying identities of the catalog ([`identities`](@ref)) with the expanding
direction of the log and power identities, whose conditions use declared assumptions; the rules
for declared properties of functions ([`function_identities`](@ref)); and, if relations among
matrix entries are recorded, reduction modulo them ([`apply_relations`](@ref)).

Giac's `simplify`, `lnexpand` and `lncollect` are left out: they combine and split logarithms
without regard to sign.
"""
standard_transforms() = vcat(Any[
    _giac(:normal, "normal form"), _giac(:factor, "factor"), _giac(:expand, "expand"),
    _giac(:partfrac, "partial fractions"),
    _giac(:tlin, "linearize trig"), _giac(:tcollect, "collect trig"),
    _at_calls(:texpand, "expand trig", _TRIG),
    _giac(:trigsin, "in sin"), _giac(:trigcos, "in cos"), _giac(:trigtan, "in tan"),
    _giac(:exp2pow, "exp to powers"),
    identities(), log_expand(), power_expand(), function_identities()],
    isempty(Giac._relations()) ? Any[] : Any[Transform("declared relations", apply_relations)])

# ---------------------------------------------------------------------------------------------
# Goals: what "simplest" means
# ---------------------------------------------------------------------------------------------

_count(pred, e::GiacExpr) = Int(pred(e)) + (iscall(e) ? sum(a -> _count(pred, a), arguments(e); init = 0) : 0)

_has_sum(e) = iscall(e) && any(a -> _head(a) == "+", arguments(e))
_trig_power(e) = _head(e) == "^" && _head(first(arguments(e))) in _TRIG ||
                 _head(e) == "*" && count(a -> _head(a) in _TRIG || _head(a) == "^" && _head(first(arguments(a))) in _TRIG, arguments(e)) > 1

"""
The cost of `e` for a goal, compared lexicographically: what the goal minimises first, then the
size of the expression.

- `:compact`: the fewest nodes ([`expression_size`](@ref)).
- `:expanded`: no product or power of a sum.
- `:factored`: as few terms at the top as possible.
- `:trig_linear`: no powers or products of trigonometric functions (sums of `sin` and `cos` of
  multiple angles, as for a Fourier series).
"""
function goal_cost(goal::Symbol, e::GiacExpr)
    n = expression_size(e)
    goal === :compact && return (n,)
    goal === :expanded && return (_count(x -> _head(x) in ("*", "^") && _has_sum(x), e), n)
    goal === :factored && return (_head(e) == "+" ? length(arguments(e)) : 1, n)
    goal === :trig_linear && return (_count(_trig_power, e), n)
    throw(ArgumentError("unknown goal :$goal; use :compact, :expanded, :factored or :trig_linear, or pass a cost function"))
end
