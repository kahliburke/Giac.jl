# Rules from identities written as equations, rules turned around, and substitution of a
# relation (`x^2 + y^2 => r^2`) where Giac's own substitution only replaces variables and calls.

"An expression for the pattern `p` with its slots and segments taken from the bindings `b`."
_instantiate(p::Leaf, b) = p.expr
function _instantiate(p::Slot, b)
    haskey(b, p.name) && return b[p.name]
    p.default === nothing && throw(ArgumentError("no value for the slot ~$(p.name)"))
    return p.default
end
_instantiate(p::Segment, b) = b[p.name]
function _instantiate(p::Term, b)
    args = GiacExpr[]
    for a in p.args
        v = _instantiate(a, b)
        v isa AbstractVector ? append!(args, v) : push!(args, convert(GiacExpr, v))
    end
    haskey(_AC, p.head) && return _combine(p.head, args)
    p.head == "^" && return args[1]^args[2]
    p.head == "-" && return -only(args)
    p.head == "inv" && return inv(only(args))
    p.head == "of" && return args[1](args[2:end]...)
    return Giac.invoke_cmd(Symbol(p.head), args...)
end

# the slots and segments of a pattern, each once
function _slots(p::Pattern, out = Pattern[])
    if p isa Slot || p isa Segment
        any(q -> q.name === p.name, out) || push!(out, p)
    elseif p isa Term
        foreach(a -> _slots(a, out), p.args)
    end
    return out
end

"""
    rule(eq, vars...)

A rule from an identity written as an equation, `lhs ~ rhs`, with the variables `vars` standing
for any expression: `rule(sin(2x) ~ 2sin(x)cos(x), x)` is `@rule sin(2 * ~x) => 2sin(~x)cos(~x)`.
A variable can carry a test, `x => is_positive`. Other variables in the equation match only
themselves.

```julia
@giac_var x y
double_angle = rule(sin(2x) ~ 2sin(x) * cos(x), x)
inverse(double_angle)              # 2sin(x)cos(x) → sin(2x)
```
"""
function rule(eq::GiacExpr, vars...)
    Giac._is_equation(eq) || throw(ArgumentError("rule takes an equation `lhs ~ rhs`, got $eq"))
    vs = GiacExpr[v isa Pair ? first(v) : v for v in vars]
    preds = Any[v isa Pair ? last(v) : nothing for v in vars]
    return _rule(eq, vs, preds, [Symbol(string(v)) for v in vs])
end

# A rule from `eq` with the variables `vs` as slots named `names` (with tests `preds`). The tests
# show as a `where` clause.
function _rule(eq::GiacExpr, vs, preds, names)
    left = substitute(Giac.lhs(eq), Dict{GiacExpr, Any}(v => _slot_var(n) for (v, n) in zip(vs, names)))
    right = Giac.rhs(eq)
    rhs = b -> substitute(right, Dict{GiacExpr, Any}(v => b[n] for (v, n) in zip(vs, names)))
    rename = Dict(Symbol(string(v)) => n for (v, n) in zip(vs, names))
    written(e) = _slotted(Meta.parse(string(e)), rename)
    conds = [_test_code(p, n) for (p, n) in zip(preds, names) if p !== nothing]
    body = written(right)
    body = isempty(conds) ? body : Expr(:where, body, foldl((a, c) -> Expr(:&&, a, c), conds))
    slots = Any[(n, :slot, p, nothing) for (n, p) in zip(names, preds)]
    return Rule(left, slots, rhs, nothing, Expr(:call, :(=>), written(Giac.lhs(eq)), body))
end

# a slot's test as written code, for display
_test_code(p::InInterval, n) = Expr(:call, :in_interval, Expr(:call, :~, n), p.lo, p.hi)
_test_code(p, n) = Expr(:call, nameof(p), Expr(:call, :~, n))

# written code with the variables in `rename` as the slots they stand for
_slotted(ex::Symbol, rename) = haskey(rename, ex) ? Expr(:call, :~, rename[ex]) : ex
_slotted(ex::Expr, rename) = Expr(ex.head, (_slotted(a, rename) for a in ex.args)...)
_slotted(ex, rename) = ex

"""
    inverse(r)

The rule `r` turned around, right side to left, with the same condition: `inverse.(log_expand())`
combines logarithms. The right side must be an expression of the slots (no Julia logic, no
segments), and use every slot of the left.
"""
function inverse(r::Rule)
    slots = _slots(r.lhs)
    any(s -> s isa Segment, slots) && throw(ArgumentError("a rule with segments cannot be turned around: $(r.source)"))
    rhs, cond = _rule_sides(r.source)
    _is_math(_plain(rhs)) || throw(ArgumentError("the right side of $(r.source) is Julia code, not an expression"))
    placeholders = foldl((b, s) -> Bindings(b, s.name, _slot_var(s.name)), slots; init = Bindings())
    left = convert(GiacExpr, r.rhs(placeholders))
    table = Dict{String, Pattern}(_key(_slot_var(s.name)) => s for s in slots)
    pattern = _pattern(left, table)
    used = Set(s.name for s in _slots(pattern))
    missing = [s.name for s in slots if s.name ∉ used]
    isempty(missing) || throw(ArgumentError("the right side of $(r.source) does not use ~$(join(missing, ", ~")), so it cannot be turned around"))
    newlhs = r.source.args[2]
    source = Expr(:call, :(=>), rhs, cond === nothing ? newlhs : Expr(:where, newlhs, cond))
    return Rule(pattern, b -> _instantiate(r.lhs, b), r.cond, source)
end

# ---------------------------------------------------------------------------------------------
# Substituting a relation
# ---------------------------------------------------------------------------------------------

# how often the variables named in `names` occur in `e`
_occurrences(e::GiacExpr, names) = iscall(e) ? sum(a -> _occurrences(a, names), arguments(e); init = 0) :
                                   Int(_key(e) in names)

"""
Replace the expression `key` (a sum, product or power) by `value` in `e`. Occurrences are found
as a rule finds them (in any order, up to a common factor: `2x^2 + 2y^2 + z` has `x^2 + y^2`
twice); where that leaves `key`'s variables in place, a Gröbner reduction (see
[`express`](@ref)) may write more of `e` in terms of `key`, and is used if it does.
"""
function _substitute_relation(e::GiacExpr, key::GiacExpr, value)
    r = Rule(_pattern(key, Dict{String, Pattern}()), b -> value, nothing, :($(Symbol(string(key))) => $(Symbol(string(value)))))
    out = rewrite(e, r; once = true)
    names = Set(string(v) for v in Giac.free_symbols(key))
    left = _occurrences(out, names)
    left == 0 && return out
    name = giac_identifier("substituted_relation")
    reduced = try
        Giac.express(out, name => key)
    catch
        return out
    end
    _occurrences(reduced, names) < left || return out
    return Giac.Commands.factor(substitute(reduced, name => value))
end
