# Rules from the properties declared for function variables (`@giac_var φ(x)::odd`).

"""
    function_identities()

The rules given by the properties declared for function variables: `φ(-x) = -φ(x)` for
`@giac_var φ(x)::odd`, `ψ(t + 2kπ) = ψ(t)` for `ψ(t)::periodic(2π)`, and so on. They are
declarations, not checked: `verify` cannot evaluate an undefined function. [`simplest`](@ref)
includes them.
"""
function function_identities()
    rules, topics = Rule[], Symbol[]
    for (name, props) in sort!(collect(Giac._function_facts()); by = first)
        f = giac_identifier(name)
        for p in props, r in _property_rules(f, p)
            names = Dict{Symbol, Any}(:f => Symbol(name))
            p isa Giac.InverseOf && (names[:g] = Symbol(p.name))
            p isa Giac.Periodic && (names[:T] = p.period)
            push!(rules, _named(r, names))
            push!(topics, Symbol(name))
        end
    end
    return IdentityCatalog(rules, topics)
end

# The rules for one property of `f`, written with `f` (renamed in their display).
function _property_rules(f, p)
    p === :odd && return [@rule f(-~x) => -f(~x)]
    p === :even && return [@rule f(-~x) => f(~x)]
    p === :involution && return [@rule f(f(~x)) => ~x]
    p === :additive && return [@rule f(+(~~ts)) => sum(f(t) for t in ~~ts)]
    p === :linear && return [@rule(f(+(~~ts)) => sum(f(t) for t in ~~ts)),
                             @rule(f(~c::is_number * ~x) => ~c * f(~x))]
    if p isa Giac.Periodic
        T = p.period
        return [@rule(f(~x + ~k) => f(~x) where multiple_of(~k, T)),
                @rule(f(~k) => f(0) where multiple_of(~k, T) && !iszero(~k))]
    end
    if p isa Giac.InverseOf
        g = giac_identifier(p.name)
        return [@rule(f(g(~x)) => ~x), @rule(g(f(~x)) => ~x)]
    end
    throw(ArgumentError("unknown property $p"))
end

# `r` displayed with names: the local `f` written as the function's own name, and values
# (a period) in place of the locals that held them
function _named(r::Rule, names)
    rename(ex::Symbol) = get(names, ex, ex)
    rename(ex::Expr) = Expr(ex.head, map(rename, ex.args)...)
    rename(ex) = ex
    return Rule(r.lhs, r.rhs, r.cond, rename(r.source), r.coefficient, r.heads)
end
