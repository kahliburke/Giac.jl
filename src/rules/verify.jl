# Checking that two expressions are equal, and that a rule rewrites to an equal expression.

"""
    equivalent(a, b; samples = 6, rtol = 1e-9)

Whether `a` and `b` are equal as functions of their variables. Equal if they are written the
same, or Giac reduces `a - b` to 0; otherwise they are compared at `samples` random points, with
values chosen to fit what is assumed about each variable (`@giac_var k::positive` takes positive
values). A random comparison can miss a difference confined to special values, but not one that
holds over a region, which is what a wrong identity or an unsound rewrite gives. `false` if they
differ anywhere sampled, or could not be evaluated at any point.
"""
function equivalent(a, b; samples::Integer = 6, rtol::Real = 1e-9)
    a, b = convert(GiacExpr, a), convert(GiacExpr, b)
    _key(a) == _key(b) && return true
    _key(Giac.Commands.normal(a - b)) == "0" && return true
    vars = Giac._variables_of(GiacExpr[a, b])
    agreed = 0
    for _ in 1:(4 * samples)
        point = Dict{GiacExpr, Any}(v => _sample(v) for v in vars)
        va, vb = _value_at(a, point), _value_at(b, point)
        (va === nothing || vb === nothing) && continue
        isapprox(va, vb; rtol, atol = rtol) || return false
        (agreed += 1) >= samples && return true
    end
    return agreed > 0
end

# a random value for `v`, within what is assumed about it
function _sample(v::GiacExpr)
    tag = Giac._declared(v)
    tag === :positive && return 0.2 + 2rand()
    tag === :negative && return -0.2 - 2rand()
    tag === :nonnegative && return 2rand()
    tag === :nonpositive && return -2rand()
    tag === :integer && return rand(-6:6)
    tag isa Tuple && return tag[1] + (tag[2] - tag[1]) * (0.05 + 0.9rand())
    return 4rand() - 2
end

# `e` as a number at `point`, or `nothing` where it is undefined or infinite
function _value_at(e::GiacExpr, point)
    v = try
        Giac.to_julia(Giac.Commands.evalf(substitute(e, point)))
    catch
        return nothing
    end
    v isa Number && isfinite(v) || return nothing
    return v
end

"""
    verify(r; samples = 6) -> Union{Bool, Nothing}

Check that the rule `r` rewrites to an equal expression: its left side is instantiated with a
fresh variable for each slot, the rule applied, and the two sides compared with
[`equivalent`](@ref). Each fresh variable is assumed what its slot's test asks for (an interval,
an integer, a sign); slots without a test are tried positive, then real, then integer, so a
`where` condition can hold. `nothing` if the rule does not apply to such an instance.
"""
function verify(r::Rule; samples::Integer = 6)
    slots = _slots(r.lhs)
    fresh = Dict(s.name => (s isa Segment ? [giac_identifier("verify_$(s.name)_$i") for i in 1:2] :
                                            giac_identifier("verify_$(s.name)")) for s in slots)
    vars = reduce(vcat, (v isa Vector ? v : [v] for v in values(fresh)); init = GiacExpr[])
    b = foldl((b, (n, v)) -> Bindings(b, n, v), fresh; init = Bindings())
    tested = Dict(s.name => _assumption_for(s) for s in slots if s isa Slot && _assumption_for(s) !== nothing)
    try
        for tag in (:positive, :real, :integer)
            for (name, v) in fresh, w in (v isa Vector ? v : [v])
                Giac._declare!(w, get(tested, name, tag))
            end
            subject = _instantiate(r.lhs, b)
            result = r(subject)
            result === nothing && continue
            return equivalent(subject, result; samples)
        end
        return nothing
    finally
        foreach(v -> Giac._declare!(v, nothing), vars)
    end
end
verify(rs::AbstractVector{Rule}; kwargs...) = [verify(r; kwargs...) for r in rs]

# what to assume of a fresh variable for a slot with a test, so the test can hold
function _assumption_for(s::Slot)
    p = s.predicate
    p isa InInterval && return (_float(p.lo), _float(p.hi))
    p === is_integer_valued && return :integer
    p === is_negative && return :negative
    p === is_positive && return :positive
    p === is_nonnegative && return :nonnegative
    p === is_real && return :real
    return nothing
end
