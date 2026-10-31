# Checking that two expressions are equal, and that a rule rewrites to an equal expression.

"""
    equivalent(a, b; samples = 8, rtol = 1e-9)

Whether `a` and `b` are equal as functions of their variables. Equal if they are written the
same, or Giac reduces `a - b` to 0; otherwise they are compared at `samples` random points, with
values chosen to fit what is assumed about each variable (`@giac_var k::positive` takes positive
values). A variable with no assumption takes each combination of signs with the others in turn
(sample `k` gives the `i`th variable the sign of bit `i` of `k`), so a difference confined to a
quadrant, like `log(x*y) ≠ log(x) + log(y)` for negative `x` and `y`, is found. A random
comparison can miss a difference confined to special values, but not one over a region, which is
what a wrong identity or an unsound rewrite gives. `false` if they differ anywhere sampled, or
could not be evaluated at any point. Where relations among matrix entries are recorded, reduction
modulo them decides instead.
"""
function equivalent(a, b; samples::Integer = 8, rtol::Real = 1e-9)
    a, b = convert(GiacExpr, a), convert(GiacExpr, b)
    _key(a) == _key(b) && return true
    _key(Giac.Commands.normal(a - b)) == "0" && return true
    vars = Giac._variables_of(GiacExpr[a, b])
    # values sampled at random would not satisfy recorded relations among matrix entries; where
    # they apply, reduction modulo them decides
    isempty(_relations_for(vars)) || return _key(apply_relations(a - b)) == "0"
    agreed = 0
    for k in 0:(4 * samples - 1)
        point = Dict{GiacExpr, Any}(v => _sample(v, isodd(k >> (i - 1))) for (i, v) in enumerate(vars))
        va, vb = _value_at(a, point), _value_at(b, point)
        (va === nothing || vb === nothing) && continue
        isapprox(va, vb; rtol, atol = rtol) || return false
        (agreed += 1) >= samples && return true
    end
    return agreed > 0
end

# a random value for `v`, within what is assumed about it (negative, if nothing is assumed and
# `negative` is set)
function _sample(v::GiacExpr, negative::Bool = false)
    tag = Giac._declared(v)
    tag === :positive && return 0.2 + 2rand()
    tag === :negative && return -0.2 - 2rand()
    tag === :nonnegative && return 2rand()
    tag === :nonpositive && return -2rand()
    tag === :integer && return rand(-6:6)
    tag isa Tuple && return tag[1] + (tag[2] - tag[1]) * (0.05 + 0.9rand())
    m = 0.2 + 1.8rand()
    return negative ? -m : m
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

"""
    equivalent(A, B)

For arrays, entry by entry; `B` may be `I` (a `UniformScaling`): `equivalent(Q' * Q, I)`.
"""
equivalent(A::AbstractArray, B::AbstractArray; kwargs...) =
    size(A) == size(B) && all(equivalent(a, b; kwargs...) for (a, b) in zip(A, B))
equivalent(A::AbstractMatrix, J::Giac.LinearAlgebra.UniformScaling; kwargs...) =
    equivalent(A, [i == j ? J.λ : 0 for i in axes(A, 1), j in axes(A, 2)]; kwargs...)

# ---------------------------------------------------------------------------------------------
# Relations among matrix entries
# ---------------------------------------------------------------------------------------------

# the recorded relations that share a variable with `vars`
function _relations_for(vars)
    names = Set(string(v) for v in vars)
    return filter(r -> any(v -> string(v) in names, Giac.free_symbols(r)), Giac._relations())
end

const _BASES = Dict{String, Tuple{GiacExpr, Vector{GiacExpr}}}()
const _BASES_LOCK = ReentrantLock()

"""
    apply_relations(e)

`e` reduced modulo the relations recorded among matrix entries (`@giac_matrix Q[2, 2]::orthogonal`
records `QᵀQ = QQᵀ = I`): numerator and denominator are reduced by a Gröbner basis of the relations
that share a variable with `e`, in a graded order, so what the relations make smaller is replaced.
`q11^2 + q21^2` becomes `1`.
"""
function apply_relations(e::GiacExpr)
    rels = _relations_for(Giac.free_symbols(e))
    isempty(rels) && return e
    basis, vars = _basis(rels)
    order = Giac.giac_eval("revlex")
    reduce(p) = Giac.is_constant(p) ? p : Giac.invoke_cmd(:greduce, p, basis, vars, order)
    rational = Giac.Commands.normal(e)
    num, den = Giac.invoke_cmd(:numer, rational), Giac.invoke_cmd(:denom, rational)
    return Giac.Commands.normal(reduce(num) / reduce(den))
end
apply_relations(A::AbstractArray) = map(apply_relations, A)

function _basis(rels)
    key = join(sort!(_key.(rels)), ";")
    lock(_BASES_LOCK) do
        get!(_BASES, key) do
            vars = Giac._variables_of(rels)
            (Giac.invoke_cmd(:gbasis, rels, vars, Giac.giac_eval("revlex")), vars)
        end
    end
end

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
