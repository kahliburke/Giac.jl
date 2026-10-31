# Matching, by backtracking: each `_match` calls the continuation `k` with the extended bindings
# and returns its result, or returns `nothing` so the caller tries the next alternative.

const Bindings = Base.ImmutableDict{Symbol, Any}     # a slot's expression, a segment's vector

_test(pred, s) = pred === nothing || pred(s) === true

_match(p::Leaf, s, b, k) = p.key == _key(s) ? k(b) : nothing

function _match(p::Slot, s, b, k)
    bound = get(b, p.name, nothing)
    bound === nothing || return _key(bound) == _key(s) ? k(b) : nothing
    _test(p.predicate, s) || return nothing
    return k(Bindings(b, p.name, s))
end

# a segment on its own matches a single expression
_match(p::Segment, s, b, k) = _bind_segment(p, GiacExpr[s], b, k)

function _bind_segment(p::Segment, terms, b, k)
    all(t -> _test(p.predicate, t), terms) || return nothing
    bound = get(b, p.name, nothing)
    bound === nothing || return sort(_key.(bound)) == sort(_key.(terms)) ? k(b) : nothing
    return k(Bindings(b, p.name, terms))
end

function _match(p::Term, s, b, k)
    if haskey(_AC, p.head)
        # `+(~~xs)` matches a sum, not a single term
        length(p.args) == 1 && p.args[1] isa Segment && _head(s) != p.head && return nothing
        return _match_ac(p.head, p.args, 1, _ac_args(p.head, s), b, (b′, rest) -> isempty(rest) ? k(b′) : nothing, true)
    end
    p.head == "^" && return _match_power(p, s, b, k)
    _head(s) == p.head || return nothing
    return _match_seq(p.args, arguments(s), 1, 1, b, k)
end

# A power `base^exponent` also matches `inv(a)` as `a^-1` (`inv(a^n)` as `a^-n`), and, if the
# exponent is a slot with a default, any other expression as itself to that power.
function _match_power(p::Term, s, b, k)
    view = _power_view(s)
    if view === nothing
        ex = p.args[2]
        ex isa Slot && ex.default !== nothing || return nothing
        view = (s, ex.default)
    end
    return _match_seq(p.args, GiacExpr[view...], 1, 1, b, k)
end

function _power_view(s::GiacExpr)
    h = _head(s)
    h == "^" && return Tuple(arguments(s))
    h == "inv" || return nothing
    a = only(arguments(s))
    inner = _power_view(a)
    return inner === nothing ? (a, convert(GiacExpr, -1)) : (inner[1], -inner[2])
end

# Match the pattern arguments `ps[i:end]` against the subject arguments `ss[j:end]` in order; a
# segment takes a run of them, as many as it can, and a slot with a default may be left out at
# the end.
function _match_seq(ps, ss, i, j, b, k)
    i > length(ps) && return j > length(ss) ? k(b) : nothing
    p = ps[i]
    if p isa Segment
        for n in (length(ss) - j + 1):-1:0
            r = _bind_segment(p, ss[j:(j + n - 1)], b, b′ -> _match_seq(ps, ss, i + 1, j + n, b′, k))
            r === nothing || return r
        end
        return nothing
    end
    if j > length(ss)
        p isa Slot && p.default !== nothing || return nothing
        return _match(p, p.default, b, b′ -> _match_seq(ps, ss, i + 1, j, b′, k))
    end
    return _match(p, ss[j], b, b′ -> _match_seq(ps, ss, i + 1, j + 1, b′, k))
end

"The terms of `s` as arguments of `head`: a negated product is a product with a factor `-1`."
function _ac_args(head, s::GiacExpr)
    h = _head(s)
    h == head && return arguments(s)
    head == "*" && h == "-" && return vcat(convert(GiacExpr, -1), _ac_args("*", only(arguments(s))))
    return [s]
end

"Combine terms with `head` (a sum or a product), Giac normalising the result."
function _combine(head, args)
    isempty(args) && return convert(GiacExpr, _AC[head])
    return reduce(head == "+" ? (+) : (*), args)
end

# Match the pattern arguments `ps[i:end]` against the multiset `rest` of subject arguments; `k`
# gets the bindings and the subject arguments left over. With `absorb`, the last free slot takes
# all remaining arguments.
function _match_ac(head, ps, i, rest, b, k, absorb)
    i > length(ps) && return k(b, rest)
    p = ps[i]
    if p isa Segment
        # a segment with a test takes every remaining term that passes it; the one without, all
        # the rest (segments come last, those with a test first)
        p.predicate === nothing && return _bind_segment(p, rest, b, b′ -> _match_ac(head, ps, i + 1, GiacExpr[], b′, k, absorb))
        pass = [_test(p.predicate, t) for t in rest]
        return _bind_segment(p, rest[pass], b, b′ -> _match_ac(head, ps, i + 1, rest[.!pass], b′, k, absorb))
    end
    if absorb && p isa Slot && i == length(ps) && !haskey(b, p.name)
        return _match(p, _combine(head, rest), b, b′ -> k(b′, GiacExpr[]))
    end
    for j in eachindex(rest)
        r = _match(p, rest[j], b, b′ -> _match_ac(head, ps, i + 1, deleteat!(copy(rest), j), b′, k, absorb))
        r === nothing || return r
    end
    # a slot with no term left for it (or already bound to it) stands for its default, or else
    # the identity
    if p isa Slot && (isempty(rest) || haskey(b, p.name))
        absent = something(p.default, convert(GiacExpr, _AC[head]))
        return _match(p, absent, b, b′ -> _match_ac(head, ps, i + 1, rest, b′, k, absorb))
    end
    # a product with a factor that defaults to 0, `(~b = 0) * x`, may be absent from a sum
    if head == "+" && p isa Term && p.head == "*"
        b′ = _absent_product(p, b)
        b′ === nothing || return _match_ac(head, ps, i + 1, rest, b′, k, absorb)
    end
    return nothing
end

# The bindings for a product term that is absent from a sum: its slots take their defaults, one of
# them 0. `nothing` if it has a slot without a default (or bound to something else).
function _absent_product(p::Term, b)
    zero_default = false
    for a in p.args
        a isa Slot || continue
        if a.default === nothing
            haskey(b, a.name) || return nothing
        else
            bound = get(b, a.name, nothing)
            bound === nothing || _key(bound) == _key(a.default) || return nothing
            zero_default |= _key(a.default) == "0"
            bound === nothing && (b = Bindings(b, a.name, a.default))
        end
    end
    return zero_default ? b : nothing
end

# ---------------------------------------------------------------------------------------------
# A sum up to a common factor: `3sin(x)^2 + 3cos(x)^2` matches `sin(~t)^2 + cos(~t)^2` with the
# factor 3, and `3sin(x)^2 + 2cos(x)^2` matches with 2, leaving `sin(x)^2` over.
# ---------------------------------------------------------------------------------------------

_is_number(e::GiacExpr) = Giac.is_numeric(e) || Giac.is_fraction(e)

"A term as its numeric coefficient (a Julia number) and its other factors."
function _split_term(t::GiacExpr)
    c, fs = 1, GiacExpr[]
    for f in _ac_args("*", t)
        _is_number(f) ? (c *= Giac.to_julia(f)) : push!(fs, f)
    end
    return c, fs
end

"The numeric coefficient written in a pattern term (1 if none)."
function _coefficient(p::Pattern)
    p isa Leaf && return (n = _leaf_number(p)) === nothing ? 1 : n
    p isa Term && p.head == "-" && length(p.args) == 1 && return -_coefficient(p.args[1])
    p isa Term && p.head == "*" || return 1
    return prod((n for a in p.args if a isa Leaf for n in (_leaf_number(a),) if n !== nothing); init = 1)
end

function _leaf_number(p::Leaf)
    e = Giac.giac_eval(p.key)
    return _is_number(e) ? Giac.to_julia(e) : nothing
end

# exact division for integers and rationals, which a coefficient keeps
_div(a, b) = a isa Union{Integer, Rational} && b isa Union{Integer, Rational} ? a // b : a / b
_exact(x) = x isa Rational && isone(denominator(x)) ? numerator(x) : x

# `c * prod(factors)` as Giac writes it
_scaled(c, factors) = _combine("*", vcat(convert(GiacExpr, _exact(c)), factors))

# `fs` with one copy of each factor of `K` removed, or `nothing` if some factor is not there
function _without(fs, K)
    rest = copy(fs)
    for f in K
        j = findfirst(g -> _key(g) == _key(f), rest)
        j === nothing && return nothing
        deleteat!(rest, j)
    end
    return rest
end

"""
Match the terms `ps` of a sum pattern against the sum's `terms` up to a common factor `k`: one
term (the anchor) fixes `k` as its numeric coefficient over the pattern's and the factors that
are not part of the matched expression; every other pattern term must then match a subject term
divided by `k`. A subject term with a larger coefficient than needed is used in part, and the
rest of it is left over. `done(b, k, others)` gets the bindings, the factor, and the subject terms
not used (with the parts left over).
"""
function _match_scaled(ps, terms, done)
    any(p -> p isa Segment, ps) && return nothing
    pcs = map(_coefficient, ps)
    split = map(_split_term, terms)
    # each pattern term may be the anchor: the one matched in full sets the factor
    for anchor in eachindex(ps)
        ps[anchor] isa Slot || ps[anchor] isa Leaf && _leaf_number(ps[anchor]) !== nothing && continue
        r = _match_anchored(ps, terms, anchor, pcs, split, done)
        r === nothing || return r
    end
    return nothing
end

function _match_anchored(ps, terms, anchor, pcs, split, done)
    rest = [i for i in eachindex(ps) if i != anchor]
    for j in eachindex(terms)
        cj, fj = split[j]
        (isempty(fj) || length(fj) > 8) && continue
        n = _div(cj, pcs[anchor])
        for mask in 1:(2^length(fj) - 1)
            inside = [isodd(mask >> (t - 1)) for t in eachindex(fj)]
            K = fj[.!inside]
            q = _scaled(pcs[anchor], fj[inside])
            unused = [l for l in eachindex(terms) if l != j]
            r = _match(ps[anchor], q, Bindings(),
                       b -> _match_scaled_rest(ps, rest, 1, terms, split, unused, K, n, pcs, GiacExpr[], b, done))
            r === nothing || return r
        end
    end
    return nothing
end

function _match_scaled_rest(ps, rest, i, terms, split, unused, K, n, pcs, over, b, done)
    if i > length(rest)
        k = _scaled(n, K)
        return done(b, k, vcat(over, terms[unused]))
    end
    p, pc = ps[rest[i]], pcs[rest[i]]
    for (u, l) in enumerate(unused)
        cl, fl = split[l]
        inner = _without(fl, K)
        inner === nothing && continue
        others = deleteat!(copy(unused), u)
        if p isa Slot
            # a slot takes the whole term divided by k
            q, left = _scaled(_div(cl, n), inner), GiacExpr[]
        else
            ratio = _div(cl, n * pc)
            ratio >= 1 || continue
            q = _scaled(pc, inner)
            left = ratio == 1 ? GiacExpr[] : GiacExpr[_scaled(cl - n * pc, fl)]
        end
        r = _match(p, q, b, b′ -> _match_scaled_rest(ps, rest, i + 1, terms, split, others, K, n, pcs, vcat(over, left), b′, done))
        r === nothing || return r
    end
    return nothing
end
