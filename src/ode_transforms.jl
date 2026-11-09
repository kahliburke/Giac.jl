# Linear differential equations GIAC's desolve does not solve. An Euler–Cauchy equation
# Σₖ aₖ xᵏ y⁽ᵏ⁾ = f(x) has constant coefficients in s = ln x, as xᵏ dᵏ/dxᵏ = θ(θ−1)⋯(θ−k+1) for
# θ = d/ds; GIAC solves that one and the solution is taken back to x.

# the equation `eq` in the function `y` of `x` with each derivative y⁽ᵏ⁾ a symbol dₖ, y itself d₀:
# the expression `E` (the equation's sides subtracted), the symbols `d`, and the tree of `eq`;
# nothing where y has no derivative in it
function _derivative_symbols(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    e = _node(_is_equation(eq) ? lhs(eq) - rhs(eq) : eq)
    (_node(x) isa SymNode) || return nothing
    derivatives = Dict{Int, Node}()
    _any_node(e) do m
        if m isa CallNode && m.f == "diff"
            d = _derivative_in(GiacExpr(m), x)
            d !== nothing && isequal(_node(d[1]), _node(y)) && (derivatives[d[2]] = m)
        end
        false
    end
    isempty(derivatives) && return nothing
    order = maximum(keys(derivatives))
    d = [_fresh_symbol(e, "d$(k)__") for k in 0:order]
    t = e
    for (k, m) in derivatives
        t = _replace_node(t, m, _node(d[k + 1]))
    end
    return (E = GiacExpr(_replace_node(t, _node(y), _node(d[1]))), d = d, e = e)
end

# the equation `eq` in the function `y` of `x` as Σₖ qₖ y⁽ᵏ⁾ = f: the coefficients q₀, …, qₙ, f,
# and the tree of `eq`; nothing where it is not linear in y and its derivatives
function _linear_coefficients(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    S = _derivative_symbols(eq, x, y)
    S === nothing && return nothing
    E, d = S.E, S.d
    names = Set(string.(d))
    q = GiacExpr[]
    for k in eachindex(d)
        c = invoke_cmd(:diff, E, d[k])
        isempty(intersect(_free_names(c), names)) || return nothing
        push!(q, c)
    end
    return (q = q, f = -substitute(E, [v => 0 for v in d]...), e = S.e)
end

# the general solution of the equation `eq` in the function `y` of `x` when it is an Euler–Cauchy
# equation that GIAC solves in ln x; nothing otherwise
function _euler_cauchy(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    c = _linear_coefficients(eq, x, y)
    c === nothing && return nothing
    e, f, order = c.e, c.f, length(c.q) - 1
    # Σ qₖ xᵏ dₖ - f with each qₖ a constant
    q = GiacExpr[]
    for k in 0:order
        qk = invoke_cmd(:simplify, c.q[k + 1] / x^k)
        string(x) in _free_names(qk) && return nothing
        push!(q, qk)
    end
    # the equation in s: Σ qₖ θ(θ−1)⋯(θ−k+1) z = f(eˢ)
    θ = _fresh_symbol(e, "theta")
    L = invoke_cmd(:expand, sum(q[k + 1] * prod((θ - j for j in 0:(k - 1)); init = one(GiacExpr)) for k in 0:order))
    s, z = _fresh_symbol(e, "s"), string(_fresh_symbol(e, "z"))
    terms = String[]
    for j in 0:order
        b = invoke_cmd(:coeff, L, θ, j)
        _is_zero_expr(b) && continue
        push!(terms, "($(b))*" * (j == 0 ? "$z($s)" : "diff($z($s),$s,$j)"))
    end
    F = invoke_cmd(:simplify, substitute(f, x => exp(s)))
    sol = try
        giac_eval("desolve(" * join(terms, "+") * " = $(F), $s, $z)")
    catch err
        err isa GiacError || rethrow()
        return nothing
    end
    giac_type(sol) == GenTypes.VECT && (length(sol) == 1 || return nothing; sol = sol[1])
    return invoke_cmd(:simplify, substitute(sol, s => log(x)))
end

# GIAC's answer when desolve finds nothing: a message, or a list of none but messages
_gave_up(r::GiacExpr) = giac_type(r) == GenTypes.STRNG ||
    (giac_type(r) == GenTypes.VECT && all(i -> giac_type(r[i]) == GenTypes.STRNG, 1:length(r)))
