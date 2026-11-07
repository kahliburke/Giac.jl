# Linear differential equations GIAC's desolve does not solve. An Euler–Cauchy equation
# Σₖ aₖ xᵏ y⁽ᵏ⁾ = f(x) has constant coefficients in s = ln x, as xᵏ dᵏ/dxᵏ = θ(θ−1)⋯(θ−k+1) for
# θ = d/ds; GIAC solves that one and the solution is taken back to x.

# the general solution of the equation `eq` in the function `y` of `x` when it is an Euler–Cauchy
# equation that GIAC solves in ln x; nothing otherwise
function _euler_cauchy(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    e = _node(_is_equation(eq) ? lhs(eq) - rhs(eq) : eq)
    (_node(x) isa SymNode) || return nothing
    # each derivative y⁽ᵏ⁾ as a symbol dₖ, y itself as d₀
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
    for (k, m) in derivatives
        e = _replace_node(e, m, _node(d[k + 1]))
    end
    E = GiacExpr(_replace_node(e, _node(y), _node(d[1])))
    names = Set(string.(d))
    # Σ qₖ xᵏ dₖ - f with each qₖ a constant
    q = GiacExpr[]
    for k in 0:order
        c = invoke_cmd(:diff, E, d[k + 1])
        isempty(intersect(_free_names(c), names)) || return nothing
        qk = invoke_cmd(:simplify, c / x^k)
        string(x) in _free_names(qk) && return nothing
        push!(q, qk)
    end
    f = -substitute(E, [v => 0 for v in d]...)
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
