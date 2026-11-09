# Partial differential equations in a function u of two variables, which GIAC's desolve does not
# solve: first-order linear equations by characteristics, with an arbitrary function F_0 of the
# characteristic variable or a condition on a line fixing it, and the heat and wave equations on
# an interval with homogeneous Dirichlet or Neumann conditions at its ends, by separation of
# variables.

# the solution of `eqs`, an equation or a vector of one and conditions, for `u`, a function of two
# variables; nothing where no method applies
function _pdesolve(eqs, u::GiacExpr)
    parsed = _function_call(u)
    (parsed === nothing || length(parsed[2]) != 2) && return nothing
    name, vars = String(parsed[1]), [giac_eval(String(v)) for v in parsed[2]]
    eq, conditions = eqs isa GiacExpr ? (eqs, Any[]) : (first(eqs), collect(Any, eqs[2:end]))
    eq isa GiacExpr || return nothing
    q = _pde_coefficients(eq, name, vars)
    q === nothing && return nothing
    lines = Any[]
    for c in conditions
        l = _line_condition(c, name, vars)
        l === nothing && return nothing
        push!(lines, l)
    end
    orders = keys(q.q)
    if all(α -> sum(α) <= 1, orders)
        return _first_order_pde(q, vars, lines)
    end
    return _separated_pde(q, vars, lines)
end

# a node that is u or one of its partial derivatives, diff(…diff(u, k₁)…, kₘ)(a₁, a₂) with kᵢ the
# argument differentiated: the multi-index of the derivative and the arguments; nothing otherwise
function _partial_index(n::Node, name::String)
    n isa CallNode || return nothing
    n.f == name && length(n.args) == 2 && return ((0, 0), n.args)
    (n.f == "of" && length(n.args) == 3) || return nothing
    # a derivative at (x, y) applied to other arguments, as diff(u, x)(0, t) is written
    if n.args[1] isa CallNode && n.args[1].f == "of"
        p = _partial_index(n.args[1], name)
        return p === nothing ? nothing : (p[1], n.args[2:3])
    end
    D, α = n.args[1], [0, 0]
    while D isa CallNode && D.f == "diff" && length(D.args) == 2 && D.args[2] isa NumNode && D.args[2].v in (0, 1)
        α[D.args[2].v + 1] += 1
        D = D.args[1]
    end
    (D isa SymNode && D.name == name) || return nothing
    return ((α[1], α[2]), n.args[2:3])
end

# the equation as Σ q_α ∂^α u = f: the coefficients by multi-index, and f; nothing where it is not
# linear in u and its derivatives, or has them at points other than (x, y)
function _pde_coefficients(eq::GiacExpr, name::String, vars)
    e = _node(_is_equation(eq) ? lhs(eq) - rhs(eq) : eq)
    found = Dict{Tuple{Int, Int}, Node}()
    elsewhere = _any_node(e) do m
        p = _partial_index(m, name)
        p === nothing && return false
        all(isequal.(p[2], _node.(vars))) || return true
        found[p[1]] = m
        false
    end
    (elsewhere || isempty(found)) && return nothing
    t, d = e, Dict{Tuple{Int, Int}, GiacExpr}()
    for (α, m) in found
        d[α] = _fresh_symbol(e, "d$(α[1])$(α[2])__")
        t = _replace_node(t, m, _node(d[α]))
    end
    E = GiacExpr(t)
    names = Set(string.(values(d)))
    q = Dict{Tuple{Int, Int}, GiacExpr}()
    for (α, s) in d
        c = invoke_cmd(:diff, E, s)
        isempty(intersect(_free_names(c), names)) || return nothing
        _is_zero_expr(c) || (q[α] = c)
    end
    return (q = q, f = -substitute(E, [s => 0 for s in values(d)]...))
end

# a condition ∂^α u = g on a line where one variable has the value `at`: (k, at, α, g) with k the
# index of that variable; nothing for other conditions
function _line_condition(c, name::String, vars)
    parts = c isa GiacExpr && _is_equation(c) ? (lhs(c), rhs(c)) : nothing
    parts === nothing && return nothing
    p = _partial_index(_node(parts[1]), name)
    p === nothing && return nothing
    α, args = p
    names = Set(string.(vars))
    for k in 1:2
        other = 3 - k
        if isequal(args[other], _node(vars[other])) && isempty(intersect(_free_names(GiacExpr(args[k])), names))
            return (k = k, at = GiacExpr(args[k]), α = α, g = parts[2])
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# First-order linear equations: a u_x + b u_y + c u = f
# ---------------------------------------------------------------------------------------------

# Along a characteristic, dy/dx = b/a, ξ(x, y) is constant and u solves a du/dx + c u = f, so
# u = A F₀(ξ) + B with A = 1/μ, μ = exp ∫ c/a dx, B = (1/μ) ∫ f μ / a dx, the integrals taken with y
# written by x and ξ. A condition u = g on a line fixes F₀.
function _first_order_pde(q, vars, lines)
    zero_ = convert(GiacExpr, 0)
    a, b = get(q.q, (1, 0), zero_), get(q.q, (0, 1), zero_)
    c, f = get(q.q, (0, 0), zero_), q.f
    flipped = _is_zero_expr(a)
    x, y = flipped ? (vars[2], vars[1]) : (vars[1], vars[2])
    flipped && ((a, b) = (b, a))
    length(lines) <= 1 || return nothing
    w = _fresh_symbol(_node(a + b + c + f), "w")
    ξ = _characteristic(_simplified(b / a), x, y)
    ξ === nothing && return nothing
    # y on the characteristic ξ = w, where the coefficients need it
    needs_y = any(e -> string(y) in _free_names(e), (c / a, f / a))
    Y = needs_y ? _unique_solution(ξ - w, y) : y
    Y === nothing && return nothing
    on(e) = _simplified(substitute(e, y => Y))
    # any multiple of an integrating factor is one, so ln|g| is taken as ln g
    μ = exp(_without_abs(Commands.integrate(on(c / a), x)))
    B = Commands.integrate(on(f / a) * μ, x) / μ
    back(e) = _simplified(substitute(e, w => ξ))
    A, B = back(1 / μ), back(B)
    F = isempty(lines) ? giac_eval("F_0($(ξ))") : _fitted_function(only(lines), A, B, ξ, flipped ? reverse(vars) : vars, w)
    F === nothing && return nothing
    return _exp_powers(_simplified(A * F + B))
end

# e with each |z| written z
_without_abs(e::GiacExpr) = GiacExpr(_without_abs(_node(e)))
_without_abs(n::Node) = n
_without_abs(n::CallNode) = n.f == "abs" && length(n.args) == 1 ? _without_abs(n.args[1]) : _call(n.f, Node[_without_abs(a) for a in n.args])
_without_abs(n::AddNode) = _add((_scale(_without_abs(t), c) for (t, c) in n.terms)...)
_without_abs(n::MulNode) = _mul(NumNode(n.coef), (_pow(_without_abs(b), _without_abs(p)) for (b, p) in n.factors)...)

# a first integral of dy/dx = r: ξ(x, y), constant on the solutions
function _characteristic(r::GiacExpr, x::GiacExpr, y::GiacExpr)
    string(y) in _free_names(r) || return y - Commands.integrate(r, x)
    Yf = giac_eval("Y__($(x))")
    S = try
        Commands.desolve(Commands.diff(Yf, x) ~ substitute(r, y => Yf), Yf)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (S isa GiacExpr && !_gave_up(S)) || return nothing
    # the branches of an implicit solution, y = ±√(c - x²), give the same first integral
    giac_type(S) == GenTypes.VECT && (S = S[1])
    return _unique_solution(S - y, GiacExpr(SymNode("c_0")))
end

# the one solution of e = 0 for v, else nothing
function _unique_solution(e::GiacExpr, v::GiacExpr)
    s = try
        invoke_cmd(:solve, e, v)
    catch err
        _stopped(err) && rethrow()
        return nothing
    end
    (giac_type(s) == GenTypes.VECT && length(s) == 1) || return nothing
    return _simplified(s[1])
end

# F₀ for the condition u = g on the line where vars[k] = at: there A F₀(ξ) + B = g, with the free
# variable written by w = ξ
function _fitted_function(line, A, B, ξ, vars, w)
    line.α == (0, 0) || return nothing
    k, z = line.k, vars[3 - line.k]
    at(e) = substitute(e, vars[k] => line.at)
    Z = _unique_solution(at(ξ) - w, z)
    Z === nothing && return nothing
    Fw = _simplified(substitute((line.g - at(B)) / at(A), z => Z))
    return substitute(Fw, w => ξ)
end

# ---------------------------------------------------------------------------------------------
# The heat and wave equations on an interval
# ---------------------------------------------------------------------------------------------

# u_t = κ u_xx or u_tt = κ u_xx on a ≤ x ≤ b with u or u_x zero at each end, and u (with u_t for
# the wave equation) given at t = t₀: u = Σₙ Tₙ(t) Xₙ(x), the Xₙ the eigenfunctions of the ends'
# conditions and the Tₙ from the initial values' coefficients
function _separated_pde(q, vars, lines)
    _is_zero_expr(q.f) || return nothing
    any(c -> !isempty(intersect(_free_names(c), Set(string.(vars)))), values(q.q)) && return nothing
    for s in 1:2                                       # s: the index of the space variable
        τ = 3 - s
        ∂(k, j) = s == 1 ? (k, j) : (j, k)            # u_x^k u_t^j as a multi-index
        Set(keys(q.q)) == Set([∂(2, 0), ∂(0, 1)]) || Set(keys(q.q)) == Set([∂(2, 0), ∂(0, 2)]) || continue
        wave = haskey(q.q, ∂(0, 2))
        κ = _simplified(-q.q[∂(2, 0)] / q.q[wave ? ∂(0, 2) : ∂(0, 1)])
        ends = filter(l -> l.k == s, lines)
        starts = filter(l -> l.k == τ, lines)
        (length(ends) == 2 && length(starts) == (wave ? 2 : 1)) || continue
        r = _separated_solution(κ, wave, vars[s], vars[τ], ends, starts, ∂)
        r === nothing || return r
    end
    return nothing
end

function _separated_solution(κ, wave, x, t, ends, starts, ∂)
    all(l -> _is_zero_expr(l.g) && l.α in (∂(0, 0), ∂(1, 0)), ends) || return nothing
    isequal(_node(ends[1].at), _node(ends[2].at)) && return nothing
    lo, hi = sort(ends; by = l -> to_julia(invoke_cmd(:evalf, l.at)))
    a, L = lo.at, _simplified(hi.at - lo.at)
    neumann = (lo.α != ∂(0, 0), hi.α != ∂(0, 0))      # whether u_x is zero at a, at b
    t0 = starts[1].at
    all(l -> isequal(_node(l.at), _node(t0)), starts) || return nothing
    initial = Dict(l.α => l.g for l in starts)
    haskey(initial, ∂(0, 0)) || return nothing
    wave && (haskey(initial, ∂(0, 1)) || return nothing)
    n = _fresh_symbol(_node(sum(l.g for l in starts) + x + t), "n")
    # Xₙ = sin(ωₙ(x - a)) or cos, with ωₙ = nπ/L or (n + 1/2)π/L, n from 1 or 0
    half = neumann[1] != neumann[2]
    ω = (half ? n + 1 // 2 : n) * convert(GiacExpr, π) / L
    X = (neumann[1] ? cos : sin)(ω * (x - a))
    first_n = neumann[1] || half ? 0 : 1
    norm(m) = neumann == (true, true) && m == 0 ? L : L / 2
    s = t - t0
    # the time factors for u(t₀) and u_t(t₀): e^(-κω²s), or cos(√κ ωs) and sin(√κ ωs)/(√κ ω)
    function factors(m)
        wm = substitute(ω, n => m)
        wave || return (exp(-κ * wm^2 * s), nothing)
        _is_zero_expr(wm) && return (convert(GiacExpr, 1), s)
        r = sqrt(κ) * wm
        return (cos(r * s), sin(r * s) / r)
    end
    givens = wave ? (initial[∂(0, 0)], initial[∂(0, 1)]) : (initial[∂(0, 0)],)
    coefficient(g, m) = _simplified(Commands.integrate(g * substitute(X, n => m), x, a, a + L) / norm(m))
    # the coefficients for a general n, and with sin and cos of whole multiples of π at whole n reduced
    raw = [_simplified(Commands.integrate(g * X, x, a, a + L) / (L / 2)) for g in givens]
    generic = [_at_whole(c, n) for c in raw]
    # the whole n ≥ first_n where the general coefficients do not hold: the poles of their denominators
    poles = Int[]
    for c in raw
        for r in _whole_roots(invoke_cmd(:denom, c), n)
            r >= first_n && push!(poles, r)
        end
    end
    last_explicit = max(first_n - 1, maximum(poles; init = first_n - 1), neumann == (true, true) ? 0 : first_n - 1)
    terms = GiacExpr[]
    for m in first_n:last_explicit
        F = factors(m)
        term = sum(coefficient(g, m) * F[i] for (i, g) in enumerate(givens))
        push!(terms, _exp_powers(_simplified(term * substitute(X, n => m))))
    end
    explicit = isempty(terms) ? convert(GiacExpr, 0) : sum(terms)
    all(c -> _is_zero_expr(_simplified(c)), generic) && return explicit
    F = factors(n)
    # each coefficient factored on its own, times its time factor and the eigenfunction
    term = sum(_at_whole(invoke_cmd(:factor, _simplified(generic[i])), n) * F[i] for i in eachindex(givens)) * X
    return explicit + _hold(:sum, (term, n, last_explicit + 1, giac_eval("inf")))
end

# e at whole n: sin and cos of kπn + β, k whole, written (-1)^(kn) sin β and (-1)^(kn) cos β, and
# exp(ikπn + β) as (-1)^(kn) e^β
function _at_whole(e::GiacExpr, n::GiacExpr)
    π_ = convert(GiacExpr, π)
    iπ = giac_eval("i*pi")
    rewrite(m::Node) = m
    function rewrite(m::CallNode)
        args = Node[rewrite(a) for a in m.args]
        if m.f in ("sin", "cos", "exp") && length(args) == 1 && string(n) in _free_names(GiacExpr(args[1]))
            A = GiacExpr(args[1])
            k = _simplified(invoke_cmd(:diff, A, n) / (m.f == "exp" ? iπ : π_))
            kv = _node(k)
            (kv isa NumNode && kv.v isa Integer) || return _call(m.f, args)
            β = _simplified(A - k * (m.f == "exp" ? iπ : π_) * n)
            sign = iseven(kv.v) ? convert(GiacExpr, 1) : convert(GiacExpr, -1)^n
            return _node(sign * (m.f == "sin" ? sin(β) : m.f == "cos" ? cos(β) : exp(β)))
        end
        return _call(m.f, args)
    end
    rewrite(m::AddNode) = _add((_scale(rewrite(t), c) for (t, c) in m.terms)...)
    rewrite(m::MulNode) = _mul(NumNode(m.coef), (_pow(rewrite(b), rewrite(p)) for (b, p) in m.factors)...)
    return GiacExpr(rewrite(_node(e)))
end

# the whole numbers n where the polynomial p in n vanishes
function _whole_roots(p::GiacExpr, n::GiacExpr)
    string(n) in _free_names(p) || return Int[]
    s = try
        invoke_cmd(:solve, p, n)
    catch e
        _stopped(e) && rethrow()
        return Int[]
    end
    giac_type(s) == GenTypes.VECT || return Int[]
    roots = Int[]
    for i in 1:length(s)
        v = _node(s[i])
        v isa NumNode && v.v isa Integer && push!(roots, Int(v.v))
    end
    return roots
end

# GIAC's simplify
_simplified(e) = invoke_cmd(:simplify, e)
