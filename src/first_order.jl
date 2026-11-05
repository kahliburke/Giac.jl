# Equations in unknown functions of time, as a first-order system: `q(t)` and its derivatives
# below the highest become state variables `q`, `q′`, …, and Giac solves for the highest.

# `(name, t, order)` for `q(t)` and its derivatives `diff(q(t), t, k)`, else `nothing`
function _derivative_of(e::GiacExpr)
    iscall(e) || return nothing
    head, args = symb_funcname(e), arguments(e)
    if head == "of"
        length(args) == 2 && is_identifier(args[1]) && is_identifier(args[2]) || return nothing
        return (string(args[1]), string(args[2]), 0)
    elseif head == "diff" && length(args) in (2, 3)
        d = _derivative_of(args[1])
        (d === nothing || string(args[2]) != d[2]) && return nothing
        return (d[1], d[2], d[3] + (length(args) == 3 ? Int(to_julia(args[3])) : 1))
    end
    return nothing
end

# every unknown function of time and derivative in `e`, as `(name, t, order) => node`
function _function_forms!(found, e::GiacExpr)
    d = _derivative_of(e)
    d === nothing || return push!(found, d => e)
    iscall(e) && foreach(a -> _function_forms!(found, a), arguments(e))
    return found
end
_function_forms(e::GiacExpr) = _function_forms!(Pair{Tuple{String, String, Int}, GiacExpr}[], e)

# the variable standing for the `k`-th derivative of `name`: q, q′, q″, …
_state_name(name, k) = name * "′"^k
_state(name, k) = giac_identifier(_state_name(name, k))

# `e` with each unknown function and derivative replaced by its variable (derivatives first, as
# `q(t)` occurs inside them)
function _as_states(e::GiacExpr, forms = _function_forms(e))
    for (d, node) in sort(unique(first, forms); by = f -> -f.first[3])
        e = Commands.subst(e, node ~ _state(d[1], d[3]))
    end
    return e
end

"""
    FirstOrder

Equations in unknown functions of time as a first-order system, from [`first_order`](@ref):
`states` are the unknown functions and their derivatives below the highest, `q`, `q′`, …,
followed by the unknowns that appear without derivatives; `rhs` gives the time derivative of
each of the first `differential` states, and for each other one a residual that is zero.
`derived` gives each highest derivative, by name, as solved for.
"""
struct FirstOrder
    t::GiacExpr
    states::Vector{GiacExpr}
    rhs::Vector{GiacExpr}
    differential::Int
    derived::Dict{String, GiacExpr}
end

"""
    first_order(eqs; t = nothing) -> FirstOrder

The equations `eqs`, in unknown functions of time (`@giac_var q(t)`) and their derivatives
(`Differential(t)`), as a first-order system. The states are each function and its derivatives
below the highest order in which it appears, written `q`, `q′`, `q″`; the equations with
derivatives are solved by Giac for the highest ones, so they must be linear in them. Equations
without derivatives are algebraic constraints on the unknowns that appear without derivatives.

```julia
@giac_var t q(t) c k
D = Differential(t)
first_order([D(D(q)) + c * D(q) + k * sin(q) ~ 0])
# d/dt [q, q′] = [q′, -c q′ - k sin q]
```
"""
function first_order(eqs::AbstractVector; t = nothing)
    residuals = GiacExpr[lhs(e) - rhs(e) for e in eqs]
    forms = reduce(vcat, map(_function_forms, residuals))
    isempty(forms) && throw(ArgumentError("the equations have no unknown functions of time, such as q(t) from `@giac_var q(t)`"))
    times = unique(f.first[2] for f in forms)
    length(times) == 1 || throw(ArgumentError("the unknown functions have different arguments: $(join(times, ", "))"))
    t === nothing || string(t) == only(times) ||
        throw(ArgumentError("the unknown functions are functions of $(only(times)), not $t"))
    names = unique(f.first[1] for f in forms)
    order = Dict(n => maximum(f.first[3] for f in forms if f.first[1] == n) for n in names)
    rs = [_as_states(r, forms) for r in residuals]

    dynamic = [n for n in names if order[n] > 0]
    algebraic = [n for n in names if order[n] == 0]
    highest = GiacExpr[_state(n, order[n]) for n in dynamic]
    hnames = Set(string.(highest))
    has_highest(r) = any(s -> string(s) in hnames, free_symbols(r))
    deqs, aeqs = filter(has_highest, rs), filter(!has_highest, rs)
    length(deqs) == length(dynamic) ||
        throw(ArgumentError("$(length(deqs)) equations with derivatives for the $(length(dynamic)) unknowns $(join(dynamic, ", "))"))
    length(aeqs) == length(algebraic) ||
        throw(ArgumentError("$(length(aeqs)) equations without derivatives for the $(length(algebraic)) unknowns $(join(algebraic, ", ")) that appear without them"))
    solved = try
        isempty(deqs) ? GiacExpr[] : collect(GiacExpr, Commands.linsolve(deqs, highest))
    catch err
        err isa GiacError || rethrow()
        GiacExpr[]
    end
    (length(solved) == length(highest) && !any(has_highest, solved)) ||
        throw(ArgumentError("the equations do not determine $(join(highest, ", ")): they should be linear in the highest derivatives, with a unique solution"))

    states, F = GiacExpr[], GiacExpr[]
    for (n, s) in zip(dynamic, solved), k in 0:order[n] - 1
        push!(states, _state(n, k))
        push!(F, k < order[n] - 1 ? _state(n, k + 1) : s)
    end
    differential = length(states)
    append!(states, [_state(n, 0) for n in algebraic])
    append!(F, aeqs)
    return FirstOrder(giac_identifier(only(times)), states, F, differential,
                      Dict(string(h) => s for (h, s) in zip(highest, solved)))
end

function Base.show(io::IO, s::FirstOrder)
    n = s.differential
    print(io, "d/d", s.t, " [", join(s.states[1:n], ", "), "] = [", join(s.rhs[1:n], ", "), "]")
    n < length(s.states) && print(io, ", 0 = [", join(s.rhs[n+1:end], ", "), "]")
end

function Base.show(io::IO, ::MIME"text/latex", s::FirstOrder)
    n = s.differential
    column(es) = "\\begin{bmatrix}" * join(giac_latex.(es), " \\\\ ") * "\\end{bmatrix}"
    print(io, "\$\\frac{d}{d", giac_latex(s.t), "}", column(s.states[1:n]), " = ", column(s.rhs[1:n]))
    n < length(s.states) && print(io, ", \\qquad 0 = ", column(s.rhs[n+1:end]))
    print(io, "\$")
end
