# ---------------------------------------------------------------------------
# GiacSciMLBaseExt: SciML problems from symbolic models.
#
# A model is either a vector of right-hand sides in state variables, or a vector of equations
# in unknown functions of time (`D(D(q)) + c*D(q) + sin(q) ~ u`), brought to first order with the
# highest derivatives solved for by Giac. It has parameters, possibly a time variable, and
# inputs: expressions substituted into it (a feedback law) or Julia functions of time or of the
# state (a reference signal, a controller). The right-hand side and its exact derivatives are
# translated to Julia code and compiled by `Giac._runtime_function`, so solvers call them as
# ordinary Julia functions. The problem carries a symbolic index (`GiacSystem`), so solutions
# and problems are indexed by the model's variables and expressions in them.
# ---------------------------------------------------------------------------

module GiacSciMLBaseExt

using Giac
using Giac: GiacExpr, GiacFunction, DerivativeExpr
import SciMLBase
using SciMLBase: ODEFunction, ODEProblem, NonlinearFunction, NonlinearProblem, NullParameters
import SciMLBase.SymbolicIndexingInterface as SII
using LinearAlgebra: Diagonal

const C = Giac.Commands
const _Vars = Union{GiacExpr, AbstractArray{GiacExpr}}

_vector(x::GiacExpr) = GiacExpr[x]
_vector(x::AbstractArray) = GiacExpr[v for v in vec(x)]
_vector(::Nothing) = GiacExpr[]
_vector(x::Tuple) = GiacExpr[v for v in x]

_giac(v::GiacExpr) = v
_giac(v::DerivativeExpr) = Giac._to_giac_expr(v)

# ---------------------------------------------------------------------------
# Unknown functions of time
# ---------------------------------------------------------------------------

# `(name, t, order)` for `q(t)` and its derivatives `diff(q(t), t, k)`, else `nothing`
function _derivative_of(e::GiacExpr)
    Giac.iscall(e) || return nothing
    head, args = Giac.symb_funcname(e), Giac.arguments(e)
    if head == "of"
        length(args) == 2 && Giac.is_identifier(args[1]) && Giac.is_identifier(args[2]) || return nothing
        return (string(args[1]), string(args[2]), 0)
    elseif head == "diff" && length(args) in (2, 3)
        d = _derivative_of(args[1])
        (d === nothing || string(args[2]) != d[2]) && return nothing
        return (d[1], d[2], d[3] + (length(args) == 3 ? Int(Giac.to_julia(args[3])) : 1))
    end
    return nothing
end

# every unknown function of time and derivative in `e`, as `(name, t, order) => node`
function _function_forms!(found, e::GiacExpr)
    d = _derivative_of(e)
    d === nothing || return push!(found, d => e)
    Giac.iscall(e) && foreach(a -> _function_forms!(found, a), Giac.arguments(e))
    return found
end
_function_forms(e) = _function_forms!(Pair{Tuple{String, String, Int}, GiacExpr}[], e)

# the variable standing for the `k`-th derivative of `name`: q, q′, q″, …
_state_name(name, k) = name * "′"^k
_state(name, k) = Giac.giac_identifier(_state_name(name, k))

# `e` with each unknown function and derivative replaced by its variable (derivatives first, as
# `q(t)` occurs inside them)
function _as_states(e::GiacExpr, forms = _function_forms(e))
    for (d, node) in sort(unique(first, forms); by = f -> -f.first[3])
        e = C.subst(e, node ~ _state(d[1], d[3]))
    end
    return e
end

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

"A Julia function giving the values of input variables, from the time or from `(u, p, t)`."
struct _Input
    vars::Vector{GiacExpr}
    f::Any
    state::Bool
end

struct _Model
    F::Vector{GiacExpr}                 # first-order right-hand sides (algebraic residuals last)
    states::Vector{GiacExpr}
    params::Vector{GiacExpr}
    t::Union{Nothing, GiacExpr}
    inputs::Vector{_Input}
    mass::Any                           # the mass matrix of a DAE, else `nothing`
    derived::Dict{String, GiacExpr}     # highest derivatives, by name, as solved for
end

_input_vars(m::_Model) = GiacExpr[v for i in m.inputs for v in i.vars]
_all_vars(m::_Model) = vcat(m.states, m.params, m.t === nothing ? GiacExpr[] : [m.t], _input_vars(m))

const _Symbolic = Union{GiacExpr, DerivativeExpr, Number}

_pairs(x::Pair) = Any[x]
_pairs(x::Union{AbstractVector, Tuple}) = collect(Any, x)

# Inputs given as expressions, to substitute into the model, and inputs given as Julia functions
function _inputs(inputs)
    subs, fs = Pair{GiacExpr, Any}[], _Input[]
    for (k, v) in _pairs(inputs)
        vars = _vector(k)
        if v isa _Symbolic || v isa AbstractArray{<:_Symbolic}
            vals = v isa AbstractArray ? vec(collect(v)) : [v]
            length(vals) == length(vars) ||
                throw(DimensionMismatch("$(length(vals)) values for the inputs $(join(vars, ", "))"))
            append!(subs, [a => (b isa DerivativeExpr ? _giac(b) : b) for (a, b) in zip(vars, vals)])
        else
            push!(fs, _Input(vars, v, !applicable(v, 0.0)))
        end
    end
    return subs, fs
end

_substitute(e::GiacExpr, subs) = isempty(subs) ? e : substitute(e, subs...)

function _check_free(exprs, m::_Model)
    bound = Set(string.(_all_vars(m)))
    free = Set(string(s) for e in exprs for s in Giac.free_symbols(e))
    unbound = setdiff(free, bound, ("pi", "infinity"))
    isempty(unbound) || throw(ArgumentError("the model has variables that are neither states, parameters, inputs nor time: $(join(sort!(collect(unbound)), ", ")). Give them as parameters, or `t = …` for time"))
    return nothing
end

function _vector_model(F, states, params, t, inputs)
    subs, fs = _inputs(inputs)
    length(F) == length(states) ||
        throw(DimensionMismatch("the model has $(length(F)) right-hand sides for $(length(states)) states"))
    m = _Model(GiacExpr[_substitute(f, subs) for f in F], states, params, t, fs, nothing, Dict{String, GiacExpr}())
    _check_free(m.F, m)
    return m
end

_is_equations(F) = !isempty(F) && all(Giac._is_equation, F)

# Equations in unknown functions of time, to first order: q and its derivatives below the highest
# become states, and the equations with derivatives are solved for the highest ones. Equations
# without derivatives are algebraic, for unknowns that appear without derivatives (a DAE).
function _equation_model(eqs, params, t, inputs)
    subs, fs = _inputs(inputs)
    residuals = GiacExpr[_substitute(Giac.lhs(e) - Giac.rhs(e), subs) for e in eqs]
    forms = reduce(vcat, map(_function_forms, residuals))
    isempty(forms) && throw(ArgumentError("the equations have no unknown functions of time, such as q(t) from `@giac_var q(t)`"))
    times = unique(f.first[2] for f in forms)
    length(times) == 1 || throw(ArgumentError("the unknown functions have different arguments: $(join(times, ", "))"))
    t === nothing || string(t) == only(times) ||
        throw(ArgumentError("the unknown functions are functions of $(only(times)), not $t"))
    t = Giac.giac_identifier(only(times))
    names = unique(f.first[1] for f in forms)
    order = Dict(n => maximum(f.first[3] for f in forms if f.first[1] == n) for n in names)
    rs = [_as_states(r, forms) for r in residuals]

    dynamic = [n for n in names if order[n] > 0]
    algebraic = [n for n in names if order[n] == 0]
    highest = GiacExpr[_state(n, order[n]) for n in dynamic]
    hnames = Set(string.(highest))
    has_highest(r) = any(s -> string(s) in hnames, Giac.free_symbols(r))
    deqs, aeqs = filter(has_highest, rs), filter(!has_highest, rs)
    length(deqs) == length(dynamic) ||
        throw(ArgumentError("$(length(deqs)) equations with derivatives for the $(length(dynamic)) unknowns $(join(dynamic, ", "))"))
    length(aeqs) == length(algebraic) ||
        throw(ArgumentError("$(length(aeqs)) equations without derivatives for the $(length(algebraic)) unknowns $(join(algebraic, ", ")) that appear without them"))
    solved = try
        isempty(deqs) ? GiacExpr[] : collect(GiacExpr, C.linsolve(deqs, highest))
    catch err
        err isa Giac.GiacError || rethrow()
        GiacExpr[]
    end
    (length(solved) == length(highest) && !any(has_highest, solved)) ||
        throw(ArgumentError("the equations do not determine $(join(highest, ", ")): they should be linear in the highest derivatives, with a unique solution"))

    states, F = GiacExpr[], GiacExpr[]
    for (n, s) in zip(dynamic, solved), k in 0:order[n] - 1
        push!(states, _state(n, k))
        push!(F, k < order[n] - 1 ? _state(n, k + 1) : s)
    end
    append!(states, [_state(n, 0) for n in algebraic])
    append!(F, aeqs)
    mass = isempty(algebraic) ? nothing :
           Diagonal(vcat(ones(length(F) - length(aeqs)), zeros(length(aeqs))))
    derived = Dict(string(h) => s for (h, s) in zip(highest, solved))
    m = _Model(F, states, params, t, fs, mass, derived)
    _check_free(m.F, m)
    return m
end

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

# The functions' arguments: `out`, `u`, `p`, `t` and `inputs`, renamed where the model has a
# variable of that name. A time variable of the model is the time argument itself.
function _arguments(m::_Model)
    taken = Set(string.(_all_vars(m)))
    fresh(n) = (while n in taken; n *= "_"; end; Symbol(n))
    return (out = fresh("out"), u = fresh("u"), p = fresh("p"),
            t = m.t === nothing ? fresh("t") : Symbol(string(m.t)), inputs = fresh("inputs"))
end

# The model's states, parameters and inputs, bound to the arguments
function _bindings(a, m::_Model)
    b = Expr[]
    for (i, v) in enumerate(m.states)
        push!(b, :($(Symbol(string(v))) = $(a.u)[$i]))
    end
    for (i, v) in enumerate(m.params)
        push!(b, :($(Symbol(string(v))) = $(a.p)[$i]))
    end
    for (i, input) in enumerate(m.inputs)
        call = input.state ? :($(a.inputs)[$i]($(a.u), $(a.p), $(a.t))) : :($(a.inputs)[$i]($(a.t)))
        names = Symbol.(string.(input.vars))
        push!(b, length(names) == 1 ? :($(only(names)) = $call) : :($(Expr(:tuple, names...)) = $call))
    end
    return b
end

_is_zero(c) = c isa Number && iszero(c)

_codes(exprs) = map(Giac.to_expr, exprs)

# Julia code for each entry of `∂F/∂vars`. An entry is differentiated only where its row depends
# on the variable, so a sparse model's Jacobian costs its nonzeros.
function _derivatives(F, vars)
    column = Dict(string(v) => j for (j, v) in enumerate(vars))
    codes = Any[0 for _ in eachindex(F), _ in eachindex(vars)]
    for (i, f) in enumerate(F), s in Giac.free_symbols(f)
        j = get(column, string(s), nothing)
        j === nothing || (codes[i, j] = Giac.to_expr(C.diff(f, vars[j])))
    end
    return codes
end
_derivatives(F, t::GiacExpr) = vec(_derivatives(F, GiacExpr[t]))

"A generated function, given the model's Julia inputs as its last argument."
struct _WithInputs{F, I} <: Function
    f::F
    inputs::I
end
(w::_WithInputs)(a, b, c) = w.f(a, b, c, w.inputs)
(w::_WithInputs)(a, b, c, d) = w.f(a, b, c, d, w.inputs)

# `out = codes(u, p, t)` in place, or `codes(u, p, t)` as a new array, with subexpressions they
# share computed once. A zero entry is not written in place: it is set by `fill!` for a dense
# array, and is not stored in a sparse prototype.
function _generate(codes::AbstractArray, m::_Model; inplace::Bool, dense_zeros::Bool = true)
    a = _arguments(m)
    taken = union(Set(Symbol.(string.(_all_vars(m)))), values(a))
    assignments, outs = Giac._common_subexpressions(codes; taken)
    prelude = vcat(_bindings(a, m), assignments)
    if inplace
        writes = [:($(a.out)[$(Tuple(I)...)] = $(outs[I])) for I in CartesianIndices(codes) if !_is_zero(codes[I])]
        fill = dense_zeros && any(_is_zero, codes) ? [:(fill!($(a.out), zero(eltype($(a.out)))))] : Expr[]
        body = Expr(:block, prelude..., fill..., Expr(:macrocall, Symbol("@inbounds"), nothing, Expr(:block, writes...)),
                    :(return nothing))
        args = Symbol[a.out, a.u, a.p, a.t]
    else
        entries = Expr(:vect, vec(outs)...)
        body = Expr(:block, prelude..., codes isa AbstractVector ? entries : :(reshape($entries, $(size(codes)))))
        args = Symbol[a.u, a.p, a.t]
    end
    isempty(m.inputs) && return Giac._runtime_function(args, body)
    return _WithInputs(Giac._runtime_function(push!(args, a.inputs), body), Tuple(i.f for i in m.inputs))
end

# A nonlinear problem's functions take `(u, p)`: the same code with no time argument
_drop_time(f, inplace) = inplace ? ((out, u, p) -> f(out, u, p, nothing)) : ((u, p) -> f(u, p, nothing))

_sparse_prototype(J) = Giac._sparse_pattern(map(!_is_zero, J))

# ---------------------------------------------------------------------------
# Symbolic indexing
# ---------------------------------------------------------------------------

"""
The variables of a model built from Giac expressions, for SymbolicIndexingInterface: states,
parameters and the time variable, found by name. An unknown function of time `q` is the state
`q`, and its derivative `D(q)` the state `q′`. Any other expression in them, an input or a
highest derivative is an observed quantity, compiled on first use.
"""
struct GiacSystem
    model::_Model
    timed::Bool                         # an ODE's system, time-dependent even with no time variable
    state_index::Dict{String, Int}
    param_index::Dict{String, Int}
    observed::Dict{String, Any}
end
GiacSystem(m::_Model; timed) =
    GiacSystem(m, timed, Dict(string(v) => i for (i, v) in enumerate(m.states)),
               Dict(string(v) => i for (i, v) in enumerate(m.params)), Dict{String, Any}())

function Base.show(io::IO, s::GiacSystem)
    m = s.model
    print(io, "GiacSystem(states = [", join(m.states, ", "), "], parameters = [", join(m.params, ", "), "]")
    isempty(m.inputs) || print(io, ", inputs = [", join(_input_vars(m), ", "), "]")
    m.t === nothing || print(io, ", t = ", m.t)
    print(io, ")")
end

SII.symbolic_type(::Type{GiacExpr}) = SII.ScalarSymbolic()
SII.symbolic_type(::Type{DerivativeExpr}) = SII.ScalarSymbolic()
SII.hasname(e::GiacExpr) = _name(e) !== nothing
SII.getname(e::GiacExpr) = Symbol(something(_name(e), string(e)))
SII.hasname(::DerivativeExpr) = true
SII.getname(d::DerivativeExpr) = Symbol(_name(d))

# a variable's name (`q′` for the derivative of `q(t)`), or `nothing` for an index or an
# expression that is not a variable
function _name(sym::GiacExpr)
    Giac.is_identifier(sym) && return string(sym)
    d = _derivative_of(sym)
    return d === nothing ? nothing : _state_name(d[1], d[3])
end
_name(d::DerivativeExpr) = _state_name(d.funcname, sum(last, d.steps))
_name(sym::Symbol) = string(sym)
_name(sym) = nothing

_index(d, sym) = (n = _name(sym); n === nothing ? nothing : get(d, n, nothing))

SII.is_variable(s::GiacSystem, sym) = sym isa Integer ? sym in eachindex(s.model.states) : _index(s.state_index, sym) !== nothing
SII.variable_index(s::GiacSystem, sym) = sym isa Integer ? sym : _index(s.state_index, sym)
SII.variable_symbols(s::GiacSystem) = s.model.states
SII.is_parameter(s::GiacSystem, sym) = sym isa Integer ? sym in eachindex(s.model.params) : _index(s.param_index, sym) !== nothing
SII.parameter_index(s::GiacSystem, sym) = sym isa Integer ? sym : _index(s.param_index, sym)
SII.parameter_symbols(s::GiacSystem) = s.model.params
SII.is_independent_variable(s::GiacSystem, sym) = s.model.t !== nothing && _name(sym) == string(s.model.t)
SII.independent_variable_symbols(s::GiacSystem) = s.model.t === nothing ? GiacExpr[] : [s.model.t]
SII.is_time_dependent(s::GiacSystem) = s.timed
SII.constant_structure(::GiacSystem) = true
SII.all_variable_symbols(s::GiacSystem) = s.model.states
SII.all_symbols(s::GiacSystem) = vcat(s.model.states, s.model.params, SII.independent_variable_symbols(s))
SII.default_values(::GiacSystem) = Dict()

SII.is_observed(s::GiacSystem, sym::Union{GiacExpr, DerivativeExpr}) =
    !(SII.is_variable(s, sym) || SII.is_parameter(s, sym) || SII.is_independent_variable(s, sym))
SII.is_observed(::GiacSystem, sym) = false

# `(u, p, t) -> value` for a time-dependent system, else `(u, p) -> value`. A highest derivative
# is its solved expression.
function SII.observed(s::GiacSystem, sym::Union{GiacExpr, DerivativeExpr})
    get!(s.observed, string(sym)) do
        m = s.model
        e = _as_states(_giac(sym))
        isempty(m.derived) || (e = substitute(e, (Giac.giac_identifier(k) => v for (k, v) in m.derived)...))
        _check_free((e,), m)
        f = _generate(Any[Giac.to_expr(e)], m; inplace = false)
        s.timed ? ((u, p, t) -> f(u, p, t)[1]) : ((u, p) -> f(u, p, nothing)[1])
    end
end

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# An input given as a Julia function has no symbolic derivatives: one of the state makes the
# Jacobians unknown, and one of time the time gradient. A solver then differentiates the
# compiled right-hand side itself.
function _ode_function(m::_Model; jac::Bool = true, tgrad::Bool = true, paramjac::Bool = true,
                       sparse::Bool = false, inplace::Bool = true, kwargs...)
    by_state = any(i -> i.state, m.inputs)
    fkw = Dict{Symbol, Any}(:sys => GiacSystem(m; timed = true))
    if jac && !by_state
        J = _derivatives(m.F, m.states)
        fkw[:jac] = _generate(J, m; inplace, dense_zeros = !sparse)
        sparse && (fkw[:jac_prototype] = _sparse_prototype(J))
    end
    if tgrad && m.t !== nothing && isempty(m.inputs)
        fkw[:tgrad] = _generate(_derivatives(m.F, m.t), m; inplace)
    end
    if paramjac && !isempty(m.params) && !by_state
        fkw[:paramjac] = _generate(_derivatives(m.F, m.params), m; inplace)
    end
    m.mass === nothing || (fkw[:mass_matrix] = m.mass)
    return ODEFunction{inplace, SciMLBase.FullSpecialize}(_generate(_codes(m.F), m; inplace); fkw..., kwargs...)
end

"""
    ODEFunction(F, x; p = [], t = nothing, inputs = [], jac = true, tgrad = true,
                paramjac = true, sparse = false, inplace = true, kwargs...)
    ODEFunction(eqs; p = [], inputs = [], kwargs...)
    ODEFunction(F::GiacFunction; kwargs...)

The SciML function of the model `ẋ = F(x, p, t)`, with `F` a vector of Giac expressions in the
states `x`, the parameters `p` and the time variable `t`; or of the equations `eqs` in unknown
functions of time (see [`ODEProblem`](@ref)). The right-hand side and, by default, its exact
Jacobian `∂F/∂x`, time gradient `∂F/∂t` and parameter Jacobian `∂F/∂p` are compiled to native
code. `sparse = true` (with SparseArrays loaded) gives the Jacobian as a sparse matrix with the
structure of the symbolic one. Other keywords (`mass_matrix`, …) go to `SciMLBase.ODEFunction`.

`inputs` assigns the model's input variables: `u => -k * x1` substitutes an expression (a
feedback law, part of the exact Jacobian), and `u => t -> sin(t)` or `u => (x, p, t) -> …` a
Julia function of time or of the state, called by the compiled code. A Julia function has no
symbolic derivatives, so with one the solver differentiates the parts it affects.

A function from `@giac_function` takes its arguments as SciML's `(u, p, t)`: the states, then
optionally the parameters and the time variable.
"""
function SciMLBase.ODEFunction(F::AbstractVector{GiacExpr}, x::_Vars; p = GiacExpr[], t = nothing,
                               inputs = (), kwargs...)
    return _ode_function(_vector_model(F, _vector(x), _vector(p), t, inputs); kwargs...)
end

function SciMLBase.ODEFunction(eqs::AbstractVector{GiacExpr}; p = GiacExpr[], t = nothing, inputs = (), kwargs...)
    _is_equations(eqs) || throw(ArgumentError("give the states of a vector of right-hand sides: ODEFunction(F, x)"))
    return _ode_function(_equation_model(eqs, _vector(p), t, inputs); kwargs...)
end

function SciMLBase.ODEFunction(F::GiacFunction; kwargs...)
    u, p, t = _uptime(F)
    return ODEFunction(_body_vector(F), u; p, t, kwargs...)
end

function _uptime(F::GiacFunction)
    1 <= length(F.args) <= 3 ||
        throw(ArgumentError("$(F.name) should take SciML's arguments (u, p, t): the states, then optionally the parameters and the time variable"))
    u = F.args[1]
    p = length(F.args) >= 2 ? _vector(F.args[2]) : GiacExpr[]
    t = length(F.args) == 3 ? F.args[3] : nothing
    t === nothing || t isa GiacExpr || throw(ArgumentError("the time argument of $(F.name) should be one variable"))
    return u, p, t
end

_body_vector(F::GiacFunction) = F.body isa AbstractVector ? F.body : GiacExpr[F.body]

"""
    NonlinearFunction(F, x; p = [], inputs = [], jac = true, sparse = false, inplace = true, kwargs...)

The SciML function of the system `F(x, p) = 0`, with its exact Jacobian `∂F/∂x`. An equation
`a ~ b` in `F` stands for `a - b = 0`, and `inputs` may assign expressions to input variables.
"""
function SciMLBase.NonlinearFunction(F::AbstractVector{GiacExpr}, x::_Vars; p = GiacExpr[], inputs = (),
                                     jac::Bool = true, sparse::Bool = false, inplace::Bool = true, kwargs...)
    m = _vector_model(map(_residual, F), _vector(x), _vector(p), nothing, inputs)
    isempty(m.inputs) || throw(ArgumentError("a nonlinear system's inputs are expressions, not functions"))
    fkw = Dict{Symbol, Any}(:sys => GiacSystem(m; timed = false))
    if jac
        J = _derivatives(m.F, m.states)
        fkw[:jac] = _drop_time(_generate(J, m; inplace, dense_zeros = !sparse), inplace)
        sparse && (fkw[:jac_prototype] = _sparse_prototype(J))
    end
    f = _drop_time(_generate(_codes(m.F), m; inplace), inplace)
    return NonlinearFunction{inplace, SciMLBase.FullSpecialize}(f; fkw..., kwargs...)
end

function SciMLBase.NonlinearFunction(F::GiacFunction; kwargs...)
    u, p, t = _uptime(F)
    t === nothing || throw(ArgumentError("a nonlinear system takes (u, p), without time"))
    return NonlinearFunction(_body_vector(F), u; p, kwargs...)
end

_residual(e::GiacExpr) = Giac._is_equation(e) ? Giac.lhs(e) - Giac.rhs(e) : e

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

# Variables and their values from `x => values` or `[x1 => v1, x2 => v2]`
_assignment(a::Pair) = (_vector(first(a)), _values(last(a)))
_assignment(a::AbstractVector{<:Pair}) = (Any[first(q) for q in a], _values([last(q) for q in a]))
_assignment(a::Tuple{}) = (GiacExpr[], Float64[])
_assignment(::NullParameters) = (GiacExpr[], Float64[])
_assignment(a) = throw(ArgumentError("values are given as `x => values` or `[x1 => v1, x2 => v2, …]`, got $(typeof(a))"))

_value(v::GiacExpr) = Giac.to_julia(C.evalf(v))
_value(v) = v
_values(v) = (w = map(_value, vec(collect(v))); isempty(w) ? Float64[] : float.(collect(promote(w...))))

# Keywords that build the function; the rest are the problem's
const _FUNCTION_KEYWORDS = (:jac, :tgrad, :paramjac, :sparse, :inplace, :mass_matrix, :inputs)

function _split(kwargs)
    fkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if k in _FUNCTION_KEYWORDS)
    pkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if !(k in _FUNCTION_KEYWORDS))
    return fkw, pkw
end

"""
    ODEProblem(F, u0, tspan, p = (); t = nothing, inputs = [], kwargs...)
    ODEProblem(eqs, u0, tspan, p = (); inputs = [], kwargs...)

The initial-value problem `ẋ = F(x, p, t)`, `x(t₀) = u0`, with the states and parameters taken
from the assignments `u0` and `p`, each written `x => values` or `[x1 => v1, x2 => v2]`:

```julia
@giac_var x[1:2] g ℓ c
F = [x2, -g / ℓ * sin(x1) - c * x2]
prob = ODEProblem(F, [x1 => π / 2, x2 => 0], (0, 10), [g => 9.81, ℓ => 1, c => 0.2])
sol = solve(prob, Rodas5P())
sol[x1^2 + x2^2]
```

A model may instead be equations in unknown functions of time, of any order. The states are the
functions and their derivatives below the highest, which Giac solves for; initial values are
given for them, by the functions and derivatives themselves:

```julia
@giac_var t q(t) c u
D = Differential(t)
eqs = [D(D(q)) + c * D(q) + sin(q) ~ u]
prob = ODEProblem(eqs, [q => 1, D(q) => 0], (0, 10), [c => 0.2]; inputs = [u => t -> 0.1sin(t)])
sol[D(q)], sol[D(D(q))]
```

Equations without derivatives are algebraic constraints on unknowns that appear without
derivatives, and make the problem a DAE with a mass matrix. `inputs` is described under
[`ODEFunction`](@ref). `F` may also be a function from `@giac_function` with arguments
`(u, p, t)`. The keywords of `ODEFunction` build the function; the others go to
`SciMLBase.ODEProblem`.
"""
function SciMLBase.ODEProblem(F::AbstractVector{GiacExpr}, u0, tspan, p = (); t = nothing, kwargs...)
    params, pv = _assignment(p)
    fkw, pkw = _split(kwargs)
    inputs = pop!(fkw, :inputs, ())
    if _is_equations(F)
        m = _equation_model(F, _vector(params), t, inputs)
        u = last(_ordered(u0, m.states))
    else
        states, u = _assignment(u0)
        m = _vector_model(F, _vector(states), _vector(params), t, inputs)
    end
    return ODEProblem(_ode_function(m; fkw...), u, _tspan(tspan), pv; pkw...)
end

function SciMLBase.ODEProblem(F::GiacFunction, u0, tspan, p = (); kwargs...)
    u, params, t = _uptime(F)
    return ODEProblem(_body_vector(F), _ordered(u0, _vector(u)), tspan, _ordered(p, params); t, kwargs...)
end

"""
    NonlinearProblem(F, u0, p = (); inputs = [], kwargs...)

The system `F(x, p) = 0` with initial guess `u0`, written `x => values` or `[x1 => v1, …]`, as
for [`ODEProblem`](@ref).
"""
function SciMLBase.NonlinearProblem(F::AbstractVector{GiacExpr}, u0, p = (); kwargs...)
    states, u = _assignment(u0)
    params, pv = _assignment(p)
    fkw, pkw = _split(kwargs)
    f = NonlinearFunction(F, _vector(states); p = _vector(params), fkw...)
    return NonlinearProblem(f, u, pv; pkw...)
end

function SciMLBase.NonlinearProblem(F::GiacFunction, u0, p = (); kwargs...)
    u, params, _ = _uptime(F)
    return NonlinearProblem(_body_vector(F), _ordered(u0, _vector(u)), _ordered(p, params); kwargs...)
end

_vector(x::AbstractVector{Any}) = GiacExpr[_giac(v) for v in x]

# The values of `vars` in an assignment, in their order, matched by name (`D(q)` names the
# state `q′`). An assignment may also be a plain vector of values in that order.
_ordered(a::AbstractVector{<:Number}, vars) = vars => a
_ordered(a::Tuple{}, vars) = isempty(vars) ? () : throw(ArgumentError("no values given for $(join(vars, ", "))"))
_ordered(a::NullParameters, vars) = _ordered((), vars)
function _ordered(a, vars)
    keys_, vals = _assignment(a)
    given = Dict(something(_name(k), string(k)) => v for (k, v) in zip(keys_, vals))
    missing_ = [v for v in vars if !haskey(given, string(v))]
    isempty(missing_) || throw(ArgumentError("no values given for $(join(missing_, ", "))"))
    extra = setdiff(keys(given), string.(vars))
    isempty(extra) || throw(ArgumentError("$(join(sort!(collect(extra)), ", ")) not among the model's variables"))
    return vars => [given[string(v)] for v in vars]
end

_tspan(t::Tuple) = promote(map(float ∘ _value, t)...)
_tspan(t) = t

end # module
