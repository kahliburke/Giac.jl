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

# Equations in unknown functions of time, through `first_order`: unknowns that appear without
# derivatives make it a DAE, with a mass matrix
function _equation_model(eqs, params, t, inputs)
    subs, fs = _inputs(inputs)
    fo = first_order([_substitute(Giac.lhs(e), subs) ~ _substitute(Giac.rhs(e), subs) for e in eqs]; t)
    n = length(fo.states)
    mass = fo.differential == n ? nothing : Diagonal([i <= fo.differential ? 1.0 : 0.0 for i in 1:n])
    m = _Model(fo.rhs, fo.states, params, fo.t, fs, mass, fo.derived)
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
            t = m.t === nothing ? fresh("t") : Symbol(string(m.t)), inputs = fresh("inputs"), v = fresh("v"),
            stored = fresh("stored"))
end

# The model's states, parameters and inputs that the code uses, bound to the arguments
function _bindings(a, m::_Model, used)
    b = Expr[]
    for (i, v) in enumerate(m.states)
        Symbol(string(v)) in used && push!(b, :($(Symbol(string(v))) = $(a.u)[$i]))
    end
    for (i, v) in enumerate(m.params)
        Symbol(string(v)) in used && push!(b, :($(Symbol(string(v))) = $(a.p)[$i]))
    end
    for (i, input) in enumerate(m.inputs)
        names = Symbol.(string.(input.vars))
        any(in(used), names) || continue
        call = input.state ? :($(a.inputs)[$i]($(a.u), $(a.p), $(a.t))) : :($(a.inputs)[$i]($(a.t)))
        push!(b, length(names) == 1 ? :($(only(names)) = $call) : :($(Expr(:tuple, names...)) = $call))
    end
    return b
end

# The bindings and shared subexpressions of `codes`, and the codes in terms of them
function _prelude(codes::AbstractArray, m::_Model, a)
    taken = union(Set(Symbol.(string.(_all_vars(m)))), values(a))
    assignments, outs = Giac._common_subexpressions(codes; taken)
    used = Giac._symbols!(Giac._symbols!(Set{Symbol}(), assignments), outs)
    return vcat(_bindings(a, m, used), assignments), outs
end

# `(args...) -> body`, given the model's Julia inputs as a last argument
function _function(args, body, m::_Model, a)
    isempty(m.inputs) && return Giac._runtime_function(args, body)
    return _WithInputs(Giac._runtime_function(push!(args, a.inputs), body), Tuple(i.f for i in m.inputs))
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

# Code for the product of the Jacobian `J`'s code with the vector argument `v`: `J * v`, or
# `transpose(J) * v`, from the nonzero entries
function _products(J, v::Symbol; transpose::Bool = false)
    K = transpose ? permutedims(J) : J
    map(eachrow(K)) do row
        terms = [c == 1 ? :($v[$j]) : :($c * $v[$j]) for (j, c) in enumerate(row) if !_is_zero(c)]
        isempty(terms) ? 0 : length(terms) == 1 ? only(terms) : Expr(:call, :+, terms...)
    end
end

"A generated function, given the model's Julia inputs as its last argument."
struct _WithInputs{F, I} <: Function
    f::F
    inputs::I
end
(w::_WithInputs)(a, b, c) = w.f(a, b, c, w.inputs)
(w::_WithInputs)(a, b, c, d) = w.f(a, b, c, d, w.inputs)
(w::_WithInputs)(a, b, c, d, e) = w.f(a, b, c, d, e, w.inputs)

# `out = codes(u, p, t)` in place, or `codes(u, p, t)` as a new array of the kind of `u`, with
# subexpressions they share computed once. A zero entry is not written in place: it is set by
# `fill!` for a dense array. With `sparse`, `out` has the structure of the nonzero codes (from
# `_sparse_prototype`), and they are written straight into its stored values, which hold them in
# the same column order. With `vector`, the functions take a vector `v` before `u`, as a
# Jacobian-vector product does.
function _generate(codes::AbstractArray, m::_Model; inplace::Bool, sparse::Bool = false, vector::Bool = false)
    a = _arguments(m)
    v = vector ? Symbol[a.v] : Symbol[]
    prelude, outs = _prelude(codes, m, a)
    if inplace
        nonzero = [I for I in CartesianIndices(codes) if !_is_zero(codes[I])]
        if sparse
            start = [:($(a.stored) = $(GlobalRef(Giac, :_stored_values))($(a.out)))]
            writes = [:($(a.stored)[$k] = $(outs[I])) for (k, I) in enumerate(nonzero)]
        else
            start = length(nonzero) < length(codes) ? [:(fill!($(a.out), zero(eltype($(a.out)))))] : Expr[]
            writes = [:($(a.out)[$(Tuple(I)...)] = $(outs[I])) for I in nonzero]
        end
        body = Expr(:block, prelude..., start..., Expr(:macrocall, Symbol("@inbounds"), nothing, Expr(:block, writes...)),
                    :(return nothing))
        args = Symbol[a.out, v..., a.u, a.p, a.t]
    else
        entries = Expr(:tuple, vec(outs)...)
        body = Expr(:block, prelude..., :($(GlobalRef(Giac, :_array_like))($(a.u), $(Val(size(codes))), $entries)))
        args = Symbol[v..., a.u, a.p, a.t]
    end
    return _function(args, body, m, a)
end

# `code(u, p, t)`, a scalar
function _generate_scalar(code, m::_Model)
    a = _arguments(m)
    prelude, outs = _prelude(Any[code], m, a)
    return _function(Symbol[a.u, a.p, a.t], Expr(:block, prelude..., only(outs)), m, a)
end

# `(u, p, t)` giving the values of `codes` as a tuple
function _generate_tuple(codes, m::_Model)
    a = _arguments(m)
    prelude, outs = _prelude(codes, m, a)
    return _function(Symbol[a.u, a.p, a.t], Expr(:block, prelude..., Expr(:tuple, outs...)), m, a)
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
    d = Giac._derivative_of(sym)
    return d === nothing ? nothing : Giac._state_name(d[1], d[3])
end
_name(d::DerivativeExpr) = Giac._state_name(d.funcname, sum(last, d.steps))
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
        f = _generate_scalar(Giac.to_expr(_in_states(sym, s.model)), s.model)
        s.timed ? f : ((u, p) -> f(u, p, nothing))
    end
end

# An expression of the model in its state variables: unknown functions of time and their
# derivatives by their variables, highest derivatives by their solved expressions
function _in_states(e, m::_Model)
    e = Giac._as_states(_giac(e))
    isempty(m.derived) || (e = substitute(e, (Giac.giac_identifier(k) => v for (k, v) in m.derived)...))
    _check_free((e,), m)
    return e
end
_in_states(e::Number, ::_Model) = convert(GiacExpr, e)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# An input given as a Julia function has no symbolic derivatives: one of the state makes the
# Jacobians unknown, and one of time the time gradient. A solver then differentiates the
# compiled right-hand side itself.
function _ode_function(m::_Model; inplace::Bool = true, analytic::Bool = false, kwargs...)
    f, fkw, rest = _rhs(m; inplace, kwargs...)
    analytic && (fkw[:analytic] = _analytic(m))
    return ODEFunction{inplace, SciMLBase.FullSpecialize}(f; fkw..., rest...)
end

# The exact solution of a model affine in its states, `ẋ = A x + b` with `A` and `b` constant
# in time: `[x; 1]` evolves by the matrix exponential of `t * [A b; 0 0]`, compiled from the
# exact `A` and `b`
function _analytic(m::_Model)
    M = _affine(m)
    M === nothing && throw(ArgumentError("an exact solution needs a model affine in its states, ẋ = A x + b, with A and b free of time and given as expressions"))
    n = length(m.states)
    Mf = _generate(M, m; inplace = false)
    return (u0, p, t) -> (exp(t * Mf(nothing, p, t)) * vcat(u0, 1))[1:n]
end

# The code of `[A b; 0 0]` for `F = A x + b`, or `nothing` if `F` is not affine in the states
function _affine(m::_Model)
    (isempty(m.inputs) && m.mass === nothing) || return nothing
    n = length(m.states)
    names = Set(string.(m.states))
    m.t === nothing || all(f -> !any(s -> string(s) == string(m.t), Giac.free_symbols(f)), m.F) || return nothing
    M = Any[0 for _ in 1:n+1, _ in 1:n+1]
    for (i, f) in enumerate(m.F), s in Giac.free_symbols(f)
        j = findfirst(x -> string(x) == string(s), m.states)
        j === nothing && continue
        a = C.diff(f, m.states[j])
        any(x -> string(x) in names, Giac.free_symbols(a)) && return nothing
        M[i, j] = Giac.to_expr(a)
    end
    for (i, f) in enumerate(m.F)
        M[i, n+1] = Giac.to_expr(substitute(f, (x => 0 for x in m.states)...))
    end
    return M
end

# The compiled right-hand side, the keywords of its derivatives and index, and the other keywords
function _rhs(m::_Model; jac::Bool = true, tgrad::Bool = true, paramjac::Bool = true, jvp::Bool = false,
              vjp::Bool = false, sparse::Bool = false, inplace::Bool = true, kwargs...)
    by_state = any(i -> i.state, m.inputs)
    fkw = Dict{Symbol, Any}(:sys => GiacSystem(m; timed = true))
    J = (jac || jvp || vjp) && !by_state ? _derivatives(m.F, m.states) : nothing
    if jac && !by_state
        fkw[:jac] = _generate(J, m; inplace, sparse)
        sparse && (fkw[:jac_prototype] = _sparse_prototype(J))
    end
    v = _arguments(m).v
    jvp && !by_state && (fkw[:jvp] = _generate(_products(J, v), m; inplace, vector = true))
    vjp && !by_state && (fkw[:vjp] = _generate(_products(J, v; transpose = true), m; inplace, vector = true))
    if tgrad && m.t !== nothing && isempty(m.inputs)
        fkw[:tgrad] = _generate(_derivatives(m.F, m.t), m; inplace)
    end
    if paramjac && !isempty(m.params) && !by_state
        fkw[:paramjac] = _generate(_derivatives(m.F, m.params), m; inplace)
    end
    m.mass === nothing || (fkw[:mass_matrix] = m.mass)
    return _generate(_codes(m.F), m; inplace), fkw, kwargs
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
structure of the symbolic one. `jvp = true` and `vjp = true` add the products `J * v` and
`transpose(J) * v` with the exact Jacobian, for matrix-free (Krylov) solvers. For a model affine in
its states, `ẋ = A x + b` with `A` and `b` constant in time, `analytic = true` adds the exact
solution, `exp(t [A b; 0 0])` applied to `[x₀; 1]` with the exact `A` and `b`, so a solution
reports its errors (`sol.errors`). Other keywords (`mass_matrix`, …) go to `SciMLBase.ODEFunction`.

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
        fkw[:jac] = _drop_time(_generate(J, m; inplace, sparse), inplace)
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

# `solve` is shared with Giac's equation solver, which takes numbers, vectors and expressions:
# with a SciML algorithm, the first argument should have been a problem
function SciMLBase.solve(x::Giac.GiacInput, alg::SciMLBase.AbstractSciMLAlgorithm, args...; kwargs...)
    throw(ArgumentError("solve(x, $(nameof(typeof(alg)))) solves a SciML problem, and x is a $(typeof(x))"))
end

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

# The function whose zero crossings are the event, and whether only its upward crossings count:
# `a ~ b` at either crossing of `a - b`, and `a > b` (`a < b`, `a ≥ b`) when it becomes true.
# Giac writes every inequality with `>` or `>=`.
function _crossing(c)
    c = _giac(c)
    Giac._is_equation(c) && return Giac.lhs(c) - Giac.rhs(c), false
    if Giac.iscall(c) && Giac.symb_funcname(c) in (">", ">=")
        a, b = Giac.arguments(c)
        return a - b, true
    end
    return c, false
end

# The affect of an event: assignments to states and parameters, computed from the values at
# the event, or a Julia function of the integrator. Also whether it assigns parameters.
_affect(f, m::_Model) = (f, false)
function _affect(a::Union{Pair, AbstractVector{<:Pair}}, m::_Model)
    index(vs) = Dict(string(v) => i for (i, v) in enumerate(vs))
    states, params = index(m.states), index(m.params)
    targets, codes = Tuple{Symbol, Int}[], Any[]
    for (k, v) in _pairs(a), (var, val) in zip(_affected(k), _affected(v))
        n = _name(var)
        push!(targets, haskey(states, n) ? (:u, states[n]) : haskey(params, n) ? (:p, params[n]) :
              throw(ArgumentError("an event assigns states and parameters, and $var is neither")))
        push!(codes, Giac.to_expr(_in_states(val, m)))
    end
    f = _generate_tuple(codes, m)
    affect!(integ) = _assign!(integ, targets, f(integ.u, integ.p, integ.t))
    return affect!, any(t -> first(t) === :p, targets)
end

# Set each target `(:u, i)` or `(:p, i)` of the integrator to its value: in place in a mutable
# array, else by replacing the array
function _assign!(integ, targets, vals)
    for (k, (s, i)) in enumerate(targets)
        x = getproperty(integ, s)
        ismutable(x) ? (x[i] = vals[k]) : setproperty!(integ, s, Base.setindex(x, vals[k], i))
    end
end
_affected(x::AbstractArray) = vec(x)
_affected(x) = (x,)

# A continuous callback for each event `condition => affect`
function _callbacks(events, m::_Model)
    cbs = SciMLBase.ContinuousCallback[]
    for (condition, affect) in _pairs(events)
        e, upward = _crossing(condition)
        c = _generate_scalar(Giac.to_expr(_in_states(e, m)), m)
        a!, assigns_p = _affect(affect, m)
        push!(cbs, SciMLBase.ContinuousCallback((u, t, integ) -> c(u, integ.p, t), a!, upward ? nothing : a!;
                                                initialize = assigns_p ? _own_parameters : SciMLBase.INITIALIZE_DEFAULT))
    end
    return cbs
end

# A solver's integrator shares the problem's parameter values: an event that assigns parameters
# gives it its own copy, so the problem keeps them
function _own_parameters(c, u, t, integ)
    integ.p = copy(integ.p)
    SciMLBase.u_modified!(integ, false)
end

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

# Variables and their values from `x => values` or `[x1 => v1, x2 => v2]`
# Values may be expressions in `known`, the parameters' values.
_assignment(a::Pair, known = ()) = (_vector(first(a)), _values(last(a), known))
_assignment(a::AbstractVector{<:Pair}, known = ()) = (Any[first(q) for q in a], _values([last(q) for q in a], known))
_assignment(a::Tuple{}, known = ()) = (GiacExpr[], Float64[])
_assignment(::NullParameters, known = ()) = (GiacExpr[], Float64[])
_assignment(a, known = ()) = throw(ArgumentError("values are given as `x => values` or `[x1 => v1, x2 => v2, …]`, got $(typeof(a))"))

# A value computed in Julia from its exact expression, with the parameters' values `known`
# bound by name
function _value(v::GiacExpr, known = ())
    env = Dict{Symbol, Any}(Symbol(string(k)) => x for (k, x) in known)
    free = setdiff(Set(Symbol(string(x)) for x in Giac.free_symbols(v)), keys(env), (:pi, :infinity))
    isempty(free) || throw(ArgumentError("the value $v uses $(join(sort!(collect(free)), ", ")): a value may be written only in the parameters"))
    return _evaluate(Giac.to_expr(v), env)
end
_evaluate(x, env) = x
_evaluate(s::Symbol, env) = haskey(env, s) ? env[s] : getglobal(Base, s)
function _evaluate(ex::Expr, env)
    ex.head === :call || throw(ArgumentError("cannot compute the value $ex"))
    f = ex.args[1] isa Symbol ? getglobal(Base, ex.args[1]) : ex.args[1]
    return f((_evaluate(a, env) for a in ex.args[2:end])...)
end
_value(v, known = ()) = v
_values(v, known = ()) = (w = map(x -> _value(x, known), vec(collect(v))); isempty(w) ? Float64[] : float.(collect(promote(w...))))

# The parameters' values, to substitute into the initial values
_known(params, pv) = [_giac(k) => v for (k, v) in zip(params, pv)]

# Keywords that build the function; the rest are the problem's
const _FUNCTION_KEYWORDS = (:jac, :tgrad, :paramjac, :jvp, :vjp, :analytic, :sparse, :inplace, :mass_matrix, :inputs, :static)

function _split(kwargs)
    fkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if k in _FUNCTION_KEYWORDS)
    pkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if !(k in _FUNCTION_KEYWORDS))
    return fkw, pkw
end

# With `static = true`, the state and parameters as static arrays, for out-of-place functions
function _static!(fkw, u, pv)
    pop!(fkw, :static, false) || return u, pv
    fkw[:inplace] = false
    return Giac._static(u), isempty(pv) ? pv : Giac._static(pv)
end

"""
    ODEProblem(F, u0, tspan, p = (); t = nothing, inputs = [], events = [], kwargs...)
    ODEProblem(eqs, u0, tspan, p = (); inputs = [], events = [], kwargs...)

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
[`ODEFunction`](@ref).

`events` lists events `condition => affect`, located by the solver as zero crossings. The
condition `a ~ b` occurs when `a - b` crosses zero either way, and an inequality `a < b` when
it becomes true. The affect assigns states and parameters, all computed from the values at the
event, or is a Julia function of the integrator, such as `terminate!`:

```julia
@giac_var x[1:2] r
events = [x1 < 0 => [x2 => -r * x2]]       # a bouncing ball, x1 its height
```

An affect that puts the state exactly on the event's surface (here `x1 => 0`) hides the next
crossing from the solver. An event's parameter changes are the solve's own: the problem keeps
its values.

`F` may also be a function from `@giac_function` with arguments
`(u, p, t)`. The keywords of `ODEFunction` build the function; the others go to
`SciMLBase.ODEProblem`.
"""
function SciMLBase.ODEProblem(F::AbstractVector{GiacExpr}, u0, tspan, p = (); t = nothing, kwargs...)
    m, u, pv, fkw, pkw, _ = _problem_model(F, u0, p, t, kwargs)
    return ODEProblem(_ode_function(m; fkw...), u, _tspan(tspan), pv; pkw...)
end

# A problem's model, its initial state and parameter values, the keywords of its function and
# of the problem, with the events as callbacks, and its inputs
function _problem_model(F, u0, p, t, kwargs)
    params, pv = _assignment(p)
    fkw, pkw = _split(kwargs)
    inputs = pop!(fkw, :inputs, ())
    if _is_equations(F)
        m = _equation_model(F, _vector(params), t, inputs)
        u = last(_ordered(u0, m.states, _known(params, pv)))
    else
        states, u = _assignment(u0, _known(params, pv))
        m = _vector_model(F, _vector(states), _vector(params), t, inputs)
    end
    u, pv = _static!(fkw, u, pv)
    events = pop!(pkw, :events, ())
    if !isempty(events)
        cbs = _callbacks(events, m)
        pkw[:callback] = SciMLBase.CallbackSet(get(pkw, :callback, nothing), cbs...)
    end
    return m, u, pv, fkw, pkw, inputs
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
    params, pv = _assignment(p)
    states, u = _assignment(u0, _known(params, pv))
    fkw, pkw = _split(kwargs)
    u, pv = _static!(fkw, u, pv)
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
_ordered(a::AbstractVector{<:Number}, vars, known = ()) = vars => a
_ordered(a::Tuple{}, vars, known = ()) = isempty(vars) ? () : throw(ArgumentError("no values given for $(join(vars, ", "))"))
_ordered(a::NullParameters, vars, known = ()) = _ordered((), vars)
function _ordered(a, vars, known = ())
    keys_, vals = _assignment(a, known)
    given = Dict(something(_name(k), string(k)) => v for (k, v) in zip(keys_, vals))
    missing_ = [v for v in vars if !haskey(given, string(v))]
    isempty(missing_) || throw(ArgumentError("no values given for $(join(missing_, ", "))"))
    extra = setdiff(keys(given), string.(vars))
    isempty(extra) || throw(ArgumentError("$(join(sort!(collect(extra)), ", ")) not among the model's variables"))
    return vars => [given[string(v)] for v in vars]
end

_tspan(t::Tuple) = promote(map(float ∘ _value, t)...)
_tspan(t) = t

# ---------------------------------------------------------------------------
# Stochastic models
# ---------------------------------------------------------------------------

# The noise of a model in its state variables, with its inputs substituted: a vector for a
# separate noise on each state, or a matrix with a column for each noise process
function _noise(G::AbstractVecOrMat, m::_Model, inputs)
    subs, _ = _inputs(inputs)
    size(G, 1) == length(m.states) ||
        throw(DimensionMismatch("the noise has $(size(G, 1)) rows for $(length(m.states)) states"))
    return map(g -> _in_states(g isa Number ? g : _substitute(_giac(g), subs), m), G)
end

function _sde_function(m::_Model, G; inplace::Bool = true, kwargs...)
    f, fkw, rest = _rhs(m; inplace, kwargs...)
    g = _generate(_codes(G), m; inplace)
    return SciMLBase.SDEFunction{inplace, SciMLBase.FullSpecialize}(f, g; fkw..., rest...)
end

"""
    SDEFunction(F, G, x; p = [], t = nothing, inputs = [], kwargs...)
    SDEFunction(eqs, G; p = [], inputs = [], kwargs...)

The SciML function of the stochastic model `dx = F(x, p, t) dt + G(x, p, t) dW`, with the drift
`F` as for [`ODEFunction`](@ref) and the noise `G` compiled with it. `G` is a vector for a
separate noise on each state (diagonal noise), or a matrix with a column for each noise process.
With equations, `G` is written in the unknown functions and their derivatives.
"""
function SciMLBase.SDEFunction(F::AbstractVector{GiacExpr}, G::AbstractVecOrMat, x::_Vars; p = GiacExpr[],
                               t = nothing, inputs = (), kwargs...)
    m = _vector_model(F, _vector(x), _vector(p), t, inputs)
    return _sde_function(m, _noise(G, m, inputs); kwargs...)
end

function SciMLBase.SDEFunction(eqs::AbstractVector{GiacExpr}, G::AbstractVecOrMat; p = GiacExpr[], t = nothing,
                               inputs = (), kwargs...)
    _is_equations(eqs) || throw(ArgumentError("give the states of a vector of drifts: SDEFunction(F, G, x)"))
    m = _equation_model(eqs, _vector(p), t, inputs)
    return _sde_function(m, _noise(G, m, inputs); kwargs...)
end

"""
    SDEProblem(F, G, u0, tspan, p = (); t = nothing, inputs = [], events = [], kwargs...)
    SDEProblem(eqs, G, u0, tspan, p = (); inputs = [], events = [], kwargs...)

The stochastic initial-value problem `dx = F(x, p, t) dt + G(x, p, t) dW`, `x(t₀) = u0`, with
assignments, inputs and events as for [`ODEProblem`](@ref) and the noise `G` as for
[`SDEFunction`](@ref):

```julia
@giac_var x θ μ σ
SDEProblem([θ * (μ - x)], [σ], [x => 1], (0, 10), [θ => 2, μ => 0, σ => 0.3])   # Ornstein–Uhlenbeck
```

A matrix `G` sets the problem's `noise_rate_prototype`.
"""
function SciMLBase.SDEProblem(F::AbstractVector{GiacExpr}, G::AbstractVecOrMat, u0, tspan, p = ();
                              t = nothing, kwargs...)
    m, u, pv, fkw, pkw, inputs = _problem_model(F, u0, p, t, kwargs)
    noise = _noise(G, m, inputs)
    noise isa AbstractMatrix && !haskey(pkw, :noise_rate_prototype) && (pkw[:noise_rate_prototype] = zeros(size(noise)))
    return SciMLBase.SDEProblem(_sde_function(m, noise; fkw...), u, _tspan(tspan), pv; pkw...)
end

end # module
