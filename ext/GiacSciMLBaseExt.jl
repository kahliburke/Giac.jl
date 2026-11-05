# ---------------------------------------------------------------------------
# GiacSciMLBaseExt: SciML problems from symbolic models.
#
# A model is a vector of right-hand sides in state variables, parameters and possibly a time
# variable. The right-hand side and its exact derivatives (Jacobian, time gradient, parameter
# Jacobian) are translated to Julia code with `Giac.to_expr` and compiled as runtime-generated
# functions, so solvers call them as ordinary Julia functions. The problem carries a symbolic
# index (`GiacSystem`), so solutions and problems are indexed by the model's variables:
# `sol[x1]`, `sol[x1^2 + x2^2]`, `remake(prob; p = [g => 3.7])`.
# ---------------------------------------------------------------------------

module GiacSciMLBaseExt

using Giac
using Giac: GiacExpr, GiacFunction, jacobian
import SciMLBase
using SciMLBase: ODEFunction, ODEProblem, NonlinearFunction, NonlinearProblem, NullParameters
import SciMLBase.SymbolicIndexingInterface as SII
import SciMLBase.RuntimeGeneratedFunctions as RGF

RGF.init(@__MODULE__)

const _Vars = Union{GiacExpr, AbstractArray{GiacExpr}}

_vector(x::GiacExpr) = GiacExpr[x]
_vector(x::AbstractArray) = GiacExpr[v for v in vec(x)]
_vector(::Nothing) = GiacExpr[]
_vector(x::Tuple) = GiacExpr[v for v in x]

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

# Argument names that cannot clash with a variable of the model
const _OUT, _U, _P, _T = Symbol("##out"), Symbol("##u"), Symbol("##p"), Symbol("##t")

# The model's variables bound to the function's arguments
function _bindings(states, params, t)
    b = Expr[]
    for (i, v) in enumerate(states)
        push!(b, :($(Symbol(string(v))) = $_U[$i]))
    end
    for (i, v) in enumerate(params)
        push!(b, :($(Symbol(string(v))) = $_P[$i]))
    end
    t === nothing || push!(b, :($(Symbol(string(t))) = $_T))
    return b
end

function _check_free(exprs, states, params, t)
    bound = Set(string(v) for v in Iterators.flatten((states, params, t === nothing ? () : (t,))))
    free = Set(string(s) for e in exprs for s in Giac.free_symbols(e))
    unbound = setdiff(free, bound, ("pi", "infinity"))
    isempty(unbound) || throw(ArgumentError("the model has variables that are neither states, parameters nor time: $(join(sort!(collect(unbound)), ", ")). Give them as parameters, or `t = …` for time"))
    return nothing
end

_rgf(code) = RGF.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, code)

_is_zero(c) = c isa Number && iszero(c)

# `out = exprs(u, p, t)` in place, and `exprs(u, p, t)` as a new array. A zero entry of an array
# is not written in place: it is set by `fill!` for a dense array, and is not stored in a sparse
# prototype.
function _inplace(exprs::AbstractArray{GiacExpr}, states, params, t; dense_zeros::Bool = true)
    codes = map(Giac.to_expr, exprs)
    writes = [:($_OUT[$(Tuple(I)...)] = $(codes[I])) for I in CartesianIndices(exprs) if !_is_zero(codes[I])]
    fill = dense_zeros && any(_is_zero, codes) ? (:(fill!($_OUT, zero(eltype($_OUT)))),) : ()
    body = Expr(:block, _bindings(states, params, t)..., fill..., :(@inbounds begin $(writes...) end), :(return nothing))
    return _rgf(:(($_OUT, $_U, $_P, $_T) -> $body))
end

function _outofplace(exprs::AbstractArray{GiacExpr}, states, params, t)
    entries = Expr(:vect, map(Giac.to_expr, vec(exprs))...)
    value = exprs isa AbstractVector ? entries : :(reshape($entries, $(size(exprs))))
    body = Expr(:block, _bindings(states, params, t)..., value)
    return _rgf(:(($_U, $_P, $_T) -> $body))
end

# A nonlinear problem's functions take `(u, p)`: the same code with no time argument
_drop_time(f, inplace) = inplace ? ((out, u, p) -> f(out, u, p, nothing)) : ((u, p) -> f(u, p, nothing))

_compiled(exprs, states, params, t, inplace; dense_zeros = true) =
    inplace ? _inplace(exprs, states, params, t; dense_zeros) : _outofplace(exprs, states, params, t)

# ---------------------------------------------------------------------------
# Symbolic indexing
# ---------------------------------------------------------------------------

"""
The variables of a model built from Giac expressions, for SymbolicIndexingInterface: states,
parameters and the time variable, found by name. Any other expression in them is an observed
quantity, compiled on first use.
"""
struct GiacSystem
    states::Vector{GiacExpr}
    params::Vector{GiacExpr}
    t::Union{Nothing, GiacExpr}
    timed::Bool                         # an ODE's system, time-dependent even with no time variable
    state_index::Dict{String, Int}
    param_index::Dict{String, Int}
    observed::Dict{String, Any}
end
GiacSystem(states, params, t; timed = t !== nothing) =
    GiacSystem(states, params, t, timed, Dict(string(v) => i for (i, v) in enumerate(states)),
               Dict(string(v) => i for (i, v) in enumerate(params)), Dict{String, Any}())

Base.show(io::IO, s::GiacSystem) =
    print(io, "GiacSystem(states = [", join(s.states, ", "), "], parameters = [", join(s.params, ", "), "]",
          s.t === nothing ? "" : ", t = $(s.t)", ")")

SII.symbolic_type(::Type{GiacExpr}) = SII.ScalarSymbolic()
SII.hasname(e::GiacExpr) = Giac.is_identifier(e)
SII.getname(e::GiacExpr) = Symbol(string(e))

# a variable's name, or `nothing` for an index or an expression that is not a variable
_name(sym::GiacExpr) = Giac.is_identifier(sym) ? string(sym) : nothing
_name(sym::Symbol) = string(sym)
_name(sym) = nothing

_index(d, sym) = (n = _name(sym); n === nothing ? nothing : get(d, n, nothing))

SII.is_variable(s::GiacSystem, sym) = sym isa Integer ? sym in eachindex(s.states) : _index(s.state_index, sym) !== nothing
SII.variable_index(s::GiacSystem, sym) = sym isa Integer ? sym : _index(s.state_index, sym)
SII.variable_symbols(s::GiacSystem) = s.states
SII.is_parameter(s::GiacSystem, sym) = sym isa Integer ? sym in eachindex(s.params) : _index(s.param_index, sym) !== nothing
SII.parameter_index(s::GiacSystem, sym) = sym isa Integer ? sym : _index(s.param_index, sym)
SII.parameter_symbols(s::GiacSystem) = s.params
SII.is_independent_variable(s::GiacSystem, sym) = s.t !== nothing && _name(sym) == string(s.t)
SII.independent_variable_symbols(s::GiacSystem) = s.t === nothing ? GiacExpr[] : [s.t]
SII.is_time_dependent(s::GiacSystem) = s.timed
SII.constant_structure(::GiacSystem) = true
SII.all_variable_symbols(s::GiacSystem) = s.states
SII.all_symbols(s::GiacSystem) = vcat(s.states, s.params, SII.independent_variable_symbols(s))
SII.default_values(::GiacSystem) = Dict()

SII.is_observed(s::GiacSystem, sym::GiacExpr) =
    !(SII.is_variable(s, sym) || SII.is_parameter(s, sym) || SII.is_independent_variable(s, sym))
SII.is_observed(::GiacSystem, sym) = false

# `(u, p, t) -> value` for a time-dependent system, else `(u, p) -> value`
function SII.observed(s::GiacSystem, sym::GiacExpr)
    get!(s.observed, string(sym)) do
        _check_free((sym,), s.states, s.params, s.t)
        f = _outofplace(GiacExpr[sym], s.states, s.params, s.t)
        s.timed ? ((u, p, t) -> f(u, p, t)[1]) : ((u, p) -> f(u, p, nothing)[1])
    end
end

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

_sparse_prototype(J) = Giac._sparse_pattern(map(e -> !_is_zero(Giac.to_expr(e)), J))

"""
    ODEFunction(F, x; p = [], t = nothing, jac = true, tgrad = true, paramjac = true,
                sparse = false, inplace = true, kwargs...)
    ODEFunction(F::GiacFunction; kwargs...)

The SciML function of the model `ẋ = F(x, p, t)`, with `F` a vector of Giac expressions in the
states `x`, the parameters `p` and the time variable `t`. The right-hand side and, by default,
its exact Jacobian `∂F/∂x`, time gradient `∂F/∂t` and parameter Jacobian `∂F/∂p` are compiled
to native code. `sparse = true` (with SparseArrays loaded) gives the Jacobian as a sparse
matrix with the structure of the symbolic one. Other keywords (`mass_matrix`, …) go to
`SciMLBase.ODEFunction`.

A function from `@giac_function` takes its arguments as SciML's `(u, p, t)`: the states, then
optionally the parameters and the time variable.

```julia
@giac_var x[1:2] g ℓ c t
F = [x2, -g / ℓ * sin(x1) - c * x2]
f = ODEFunction(F, x; p = [g, ℓ, c])
f.jac(zeros(2, 2), [0.1, 0.0], [9.81, 1.0, 0.2], 0.0)
```
"""
function SciMLBase.ODEFunction(F::AbstractVector{GiacExpr}, x::_Vars; p = GiacExpr[], t = nothing,
                               jac::Bool = true, tgrad::Bool = true, paramjac::Bool = true,
                               sparse::Bool = false, inplace::Bool = true, kwargs...)
    states, params = _vector(x), _vector(p)
    length(F) == length(states) ||
        throw(DimensionMismatch("the model has $(length(F)) right-hand sides for $(length(states)) states"))
    _check_free(F, states, params, t)
    sys = GiacSystem(states, params, t; timed = true)
    J = jac ? jacobian(F, states) : nothing
    fkw = Dict{Symbol, Any}(:sys => sys)
    if jac
        fkw[:jac] = _compiled(J, states, params, t, inplace; dense_zeros = !sparse)
        sparse && (fkw[:jac_prototype] = _sparse_prototype(J))
    end
    if tgrad && t !== nothing
        fkw[:tgrad] = _compiled(GiacExpr[Giac.Commands.diff(e, t) for e in F], states, params, t, inplace)
    end
    if paramjac && !isempty(params)
        fkw[:paramjac] = _compiled(jacobian(F, params), states, params, t, inplace)
    end
    f = _compiled(F, states, params, t, inplace)
    return ODEFunction{inplace, SciMLBase.FullSpecialize}(f; fkw..., kwargs...)
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
    NonlinearFunction(F, x; p = [], jac = true, sparse = false, inplace = true, kwargs...)

The SciML function of the system `F(x, p) = 0`, with its exact Jacobian `∂F/∂x`. An equation
`a ~ b` in `F` stands for `a - b = 0`.
"""
function SciMLBase.NonlinearFunction(F::AbstractVector{GiacExpr}, x::_Vars; p = GiacExpr[], jac::Bool = true,
                                     sparse::Bool = false, inplace::Bool = true, kwargs...)
    F = map(_residual, F)
    states, params = _vector(x), _vector(p)
    _check_free(F, states, params, nothing)
    fkw = Dict{Symbol, Any}(:sys => GiacSystem(states, params, nothing))
    if jac
        J = jacobian(F, states)
        fkw[:jac] = _drop_time(_compiled(J, states, params, nothing, inplace; dense_zeros = !sparse), inplace)
        sparse && (fkw[:jac_prototype] = _sparse_prototype(J))
    end
    f = _drop_time(_compiled(F, states, params, nothing, inplace), inplace)
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
_assignment(a::AbstractVector{<:Pair}) = (GiacExpr[first(q) for q in a], _values([last(q) for q in a]))
_assignment(a::Tuple{}) = (GiacExpr[], Float64[])
_assignment(::NullParameters) = (GiacExpr[], Float64[])
_assignment(a) = throw(ArgumentError("values are given as `x => values` or `[x1 => v1, x2 => v2, …]`, got $(typeof(a))"))

_value(v::GiacExpr) = Giac.to_julia(Giac.Commands.evalf(v))
_value(v) = v
_values(v) = (w = map(_value, vec(collect(v))); isempty(w) ? Float64[] : collect(promote(w...)))

# Keywords that build the function; the rest are the problem's
const _FUNCTION_KEYWORDS = (:jac, :tgrad, :paramjac, :sparse, :inplace, :mass_matrix)

function _split(kwargs)
    fkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if k in _FUNCTION_KEYWORDS)
    pkw = Dict{Symbol, Any}(k => v for (k, v) in kwargs if !(k in _FUNCTION_KEYWORDS))
    return fkw, pkw
end

"""
    ODEProblem(F, u0, tspan, p = (); t = nothing, kwargs...)

The initial-value problem `ẋ = F(x, p, t)`, `x(t₀) = u0`, with the states and parameters taken
from the assignments `u0` and `p`, each written `x => values` or `[x1 => v1, x2 => v2]`:

```julia
@giac_var x[1:2] g ℓ c
F = [x2, -g / ℓ * sin(x1) - c * x2]
prob = ODEProblem(F, [x1 => π / 2, x2 => 0], (0, 10), [g => 9.81, ℓ => 1, c => 0.2])
sol = solve(prob, Rodas5P())
sol[x1^2 + x2^2]
```

`F` may also be a function from `@giac_function` with arguments `(u, p, t)`. The keywords of
[`ODEFunction`](@ref) build the function; the others go to `SciMLBase.ODEProblem`.
"""
function SciMLBase.ODEProblem(F::AbstractVector{GiacExpr}, u0, tspan, p = (); t = nothing, kwargs...)
    states, u = _assignment(u0)
    params, pv = _assignment(p)
    fkw, pkw = _split(kwargs)
    f = ODEFunction(F, states; p = params, t, fkw...)
    return ODEProblem(f, u, _tspan(tspan), pv; pkw...)
end

function SciMLBase.ODEProblem(F::GiacFunction, u0, tspan, p = (); kwargs...)
    u, params, t = _uptime(F)
    return ODEProblem(_body_vector(F), _ordered(u0, _vector(u)), tspan, _ordered(p, params); t, kwargs...)
end

"""
    NonlinearProblem(F, u0, p = (); kwargs...)

The system `F(x, p) = 0` with initial guess `u0`, written `x => values` or `[x1 => v1, …]`, as
for [`ODEProblem`](@ref).
"""
function SciMLBase.NonlinearProblem(F::AbstractVector{GiacExpr}, u0, p = (); kwargs...)
    states, u = _assignment(u0)
    params, pv = _assignment(p)
    fkw, pkw = _split(kwargs)
    f = NonlinearFunction(F, states; p = params, fkw...)
    return NonlinearProblem(f, u, pv; pkw...)
end

function SciMLBase.NonlinearProblem(F::GiacFunction, u0, p = (); kwargs...)
    u, params, _ = _uptime(F)
    return NonlinearProblem(_body_vector(F), _ordered(u0, _vector(u)), _ordered(p, params); kwargs...)
end

# The values of `vars` in an assignment, in their order: a function's variables are fixed by its
# arguments, and an assignment may list them in any order or give a plain vector of values.
_ordered(a::AbstractVector{<:Number}, vars) = vars => a
_ordered(a::Tuple{}, vars) = isempty(vars) ? () : throw(ArgumentError("no values given for $(join(vars, ", "))"))
_ordered(a::NullParameters, vars) = _ordered((), vars)
function _ordered(a, vars)
    names, vals = _assignment(a)
    given = Dict(string(n) => v for (n, v) in zip(names, vals))
    missing_ = [v for v in vars if !haskey(given, string(v))]
    isempty(missing_) || throw(ArgumentError("no values given for $(join(missing_, ", "))"))
    extra = setdiff(keys(given), string.(vars))
    isempty(extra) || throw(ArgumentError("$(join(sort!(collect(extra)), ", ")) not among the function's variables"))
    return vars => [given[string(v)] for v in vars]
end

_tspan(t::Tuple) = promote(map(float ∘ _value, t)...)
_tspan(t) = t

end # module
