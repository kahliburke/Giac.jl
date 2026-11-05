# Julia code from Giac expressions, and native numeric functions compiled from it.

# Giac heads whose Julia function has a different name, or that Julia spells as an operator
const _JULIA_HEADS = Dict(
    "+" => :+, "*" => :*, "^" => :^, "-" => :-, "inv" => :inv, "ln" => :log, "log10" => :log10,
    "sqrt" => :sqrt, "abs" => :abs, "sign" => :sign, "exp" => :exp, "floor" => :floor, "ceil" => :ceil,
    "max" => :max, "min" => :min, "atan" => :atan, "asin" => :asin, "acos" => :acos,
    "sin" => :sin, "cos" => :cos, "tan" => :tan, "sinh" => :sinh, "cosh" => :cosh, "tanh" => :tanh,
    "asinh" => :asinh, "acosh" => :acosh, "atanh" => :atanh, "re" => :real, "im" => :imag,
    "conj" => :conj)

const _JULIA_CONSTANTS = Dict("pi" => :π, "infinity" => :Inf)

"""
    to_expr(e::GiacExpr) -> Union{Expr, Symbol, Number}

Julia code for a Giac expression: variables become symbols, numbers become Julia numbers
(integers and rationals stay exact), and functions become their Julia counterparts (`ln` →
`log`, `pi` → `π`).

```julia
@giac_var x y
Giac.to_expr(sin(x)^2 / y)          # :(sin(x) ^ 2 * inv(y))
```
"""
function to_expr(e::GiacExpr)
    if iscall(e)
        head = symb_funcname(e)
        head == "+" && return _sum_expr(arguments(e))
        head == "*" && return _product_expr(arguments(e))
        args = map(to_expr, arguments(e))
        head == "of" && return Expr(:call, args...)            # an undefined function f(a, b)
        haskey(_JULIA_HEADS, head) || throw(ArgumentError("no Julia function for Giac's `$head` in $e"))
        return Expr(:call, _JULIA_HEADS[head], args...)
    end
    name = string(e)
    haskey(_JULIA_CONSTANTS, name) && return _JULIA_CONSTANTS[name]
    is_identifier(e) && return Symbol(name)
    return to_julia(e)
end
to_expr(x::Number) = x

_is_head(e::GiacExpr, head) = iscall(e) && symb_funcname(e) == head

# Giac's `a + -b` as `a - b` (also for negative numbers)
function _sum_expr(terms)
    plus, minus = Any[], Any[]
    for t in terms
        if _is_head(t, "-")
            push!(minus, to_expr(only(arguments(t))))
        else
            x = to_expr(t)
            x isa Real && x < 0 ? push!(minus, -x) : push!(plus, x)
        end
    end
    ex = isempty(plus) ? Expr(:call, :-, popfirst!(minus)) :
         length(plus) == 1 ? plus[1] : Expr(:call, :+, plus...)
    for m in minus
        ex = Expr(:call, :-, ex, m)
    end
    return ex
end

# Giac's `a * inv(b)` as `a / b`, and a rational coefficient `(p/q) * a` as `p * a / q`
function _product_expr(factors)
    num, den = Any[], Any[]
    for f in factors
        if _is_head(f, "inv")
            push!(den, to_expr(only(arguments(f))))
        else
            x = to_expr(f)
            if x isa Rational
                numerator(x) == 1 || pushfirst!(num, numerator(x))
                push!(den, denominator(x))
            else
                push!(num, x)
            end
        end
    end
    product(xs) = isempty(xs) ? 1 : length(xs) == 1 ? xs[1] : Expr(:call, :*, xs...)
    return isempty(den) ? product(num) : Expr(:call, :/, product(num), product(den))
end

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

"""
    _runtime_function(args, body)

The function `(args...) -> body`, compiled now and callable directly, also from code compiled
before it existed.
"""
_runtime_function(args, body) =
    RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, Expr(:->, Expr(:tuple, args...), body))

_symbols!(s, ex::Symbol) = push!(s, ex)
_symbols!(s, ex::Expr) = (foreach(a -> _symbols!(s, a), ex.args); s)
_symbols!(s, ex::AbstractArray) = (foreach(a -> _symbols!(s, a), ex); s)
_symbols!(s, _) = s

"""
    _common_subexpressions(codes; taken = Set{Symbol}()) -> (assignments, outputs)

Julia code for the array of expressions `codes` in which every call that occurs more than once
is computed once, into a local variable: the assignments, in order, and the outputs in terms of
them. The local variables are named `s1`, `s2`, … avoiding every name in `codes` and `taken`.
"""
function _common_subexpressions(codes::AbstractArray; taken = Set{Symbol}())
    counts = Dict{Expr, Int}()
    function count!(ex)
        ex isa Expr && ex.head === :call || return
        n = counts[ex] = get(counts, ex, 0) + 1
        n == 1 && foreach(count!, ex.args[2:end])          # inside a repeat, computed once anyway
        return
    end
    foreach(count!, codes)
    taken = _symbols!(copy(taken), codes)
    names, assignments, k = Dict{Expr, Symbol}(), Expr[], Ref(0)
    function fresh()
        s = Symbol("s", k[] += 1)
        return s in taken ? fresh() : s
    end
    function rewrite(ex)
        ex isa Expr && ex.head === :call || return ex
        haskey(names, ex) && return names[ex]
        new = Expr(:call, ex.args[1], map(rewrite, ex.args[2:end])...)
        counts[ex] > 1 || return new
        s = names[ex] = fresh()
        push!(assignments, :($s = $new))
        return s
    end
    return assignments, map(rewrite, codes)
end

"""
A numeric function compiled from Giac expressions by [`build_function`](@ref) with
`backend = :julia`. `code` holds the generated Julia function.
"""
struct CompiledFunction{F} <: Function
    f::F
    code::Expr
end

(c::CompiledFunction)(args...) = c.f(args...)

Base.show(io::IO, c::CompiledFunction) = print(io, "CompiledFunction", c.code.args[1])

"""
Compile `exprs` (one expression or an array) into a Julia function of the variable groups
`groups`: each group is a variable (the function takes a number) or an array of variables (the
function takes an array of the same length).
"""
function _compile(exprs, groups)
    args, prelude, bound = Symbol[], Expr[], Set{String}()
    for (k, g) in enumerate(groups)
        if g isa GiacExpr
            push!(args, Symbol(string(g)))
            push!(bound, string(g))
        else
            a = Symbol("arg", k)
            push!(args, a)
            for (i, v) in enumerate(g)
                push!(prelude, Expr(:(=), Symbol(string(v)), Expr(:ref, a, i)))
                push!(bound, string(v))
            end
        end
    end
    free = Set(string(s) for e in (exprs isa GiacExpr ? (exprs,) : exprs) for s in free_symbols(e))
    unbound = setdiff(free, bound, keys(_JULIA_CONSTANTS))
    isempty(unbound) || throw(ArgumentError("the expression has variables that are not arguments: $(join(sort!(collect(unbound)), ", "))"))
    codes = exprs isa GiacExpr ? [to_expr(exprs)] : map(to_expr, exprs)
    assignments, outputs = _common_subexpressions(codes; taken = Set(args))
    value = exprs isa GiacExpr ? only(outputs) :
            exprs isa AbstractVector ? Expr(:vect, outputs...) :
            :(reshape($(Expr(:vect, vec(outputs)...)), $(size(exprs))))
    body = Expr(:block, prelude..., assignments..., value)
    code = Expr(:->, Expr(:tuple, args...), body)
    return CompiledFunction(_runtime_function(args, body), code)
end

"""
    build_function(exprs, vars...; backend = :giac)

`exprs` may be an array of expressions (the function then returns an array of the same shape),
and each of `vars` may be an array of variables (the function then takes an array for it):

```julia
@giac_var x[1:2] u
f = build_function([x2, -sin(x1) + u], x, u; backend = :julia)
f([0.1, 0.0], 0.5)                  # [0.0, 0.40016…]
```

With `backend = :julia` the expressions are translated to Julia code ([`Giac.to_expr`](@ref))
and compiled to a native function, fast enough for an ODE right-hand side. Every free variable
must be one of `vars`. The default `:giac` backend substitutes and converts through Giac on each
call.
"""
function build_function(exprs::Union{GiacExpr, AbstractArray{GiacExpr}},
                        vars::Union{GiacExpr, AbstractArray{GiacExpr}}...; backend::Symbol = :giac)
    backend === :julia && return _compile(exprs, vars)
    backend === :giac || throw(ArgumentError("unknown backend $backend for arrays; expected :giac or :julia"))
    return (vals...) -> begin
        length(vals) == length(vars) || throw(MethodError(build_function, vals))
        pairs = [Pair(v, x) for (v, x) in zip(vars, vals)]
        r = substitute(exprs, pairs...)
        r isa GiacExpr ? to_julia(r) : map(to_julia, r)
    end
end

# A sparse matrix with nonzeros where `mask` is true: the structure of a symbolic Jacobian, as a
# prototype for a solver. Defined when SparseArrays is loaded.
_sparse_pattern(mask) = throw(ArgumentError("a sparse Jacobian needs SparseArrays: `using SparseArrays`"))

# An array of size `dims` holding `entries`, of the kind of `u`: an `Array`, or a static array
# for a static `u` (defined when StaticArrays is loaded). Generated out-of-place code builds its
# results with it.
function _array_like(u, ::Val{dims}, entries::Tuple) where {dims}
    v = [entries...]                        # promoted to a common type
    return length(dims) == 1 ? v : reshape(v, dims)
end

# `v` as a static array. Defined when StaticArrays is loaded.
_static(v) = throw(ArgumentError("static arrays need StaticArrays: `using StaticArrays`"))
