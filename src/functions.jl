# Symbolic functions: `@giac_function f(x) = …` defines a function of symbolic arguments that
# can be called on expressions or numbers, differentiated, and typeset as a definition.

"""
    GiacFunction

A function defined by [`@giac_function`](@ref): a body in terms of its argument variables.
Calling it substitutes the arguments; called with floating-point numbers it runs a compiled
numeric version instead.
"""
struct GiacFunction{B} <: Function
    name::Symbol
    argnames::Vector{Symbol}
    args::Vector{Any}                   # each a variable or an array of variables
    body::B                             # an expression or an array of them
    numeric::Base.RefValue{Any}         # the compiled version, built on first numeric call
end
GiacFunction(name, argnames, args, body) = GiacFunction(Symbol(name), argnames, collect(Any, args), body, Ref{Any}())

function (f::GiacFunction)(vals...)
    length(vals) == length(f.args) ||
        throw(ArgumentError("$(f.name) takes $(length(f.args)) argument(s), got $(length(vals))"))
    if _numeric_call(vals)
        c = _numeric(f)
        c === nothing || return c(vals...)
    end
    return substitute(f.body, (a => _substitution_value(v) for (a, v) in zip(f.args, vals))...)
end

# real numbers, at least one of them floating point: the call wants a number back
_numeric_call(vals) = all(v -> v isa Real || v isa AbstractArray{<:Real}, vals) &&
                      any(v -> v isa AbstractFloat || v isa AbstractArray{<:AbstractFloat}, vals)
_substitution_value(v::AbstractArray) = Tuple(v)    # entry by entry, keeping exact constants
_substitution_value(v) = v

# The compiled body, or `nothing` if it has free variables besides the arguments (a parameter
# such as `m` in `m * g * h`): such a function stays symbolic.
function _numeric(f::GiacFunction)
    isassigned(f.numeric) && return f.numeric[]
    c = try
        _compile(f.body, f.args)
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    return f.numeric[] = c
end

# every argument variable, in order
_variables(f::GiacFunction) = GiacExpr[v for a in f.args for v in (a isa GiacExpr ? (a,) : vec(a))]

# a function computed from `f` (a derivative), like terms in its body collected
_derived(f::GiacFunction, name, body) =
    GiacFunction(name, f.argnames, f.args, body isa GiacExpr ? Commands.normal(body) : map(Commands.normal, body))

"""
    f'

The derivative of a function of one variable, as a function: `f'(2)`, `f''`.
"""
function Base.adjoint(f::GiacFunction)
    (length(f.args) == 1 && f.args[1] isa GiacExpr && f.body isa GiacExpr) ||
        throw(ArgumentError("$(f.name)' is for a scalar function of one variable; use ∇($(f.name)) or jacobian($(f.name))"))
    return _derived(f, Symbol(f.name, "′"), Commands.diff(f.body, f.args[1]))
end

"""
    ∇(f)
    jacobian(F)
    hessian(f)

The gradient, Jacobian or Hessian of a function from [`@giac_function`](@ref), with respect to
all its arguments, as a function of the same arguments.
"""
∇(f::GiacFunction) = _derived(f, Symbol("∇", f.name), ∇(f.body, _variables(f)))
jacobian(f::GiacFunction) = _derived(f, Symbol("J_", f.name), jacobian(f.body, _variables(f)))
hessian(f::GiacFunction) = _derived(f, Symbol("H_", f.name), hessian(f.body, _variables(f)))

function Base.show(io::IO, f::GiacFunction)
    print(io, f.name, "(", join(f.argnames, ", "), ") = ")
    _print_body(io, f.body)
end
_print_body(io, b::GiacExpr) = print(io, b)
_print_body(io, b::AbstractVector) = print(io, "[", join(b, ", "), "]")
_print_body(io, b::AbstractMatrix) = print(io, "[", join((join(r, " ") for r in eachrow(b)), "; "), "]")
Base.show(io::IO, ::MIME"text/plain", f::GiacFunction) = show(io, f)

function Base.show(io::IO, ::MIME"text/latex", f::GiacFunction)
    print(io, "\$\$", _function_name_latex(f.name), "\\left(",
          join((_entry_latex(giac_identifier(string(a))) for a in f.argnames), ", "), "\\right) = ",
          giac_latex(f.body), "\$\$")
end

function _function_name_latex(name::Symbol)
    s = string(name)
    primes = count(==('′'), s)
    s = replace(s, "′" => "")
    for (prefix, tex) in (("∇", "\\nabla "), ("J_", "J_"), ("H_", "H_"))
        startswith(s, prefix) && return tex * _function_name_latex(Symbol(chopprefix(s, prefix)))
    end
    return _entry_latex(giac_identifier(s)) * "'"^primes
end

"""
    @giac_function f(x) = body
    @giac_function V(x[1:2]) = body
    @giac_function F(x, u) = [body1, body2]

Define a symbolic function. The arguments are fresh variables, local to the definition; an
argument written `x[1:n]` is a vector of variables `x1 … xn` (bound by those names as well) and
takes a vector. The body can use variables from the surrounding code, which stay as they are.

The function can be called on expressions (substituting them for the arguments), on exact
numbers (giving an exact result), or on floating-point numbers, which runs a compiled version
and returns a number. `f'` is its derivative; `∇`, `jacobian` and `hessian` apply as well. It
displays as its definition.

```julia
@giac_function f(x) = x^2 * sin(x)
f(y + 1)          # (y+1)^2*sin(y+1)
f(π / 2)          # 2.467…, compiled
f'                # f′(x) = 2*x*sin(x)+x^2*cos(x)

@giac_matrix P[2, 2] symmetric
@giac_function V(x[1:2]) = x' * P * x
∇(V)              # ∇V(x) = [2*p11*x1+2*p12*x2, …]
```
"""
macro giac_function(def)
    (def isa Expr && def.head in (:(=), :function) && def.args[1] isa Expr && def.args[1].head === :call) ||
        throw(ArgumentError("@giac_function expects a definition such as f(x) = x^2"))
    call, body = def.args
    name = call.args[1]
    name isa Symbol || throw(ArgumentError("@giac_function: the function name must be a symbol, got $name"))
    decls, args, argnames = Any[], Any[], Symbol[]
    for a in call.args[2:end]
        if a isa Symbol
            push!(decls, :(local $(esc(a)) = giac_identifier($(string(a)))))
            push!(args, esc(a)); push!(argnames, a)
        elseif a isa Expr && a.head === :ref && a.args[1] isa Symbol
            push!(decls, _localize(_array_declaration(a.args[1], a.args[1], a.args[2:end])))
            push!(args, esc(a.args[1])); push!(argnames, a.args[1])
        else
            throw(ArgumentError("@giac_function: an argument is a name or a vector `x[1:n]`, got $a"))
        end
    end
    return quote
        $(esc(name)) = let
            $(decls...)
            GiacFunction($(QuoteNode(name)), $argnames, Any[$(args...)], $(esc(body)))
        end
    end
end

# declarations as locals, so a definition's arguments don't assign to globals of the same name
_localize(ex::Expr) = ex.head === :block ? Expr(:block, map(_localize, ex.args)...) :
                      ex.head === :(=) ? Expr(:local, ex) : ex
_localize(ex) = ex
