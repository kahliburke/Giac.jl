# Numbers Julia computes in floating point before Giac sees them, written exactly: in `2π * x` the
# product 2π is a Float64, and `sqrt(2)` is 1.414…. For macros whose code is symbolic.

"""
    _exact_literals(ex)

The code `ex` with the constants and literal arithmetic Julia would evaluate in floating point
made exact: `π` and `pi` as Giac's π, `ℯ` as e, `sqrt(2)` and `cbrt(2)` of an integer literal as
roots, `2^(1//3)` of an integer literal to a fractional power as a root, and `1/2` of two integer
literals as a `Rational`. Integer arithmetic and float literals are left as written. Meant for the
symbolic code a macro runs, not for numeric code, where π should stay a float.
"""
_exact_literals(ex) = ex
function _exact_literals(s::Symbol)
    s in (:π, :pi) && return _exact_call(GlobalRef(Base.MathConstants, :π))
    s === :ℯ && return _exact_call(GlobalRef(Base.MathConstants, :ℯ))
    return s
end
function _exact_literals(ex::Expr)
    ex.head in (:quote, :inert, :macrocall) && return ex
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] in (:sqrt, :cbrt) && ex.args[2] isa Integer
        return Expr(:call, GlobalRef(Base, ex.args[1]), _exact_call(ex.args[2]))
    end
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :/ && ex.args[2] isa Integer && ex.args[3] isa Integer
        return Expr(:call, GlobalRef(Base, ://), ex.args[2], ex.args[3])
    end
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :^ && ex.args[2] isa Integer && !(ex.args[3] isa Integer)
        return Expr(:call, GlobalRef(Base, :^), _exact_call(ex.args[2]), _exact_literals(ex.args[3]))
    end
    return Expr(ex.head, map(_exact_literals, ex.args)...)
end

_exact_call(x) = Expr(:call, GlobalRef(Base, :convert), GlobalRef(@__MODULE__, :GiacExpr), x)

"""
    @exact expr

`expr` with its numbers kept exact: `@exact 2π * x` is 2πx with Giac's π rather than 6.283…·x,
and `@exact sqrt(2) / 2` is √2/2. A function cannot do this, as `2π` is a float before any
function sees it. See [`_exact_literals`](@ref Giac._exact_literals) for what is rewritten; a
block works too.

```julia
@giac_var x a::positive
φ = @exact sqrt(2 / a) * sin(2π * x / a)     # √(2/a) sin(2πx/a)
```
"""
macro exact(ex)
    return esc(_exact_literals(ex))
end
