# Equations `a ~ b` as values: their sides, arithmetic on both sides, solving for a variable,
# and substitution with an equation as the rule.

"Whether `e` is an equation `a = b`."
_is_equation(e::GiacExpr) = (n = _node(e); n isa CallNode && n.f == "=")
_is_equation(::Any) = false

"""
    lhs(eq)
    rhs(eq)

The left and right sides of the equation `eq` (`a ~ b`). For an array of equations, broadcast:
`rhs.(eqs)`.

```julia
@giac_var x y
eq = 2x + 3 ~ y
lhs(eq)          # 2*x+3
rhs(eq)          # y
```
"""
lhs(eq::GiacExpr) = _is_equation(eq) ? GiacExpr(_node(eq).args[1]) : _not_an_equation(eq)
"The right side of an equation; see [`lhs`](@ref)."
rhs(eq::GiacExpr) = _is_equation(eq) ? GiacExpr(_node(eq).args[2]) : _not_an_equation(eq)

_not_an_equation(e) = throw(ArgumentError("$e is not an equation; write one as `a ~ b`"))

# A side of an operand: an equation's side, or the operand itself.
_side(f, e) = _is_equation(e) ? f(e) : e


"""
    isolate(eq, x)

Solve the equation `eq` for `x` and give the result as an equation `x ~ …`; an expression
`eq` is taken as `eq ~ 0`. With several solutions, a vector of equations, one for each.

```julia
@giac_var x y
isolate(2x + 3 ~ y, x)       # x = (y-3)/2
isolate(x^2 ~ y, x)          # [x = √y, x = -√y]
```
"""
function isolate(eq::GiacExpr, x::GiacExpr)
    sols = to_julia(solve(eq, x))
    isempty(sols) && throw(ArgumentError("$eq has no solution for $x"))
    eqs = [x ~ Commands.simplify(s) for s in sols]
    return length(eqs) == 1 ? only(eqs) : eqs
end
