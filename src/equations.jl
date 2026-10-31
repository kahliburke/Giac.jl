# Equations `a ~ b` as values: their sides, arithmetic on both sides, solving for a variable,
# and substitution with an equation as the rule.

"Whether `e` is an equation `a = b`."
_is_equation(e::GiacExpr) = giac_type(e) == SYMB && symb_funcname(e) == "="
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
lhs(eq::GiacExpr) = _is_equation(eq) ? Commands.left(eq) : _not_an_equation(eq)
"The right side of an equation; see [`lhs`](@ref)."
rhs(eq::GiacExpr) = _is_equation(eq) ? Commands.right(eq) : _not_an_equation(eq)

_not_an_equation(e) = throw(ArgumentError("$e is not an equation; write one as `a ~ b`"))

# A side of an operand: an equation's side, or the operand itself.
_side(f, e) = _is_equation(e) ? f(e) : e

# Arithmetic with an equation applies to both sides: `(a ~ b) - c` is `a - c ~ b - c`, and
# `(a ~ b) + (c ~ d)` is `a + c ~ b + d`. Each side is computed by the ordinary operation, so it
# is simplified as any other result is. `r` is the result Giac gave, an equation exactly when an
# operand was one; checking it keeps the extra cost off ordinary arithmetic.
_both_sides(op, a, b, r) = _is_equation(r) ? op(_side(lhs, a), _side(lhs, b)) ~ op(_side(rhs, a), _side(rhs, b)) : r
_both_sides(op, a, r) = _is_equation(r) ? op(lhs(a)) ~ op(rhs(a)) : r

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
