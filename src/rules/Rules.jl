"""
    Giac.Rules

Pattern rewrite rules on Giac expressions.

```julia
using Giac, Giac.Rules
@giac_var x y z
r = @rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
rewrite(x*y + x*z + 1, r)          # x*(y+z)+1
```

`~x` is a slot that matches any subexpression; a slot used twice must match equal subexpressions.
`~x::pred` matches only where `pred(subexpr)` is `true`, and `(~x = d)` stands for `d` where the
subject has nothing for it. `~~xs` is a segment: it matches any number of terms of a sum or
product, or of arguments of a call, and its value is a vector of them; `~~xs::pred` takes the
terms that pass `pred`. `+(~~xs)` and `*(~~xs)` match a whole sum or product. The right side is
ordinary Julia code evaluated with the slots' values, after an optional `where` condition; if it
returns `nothing`, the rule does not apply to that match.

The left side is evaluated by Giac with each slot replaced by a placeholder variable, so a
pattern has the same normal form as the expressions it meets (`~a - ~b` is `a + -b`, `~a / ~b`
is `a * inv(b)`). Sums and products match in any order: each slot takes one term, the last slot
in a nested sum or product takes all the remaining terms, and a slot with nothing left to match
stands for its default, or `0` in a sum and `1` in a product. A power `~x^~n` also matches
`inv(a)` as `a^-1`.

At the top of a rule, terms the pattern does not use are kept: `sin(~x)^2 + cos(~x)^2 => 1`
turns `sin(a)^2 + cos(a)^2 + b` into `b + 1`. A sum matches up to sign and up to a common
factor, using a term in part if its coefficient is larger than needed: the same rule turns
`3sin(a)^2 + 2cos(a)^2` into `2 + sin(a)^2`.

Patterns are also queries ([`@pattern`](@ref), `match`, `occursin`, `findall`, [`@match`](@ref)).
Rules can be written as identities ([`rule`](@ref)), turned around ([`inverse`](@ref)) and
checked ([`verify`](@ref), [`equivalent`](@ref)). Rule sets for identities that hold under
conditions ([`log_expand`](@ref), [`power_expand`](@ref), [`trig_simplify`](@ref), …) use
declared assumptions, and [`simplest`](@ref) searches rewrites by them and by Giac's own
transformations ([`standard_transforms`](@ref)) for the form that best meets a goal.
"""
module Rules

using ..Giac: Giac, GiacExpr, giac_identifier, iscall, arguments, operation, symb_funcname, is_integer, substitute

export @rule, @rules, rewrite
export is_free, depends_on, of_degree, is_positive, is_negative, is_nonnegative, is_real,
       is_integer_valued, in_interval, is_number, multiple_of, is_nonzero
export @pattern, @match
export simplest, expression_size, standard_transforms, Transform, goal_cost
export rule, inverse, equivalent, verify, apply_relations
export identities, IdentityCatalog, function_identities
export log_expand, log_combine, power_expand, power_combine, exp_expand, exp_combine, trig_simplify

include("patterns.jl")
include("matching.jl")
include("rule.jl")
include("predicates.jl")
include("rewrite.jl")
include("queries.jl")
include("identities.jl")
include("library.jl")
include("function_rules.jl")
include("verify.jl")
include("transforms.jl")
include("search.jl")

end # module Rules
