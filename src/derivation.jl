# `@derivation`: a block of steps typeset as aligned equations, each left side as written and
# each right side as computed.

"""
    Derivation

The typeset steps of a [`@derivation`](@ref) block: one `(left, right)` pair of LaTeX strings
per step. Displays as an aligned block of equations.
"""
struct Derivation
    rows::Vector{Tuple{String, String}}
end

function Base.show(io::IO, ::MIME"text/latex", d::Derivation)
    rows = ("$l &= $r" for (l, r) in d.rows)
    print(io, "\$\$\\begin{aligned}", join(rows, " \\\\[6pt] "), "\\end{aligned}\$\$")
end

Base.show(io::IO, d::Derivation) = print(io, "Derivation with $(length(d.rows)) step(s)")

# Julia syntax → Giac syntax for typesetting a written expression unevaluated.
const _GIAC_NAMES = Dict(:∇ => :grad, :⋅ => :dot, :tr => :trace, :jacobian => :diff, :inv => :inv)

_to_giac_syntax(x) = x
_to_giac_syntax(s::Symbol) = Symbol(replace(string(s), GIAC_RESERVED_LETTERS...))
function _to_giac_syntax(ex::Expr)
    ex.head === Symbol("'") && return Expr(:call, :transpose, _to_giac_syntax(ex.args[1]))
    args = map(_to_giac_syntax, ex.args)
    if ex.head === :call && args[1] isa Symbol
        args[1] = get(_GIAC_NAMES, ex.args[1], args[1])
    end
    return Expr(ex.head, args...)
end

"LaTeX for written Julia code, typeset by Giac without evaluating it."
function written_latex(ex)
    src = string(_to_giac_syntax(ex))
    return _entry_latex(giac_eval("'" * src * "'"))
end

"""
    @derivation begin
        name = expr        # binds `name`; shows  name = value
        expr               # shows  expr (as written) = value
        lhs ~ expr         # shows  lhs (as written) = value; binds nothing
    end
    @derivation simplify begin … end

Run a sequence of steps and typeset them as aligned equations: the left side as written in the
code (typeset by Giac, unevaluated), the right side as computed. Values can be expressions or
arrays of expressions. `lhs ~ rhs` is for left sides that are not assignable, such as
`diff(V, t) ~ ∇(V, x) ⋅ f`.

An optional function before the block (`simplify`, `expand`, `factor`, or any function of one
expression) is applied to each displayed value, entry by entry for arrays. Bound variables keep
the values as computed.

```julia
@giac_var x[1:2]
@giac_matrix P[2, 2] symmetric
@derivation begin
    V = x' * P * x / 2
    ∇(V, x)
end
```
"""
macro derivation(block)
    _derivation(:identity, block)
end

macro derivation(simplifier, block)
    _derivation(esc(simplifier), block)
end

"Apply a display simplifier to a value, entry by entry for arrays."
_shown(f, v::AbstractArray) = map(f, v)
_shown(f, v) = f(v)

function _derivation(simplifier, block)
    stmts = block isa Expr && block.head === :block ? block.args : [block]
    rows = gensym(:rows)
    out = Any[:($rows = Tuple{String, String}[])]
    # each row is typeset right after its step runs, so a later reassignment can't change it
    row(lhs, value) = :(push!($rows, (written_latex($(QuoteNode(lhs))), giac_latex(_shown($simplifier, $value)))))
    for st in stmts
        st isa LineNumberNode && (push!(out, st); continue)
        if st isa Expr && st.head === :(=) && st.args[1] isa Symbol
            name, rhs = st.args
            push!(out, :($(esc(name)) = $(esc(rhs))), row(name, esc(name)))
        elseif st isa Expr && st.head === :call && st.args[1] === :~ && length(st.args) == 3
            push!(out, row(st.args[2], esc(st.args[3])))
        else
            push!(out, row(st, esc(st)))
        end
    end
    push!(out, :(Derivation($rows)))
    return Expr(:block, out...)
end
