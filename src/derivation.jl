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
    rows = (isempty(l) ? "&$r" : "$l &= $r" for (l, r) in d.rows)       # a text row has no left side
    print(io, "\$\$\\begin{aligned}", join(rows, " \\\\[6pt] "), "\\end{aligned}\$\$")
end

"A note as a text row: `&`, `%` and `#` are escaped; `\$…\$` inside it is math."
_text_row(s::AbstractString) = ("", "\\text{" * replace(s, "&" => "\\&", "%" => "\\%", "#" => "\\#") * "}")

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
        lhs ~ mid ~ expr   # shows  lhs = mid (both as written) = value
        "a note"           # a line of text
    end
    @derivation simplify begin … end

Run a sequence of steps and typeset them as aligned equations: the left side as written in the
code (typeset by Giac, unevaluated), the right side as computed. Values can be expressions or
arrays of expressions. `lhs ~ rhs` is for left sides that are not assignable, such as
`diff(V, t) ~ ∇(V, x) ⋅ f`; in a chain `lhs ~ mid ~ … ~ expr`, every term but the last is shown
as written. A string is shown as text; write it as `raw"…"` to include `\$…\$` math.

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

_is_tilde(ex) = ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 3

_is_docstring(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] == GlobalRef(Core, Symbol("@doc")) &&
                    length(ex.args) == 4 && _is_text(ex.args[3])

# a string, an interpolated string, or `raw"…"` (for `$…$` math without interpolation)
_is_text(ex) = ex isa String || ex isa Expr && (ex.head === :string ||
                                               ex.head === :macrocall && ex.args[1] === Symbol("@raw_str"))

"Apply a display simplifier to a value, entry by entry for arrays."
_shown(f, v::AbstractArray) = map(f, v)
_shown(f, v) = f(v)

function _derivation(simplifier, block)
    stmts = block isa Expr && block.head === :block ? block.args : [block]
    # a string followed by a statement parses as a docstring for it: split them again
    stmts = mapreduce(st -> _is_docstring(st) ? Any[st.args[3], st.args[4]] : Any[st], vcat, stmts; init = Any[])
    rows = gensym(:rows)
    out = Any[:($rows = Tuple{String, String}[])]
    # each row is typeset right after its step runs, so a later reassignment can't change it
    row(lhs, value; written = ()) =
        :(push!($rows, (written_latex($(QuoteNode(lhs))),
                        join([$((:(written_latex($(QuoteNode(w)))) for w in written)...),
                              giac_latex(_shown($simplifier, $value))], " = "))))
    for st in stmts
        st isa LineNumberNode && (push!(out, st); continue)
        if _is_text(st)
            push!(out, :(push!($rows, _text_row($(esc(st))))))
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol
            name, rhs = st.args
            push!(out, :($(esc(name)) = $(esc(rhs))), row(name, esc(name)))
        elseif _is_tilde(st)
            # `a ~ b ~ c` parses as `a ~ (b ~ c)`
            terms = Any[st.args[2]]
            rest = st.args[3]
            while _is_tilde(rest)
                push!(terms, rest.args[2])
                rest = rest.args[3]
            end
            push!(out, row(terms[1], esc(rest); written = terms[2:end]))
        else
            push!(out, row(st, esc(st)))
        end
    end
    push!(out, :(Derivation($rows)))
    return Expr(:block, out...)
end
