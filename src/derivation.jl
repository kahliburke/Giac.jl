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
    rows = d.rows
    any(r -> _is_note(r) || _is_realign(r), rows) || return print(io, "\$\$", _aligned(rows), "\$\$")
    # Notes are headings, flush left; each run of steps between them is an aligned block,
    # indented. Within an alignment segment (the rows between `@realign` markers) every block's
    # left column is as wide as the segment's widest left side, so the `=` signs line up across
    # its blocks; a new segment starts a new column.
    parts, run = String[], Tuple{String, String}[]
    pad = _segment_pad(rows, 1)
    flush() = isempty(run) || (push!(parts, "\\quad " * _aligned(run, pad)); empty!(run))
    for (k, row) in enumerate(rows)
        if _is_realign(row)
            flush()
            pad = _segment_pad(rows, k + 1)
        elseif _is_note(row)
            flush()
            push!(parts, last(row))
        else
            push!(run, row)
        end
    end
    flush()
    print(io, "\$\$\\begin{array}{l}")
    for (k, p) in enumerate(parts)
        k > 1 && print(io, _is_note_part(p) ? _NOTE_SPACER : _is_note_part(parts[k - 1]) ? " \\\\[8pt] " : " \\\\ ")
        print(io, p)
    end
    print(io, "\\end{array}\$\$")
end

"""
    show(io, MIME"text/markdown", d::Derivation)

The derivation as markdown: each note a paragraph, each run of steps an `aligned` display block.
For renderers without KaTeX's layout commands (a PDF through Typst, say); it can break across
pages where a single block can't.
"""
function Base.show(io::IO, ::MIME"text/markdown", d::Derivation)
    for p in derivation_parts(d)
        haskey(p, :note) ? print(io, p.note, "\n\n") :
            print(io, "\$\$\\begin{aligned}", join(("$l &= $r" for (l, r) in p.steps), " \\\\ "), "\\end{aligned}\$\$\n\n")
    end
end

"""
    derivation_parts(d::Derivation)

The derivation's notes and runs of steps, in order, for a renderer of its own. Each part is
`(note = text,)`, the note as markdown with its `\$…\$` math, or `(steps = rows, segment = k)`, the
`(left, right)` LaTeX pairs of a run of steps. Runs in the same `segment` (the rows between
`@realign` markers) are meant to have their `=` signs line up with each other.
"""
function derivation_parts(d::Derivation)
    parts, run, segment = NamedTuple[], Tuple{String, String}[], 1
    flush() = isempty(run) || (push!(parts, (steps = copy(run), segment)); empty!(run))
    for row in d.rows
        if _is_realign(row)
            flush()
            segment += 1
        elseif _is_note(row)
            flush()
            push!(parts, (note = _note_text(last(row)),))
        else
            push!(run, row)
        end
    end
    flush()
    return parts
end

# a note row's `\text{…}` back as markdown text (its `$…$` math stays math)
_note_text(tex) = replace(chop(tex; head = 6, tail = 1), "\\&" => "&", "\\%" => "%", "\\#" => "#")

# an `@realign` marker: an empty row
const _REALIGN = ("", "")
_is_realign(row) = row == _REALIGN
_is_note(row) = isempty(first(row)) && !_is_realign(row)     # a text row has no left side
_is_note_part(p) = startswith(p, "\\text{")

# An invisible box as wide as the widest left side in the segment starting at row `k` (up to the
# next `@realign`); "" if the segment has no steps.
function _segment_pad(rows, k)
    stop = something(findnext(_is_realign, rows, k), length(rows) + 1) - 1
    lefts = [l for (l, _) in rows[k:stop] if !isempty(l)]
    isempty(lefts) && return ""
    # (a matrix sets its cells in text style, smaller than the steps' display style)
    return "\\hphantom{\\begin{matrix}" * join(("\\displaystyle " * l for l in lefts), " \\\\ ") * "\\end{matrix}}"
end

# Vertical space comes from empty rows. KaTeX treats a positive `\\[gap]` as a minimum depth for
# the row above, not as added space, so after a tall matrix it adds nothing; an empty row always
# adds its height (about 1.5em in `aligned`), and a negative gap is subtracted outright, which
# trims it to the space wanted. (Below a note, a one-line text row, `\\[8pt]` adds its full amount.)
const _STEP_SPACER = " \\\\ \\\\[-0.9em] "
const _NOTE_SPACER = " \\\\ \\rule{0pt}{1em} \\\\ "

"An aligned block of steps; `pad` widens every left side to a common width (see `show`)."
function _aligned(rows, pad = "")
    lines = join(("$l &= $r" for (l, r) in rows), _STEP_SPACER)
    # The pad sits alone in a first row whose height the negative gap takes back (an empty row is
    # about 1.5em): it sets the width of the left column, which aligns its entries to the right.
    # (Beside each left side, under `\mathllap`, a pad taller than one line moves that side down.)
    isempty(pad) || (lines = "$pad & \\\\[-1.5em] " * lines)
    return "\\begin{aligned}" * lines * "\\end{aligned}"
end

"A note as a text row: `&`, `%` and `#` are escaped; `\$…\$` inside it is math."
_text_row(s::AbstractString) = ("", "\\text{" * replace(s, "&" => "\\&", "%" => "\\%", "#" => "\\#") * "}")

Base.show(io::IO, d::Derivation) = print(io, "Derivation with $(length(d.rows)) step(s)")

# Julia syntax → Giac syntax for typesetting a written expression unevaluated.
const _GIAC_NAMES = Dict(:∇ => :grad, :⋅ => :dot, :tr => :trace, :jacobian => :diff, :inv => :inv, :integral => :integrate)

_to_giac_syntax(x) = x
_to_giac_syntax(s::Symbol) = s === :im ? :i : Symbol(replace(string(s), GIAC_RESERVED_LETTERS...))   # Julia's im is GIAC's i
function _to_giac_syntax(ex::Expr)
    ex.head === Symbol("'") && return Expr(:call, :transpose, _to_giac_syntax(ex.args[1]))
    # `C.f(x)` is written `f(x)`; a broadcast `f.(x)` is written `f(x)`
    if ex.head === :call && ex.args[1] isa Expr && ex.args[1].head === :. && ex.args[1].args[2] isa QuoteNode
        return _to_giac_syntax(Expr(:call, ex.args[1].args[2].value, ex.args[2:end]...))
    end
    # `evaluated(f, x => a)` is GIAC's subst(f, x = a); `evaluated(f, x, a, b)` its preval(f, a, b, x)
    if ex.head === :call && ex.args[1] === :evaluated
        a = ex.args[2:end]
        length(a) == 2 && a[2] isa Expr && a[2].head === :call && a[2].args[1] === :(=>) &&
            return Expr(:call, :subst, _to_giac_syntax(a[1]), Expr(:kw, map(_to_giac_syntax, a[2].args[2:3])...))
        length(a) == 4 && return Expr(:call, :preval, map(_to_giac_syntax, a[[1, 3, 4, 2]])...)
    end
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple
        return _to_giac_syntax(Expr(:call, ex.args[1], ex.args[2].args...))
    end
    # an equation `a ~ b` is GIAC's `a = b`
    _is_tilde(ex) && return Expr(:(=), _to_giac_syntax(ex.args[2]), _to_giac_syntax(ex.args[3]))
    args = map(_to_giac_syntax, ex.args)
    if ex.head === :call && args[1] isa Symbol
        args[1] = get(_GIAC_NAMES, ex.args[1], args[1])
    end
    return Expr(ex.head, args...)
end

"""
LaTeX for written Julia code, typeset without evaluating it, in the order it was written.
`values` are the values of names it uses (`name => value`): a derivative of a name whose value
depends on more than the variable it is taken in is written as a partial, `∂L/∂q̇`.
"""
function written_latex(ex, values = ())
    src = string(_to_giac_syntax(ex))
    as_written = _with(_current_style(); order = :as_is, lead = :as_is, products = :as_is)
    held = Dict{String, GiacExpr}(string(_to_giac_syntax(k)) => v for (k, v) in values if v isa GiacExpr)
    return task_local_storage(_WRITTEN_KEY, held) do
        _in_style(() -> _entry_latex(giac_eval("'" * src * "'")), as_written)
    end
end

const _WRITTEN_KEY = :giac_written_values

# Whether the written derivative of `name` in `x` is a partial: the value of `name` depends on
# more variables than `x`. Its variables are its coordinates (q, q̇) and the symbols in it other
# than time and parameters (names declared with an assumption, `m::positive`).
function _written_partial(name::AbstractString, x::AbstractString)
    v = get(get(task_local_storage(), _WRITTEN_KEY, Dict{String, GiacExpr}()), name, nothing)
    v === nothing && return false
    coordinates = [GiacExpr(c) for c in _coordinates!(Node[], _node(v))]
    # a function of `x` itself (φ(x) in x) is not a variable besides `x`
    along(c) = (n = _node(c); (_is_trajectory(n) ? n : n.args[1]).args[2] == SymNode(x))
    vars = Set{String}(something(_wrt_name(c), "") for c in coordinates if !along(c))
    functions = Set(string(_function_name(c)) for c in coordinates if _is_trajectory(_node(c)))
    parameter(s) = s == _current_style().time || s in functions || lock(() -> haskey(_DECLARED, s), _DECLARED_LOCK)
    foreach(s -> parameter(s) || push!(vars, s), (string(s) for s in free_symbols(v)))
    push!(vars, x)
    return length(vars) > 1
end

# The names an expression uses, other than the functions it calls
_names_used(ex) = _names_used!(Symbol[], ex)
_names_used!(out, s::Symbol) = (s in out || push!(out, s); out)
function _names_used!(out, ex::Expr)
    args = ex.head === :call ? ex.args[2:end] : ex.args
    foreach(a -> _names_used!(out, a), args)
    return out
end
_names_used!(out, x) = out

# code giving `name => value` for each name an expression uses that is defined
_values_code(ex) = Expr(:tuple, (:($(QuoteNode(s)) => ($(esc(:(@isdefined $s))) ? $(esc(s)) : nothing)) for s in _names_used(ex))...)

"""
    @derivation begin
        name = expr              # shows  name = value; binds `name`
        φ(x) = expr              # shows  φ(x) = value; binds the function `φ`, so `φ(a)` works
        expr                     # shows  expr (as written) = value
        name := expr             # shows  expr (as written) = value; binds `name`
        lhs ~ expr               # shows  lhs (as written) = value; binds nothing
        lhs ~ mid ~ expr         # shows  lhs = mid (both as written) = value
        cond => result where c   # shows  cond ⇒ result, c; applies `result` from here on
        name := cond => result   # shows  cond ⇒ result; binds the rule `name` instead
        "a note"                 # a line of text
        @realign                 # the steps below line up among themselves
    end
    @derivation simplify begin … end

Run a sequence of steps and typeset them as aligned equations: the left side as written in the
code (typeset by Giac, unevaluated), the right side as computed. Values can be expressions or
arrays of expressions. `lhs ~ rhs` is for left sides that are not assignable, such as
`diff(V, t) ~ ∇(V, x) ⋅ f`; in a chain `lhs ~ mid ~ … ~ expr`, every term but the last is shown
as written. A string is shown as text; write it as `raw"…"` to include `\$…\$` math.

An implication `cond => result` states that the condition (an equation, its left side shown as
written and then as computed) gives the result, an equation such as `k ~ n*π/a`, or a name alone
(`φ(0) ~ 0 => A`) for the condition solved for that name. The result is
substituted into every later value and into the names bound so far, until the derivation ends.
Written as `name := cond => result`, the result is bound to `name`, a rule to apply with
`substitute`, and nothing is substituted. Conditions follow the result after `where` or just a
comma, separated by commas or in brackets: tags as for `@giac_var` (`n::integer`, `n::positive`,
`n::nonzero`, …) and relations (`n != 0`, `n > 0`). They are shown after the result and assumed
from that row on (`≠` is shown only). The comma form (`k ~ n*π/a, n::integer`) also suits tools
that analyse code before macros expand it, which reject `where` outside a type:

```julia
@derivation begin
    φ(x) = A * cos(k * x) + B * sin(k * x)
    φ(0) ~ 0 => A ~ 0
    φ(a) ~ 0 => k ~ n * π / a where n::integer, n != 0
end
```

A written derivative of a name is a partial when the name's value depends on more variables than
the one it is taken in: with `L` bound to a Lagrangian in `q` and `q̇`, `diff(L, q̇)` is shown as
`∂L/∂q̇` and computed as the partial derivative, so a step reads
`EL := diff(diff(L, q̇), t) - diff(L, q)`.

The `=` signs line up across the whole derivation, notes included. `@realign` starts a new
alignment from that point, for when one wide left side would push everything else to the right.

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
_is_pair(ex) = ex isa Expr && ex.head === :call && ex.args[1] === :(=>) && length(ex.args) == 3

# A statement with conditions after `where` separated by commas parses, in a block, as a tuple
# `(statement, condition, …)`: the conditions go back into the result, after the `where`.
function _rejoin_conditions(st)
    (st isa Expr && st.head === :tuple && length(st.args) > 1) || return st
    head, extra = st.args[1], st.args[2:end]
    if head isa Expr && head.head === :(:=)
        return Expr(:(:=), head.args[1], _rejoin_conditions(Expr(:tuple, head.args[2], extra...)))
    end
    imp = _implication(head)
    imp === nothing && return st
    cond, result = imp
    return Expr(:call, :(=>), cond, Expr(:tuple, result, extra...))
end

# `cond => result` as `(cond, result)`, or nothing. `=>` binds more tightly than `~`, so
# `a ~ b => c ~ d` parses as `a ~ ((b => c) ~ d)`: the chain splits at the term holding `=>`.
function _implication(st)
    _is_pair(st) && return (st.args[2], st.args[3])
    _is_tilde(st) || return nothing
    terms = Any[]
    while _is_tilde(st)
        push!(terms, st.args[2])
        st = st.args[3]
    end
    push!(terms, st)
    i = findfirst(_is_pair, terms)
    i === nothing && return nothing
    chain(ts) = foldr((a, b) -> Expr(:call, :~, a, b), ts)
    return (chain([terms[1:(i - 1)]; terms[i].args[2]]), chain([terms[i].args[3]; terms[(i + 1):end]]))
end

# A result and the conditions it is stated under, `… where n::integer, n != 0`. `where` binds to
# the operand just before it, and a comma after it makes a tuple, so the clause is found where it
# lands and taken out.
function _conditions(ex)
    ex isa Expr || return (ex, Any[])
    if ex.head === :where
        conds = ex.args[2:end]
        length(conds) == 1 && conds[1] isa Expr && conds[1].head in (:vect, :braces, :tuple) && (conds = conds[1].args)
        return (ex.args[1], collect(Any, conds))
    end
    # `result, n::integer, n != 0`: the conditions after a comma, with or without `where`
    if ex.head === :tuple && !isempty(ex.args)
        first_, cs = _conditions(ex.args[1])
        return (first_, Any[cs; ex.args[2:end]])
    end
    for i in reverse(eachindex(ex.args))
        inner, cs = _conditions(ex.args[i])
        isempty(cs) && continue
        args = copy(ex.args)
        args[i] = inner
        return (Expr(ex.head, args...), cs)
    end
    return (ex, Any[])
end

# Conditions as (kind, variable or relation, …) tuples for the code that declares them:
# `n::integer` is a tag; `n != 0`, `n > 0` relations
function _conditions_code(conds)
    items = Any[]
    for c in conds
        if c isa Expr && c.head === :(::) && length(c.args) == 2
            push!(items, :((:tag, $(esc(c.args[1])), $(QuoteNode(c.args[2])))))
        elseif c isa Expr && c.head === :call && c.args[1] in (:!=, :≠, :>, :<, :>=, :<=, :≥, :≤) && length(c.args) == 3
            push!(items, :((:relation, $(QuoteNode(c.args[1])), $(esc(c.args[2])), $(esc(c.args[3])))))
        else
            throw(ArgumentError("a condition is a tag such as `n::integer` or a relation such as `n != 0`, got $c"))
        end
    end
    return Expr(:tuple, items...)
end

_conditions_latex_code(conds) = Expr(:vect, (_condition_latex_code(c) for c in conds)...)
function _condition_latex_code(c)
    c.head === :(::) && return :(_tag_latex(written_latex($(QuoteNode(c.args[1]))), $(QuoteNode(c.args[2]))))
    op = Dict(:!= => "\\ne", :≠ => "\\ne", :> => ">", :< => "<", :>= => "\\ge", :≥ => "\\ge", :<= => "\\le", :≤ => "\\le")[c.args[1]]
    return :(written_latex($(QuoteNode(c.args[2]))) * " " * $op * " " * written_latex($(QuoteNode(c.args[3]))))
end

const _TAG_LATEX = Dict(:integer => "\\in \\mathbb{Z}", :real => "\\in \\mathbb{R}", :positive => "> 0",
                        :negative => "< 0", :nonnegative => "\\ge 0", :nonpositive => "\\le 0", :nonzero => "\\ne 0")
_tag_latex(name, tag::Symbol) = haskey(_TAG_LATEX, tag) ? name * " " * _TAG_LATEX[tag] :
    throw(ArgumentError("unknown condition `::$tag`; use one of $(join(keys(_TAG_LATEX), ", "))"))

# Declare what GIAC can assume: the first condition on a name replaces what was assumed about it,
# the others add to it. A name not yet a variable is made one. `≠` is shown but not assumed.
# (Under the GIAC lock, so no other thread sees the assumptions half made; see `_declare!`.)
_declare_conditions!(conds) = with_giac_lock(() -> _declare_each_condition!(conds))

function _declare_each_condition!(conds)
    seen = Set{String}()
    for c in conds
        v = c[1] === :tag ? c[2] : c[3]
        v isa GiacExpr || continue
        first_ = !(string(v) in seen)
        push!(seen, string(v))
        if c[1] === :tag
            c[3] === :nonzero && continue
            tag = c[3]
            if first_
                _declare!(v, tag)                 # recorded, as `@giac_var n::integer` would
            elseif tag in (:integer, :real)
                Commands.invoke_cmd(:additionally, v, giac_eval(string(tag)))
            else
                _assume_more!(v, tag)
            end
        else
            op, rhs = c[2], c[4]
            op in (:!=, :≠) && continue
            rel = op in (:>,) ? v > rhs : op in (:<,) ? v < rhs : op in (:>=, :≥) ? v >= rhs : v <= rhs
            Commands.invoke_cmd(first_ ? :assume : :additionally, rel)
        end
    end
    return nothing
end

_assume_more!(v, tag) = Commands.invoke_cmd(:additionally,
    tag === :positive ? v > 0 : tag === :negative ? v < 0 : tag === :nonnegative ? v >= 0 : v <= 0)

# The row of an implication: the condition's left side as written, its value when that differs,
# its right side, ⇒, the result, the conditions
function _implication_row(left::String, cond, result, conditions::Vector)
    c = cond isa GiacExpr && _is_equation(cond) ? (giac_latex(lhs(cond)), giac_latex(rhs(cond))) : nothing
    head = c === nothing ? giac_latex(cond) : (c[1] == left ? c[2] : c[1] * " = " * c[2])
    tail = isempty(conditions) ? "" : ", \\quad " * join(conditions, ", ")
    # a condition that is its own result, φ(0) = A = 0 giving A = 0, is not stated twice
    c !== nothing && giac_latex(result) == c[1] * " = " * c[2] && return (left, head * tail)
    return (left, head * " \\quad\\Rightarrow\\quad " * giac_latex(result) * tail)
end

# `cond => x`: the condition solved for `x`, which must have one solution
function _solve_for(cond, x)
    r = isolate(cond, x)
    r isa AbstractVector || return r
    throw(ArgumentError("$cond has several solutions for $x ($(join(string.(r), ", "))); state the one that holds"))
end

# The rule a result gives: an equation, applied by substitution
function _rule(r)
    r isa GiacExpr && _is_equation(r) && return r
    r isa AbstractVector && throw(ArgumentError("the result has several values ($(join(string.(r), ", "))); state the one that holds"))
    throw(ArgumentError("the result of `=>` is an equation to apply later, got $r"))
end

# A value with the results established so far substituted in (a function `φ(x) = …`, in its body)
function _apply_rules(v, rules)
    (isempty(rules) || !(v isa GiacFunction)) && return v
    return GiacFunction(v.name, v.argnames, v.args, _apply_rules(v.body, rules))
end
_apply_rules(v::GiacExpr, rules) = isempty(rules) ? v : substitute(v, rules)
_apply_rules(v::AbstractArray{<:GiacExpr}, rules) = isempty(rules) ? v : map(e -> _apply_rules(e, rules), v)

_body(v) = v isa GiacFunction ? v.body : v

_is_docstring(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] == GlobalRef(Core, Symbol("@doc")) &&
                    length(ex.args) == 4 && _is_text(ex.args[3])

# a string, an interpolated string, or `raw"…"` (for `$…$` math without interpolation)
_is_text(ex) = ex isa String || ex isa Expr && (ex.head === :string ||
                                               ex.head === :macrocall && ex.args[1] === Symbol("@raw_str"))

"""
The rows for one step: the left side as written, the terms written after it, and the value. A
value with steps of its own (a rewrite's trace) continues as one row per step.
"""
_derivation_rows(left, written, value, simplifier) =
    [(left, join(vcat(written, giac_latex(_shown(simplifier, value))), " = "))]

# The row for a bound value: name = value, or for an equation the equation itself, aligned on its
# own `=` (`SE = iħ ψ_t ~ …` shows iħ ∂ψ/∂t = …)
function _binding_rows(left, value, simplifier)
    value isa GiacExpr && _is_equation(value) || return _derivation_rows(left, String[], value, simplifier)
    return [(giac_latex(_shown(simplifier, lhs(value))), giac_latex(_shown(simplifier, rhs(value))))]
end

"Apply a display simplifier to a value, entry by entry for arrays."
_shown(f, v::AbstractArray) = map(f, v)
_shown(f, v) = f(v)

function _derivation(simplifier, block)
    stmts = block isa Expr && block.head === :block ? block.args : [block]
    # a string followed by a statement parses as a docstring for it: split them again
    stmts = mapreduce(st -> _is_docstring(st) ? Any[st.args[3], st.args[4]] : Any[st], vcat, stmts; init = Any[])
    rows, rules = gensym(:rows), gensym(:rules)
    out = Any[:($rows = Tuple{String, String}[]), :($rules = GiacExpr[])]
    bound = Symbol[]                  # the names the derivation has bound so far
    # each row is typeset right after its step runs, so a later reassignment can't change it;
    # its value has the results established so far substituted in
    row(lhs, value; written = ()) =
        :(append!($rows, _derivation_rows(written_latex($(QuoteNode(lhs)), $(_values_code(lhs))),
                                          String[$((:(written_latex($(QuoteNode(w)), $(_values_code(w)))) for w in written)...)],
                                          _apply_rules($value, $rules), $simplifier)))
    bind(name, value) = (name in bound || push!(bound, name); :($(esc(name)) = _apply_rules($value, $rules)))
    # the code of a value: π, √2 and 1/2 kept exact rather than computed as floats by Julia
    valued(ex) = esc(_exact_literals(ex))
    # `cond => result where conditions`: shown as cond ⇒ result, conditions; the conditions are
    # declared, and with `apply` the result is substituted into every later value and binding
    function implication(cond, result, name = nothing)
        result, conditions = _conditions(result)
        left = _is_tilde(cond) ? cond.args[2] : cond
        r = gensym(:result)
        # a name alone as the result: the condition solved for it
        solved = result isa Symbol ? :(_solve_for(_apply_rules($(valued(cond)), $rules), $(esc(result)))) : valued(result)
        code = Any[:(_declare_conditions!($(_conditions_code(conditions)))),
                   :($r = $solved),
                   :(push!($rows, _implication_row(written_latex($(QuoteNode(left)), $(_values_code(left))),
                                                   _apply_rules($(valued(cond)), $rules), $r,
                                                   $(_conditions_latex_code(conditions)))))]
        if name === nothing
            push!(code, :(push!($rules, _rule($r))))
            append!(code, (:($(esc(b)) = _apply_rules($(esc(b)), GiacExpr[_rule($r)])) for b in bound))
        else
            push!(code, bind(name, r))
        end
        return code
    end
    for st in stmts
        st isa LineNumberNode && (push!(out, st); continue)
        st = _rejoin_conditions(st)
        if _is_text(st)
            push!(out, :(push!($rows, _text_row($(esc(st))))))
        elseif st isa Expr && st.head === :macrocall && st.args[1] === Symbol("@realign")
            push!(out, :(push!($rows, _REALIGN)))     # a marker read here, never expanded
        elseif st isa Expr && st.head === :(:=) && st.args[1] isa Symbol && _implication(st.args[2]) !== nothing
            # `name := cond => result`: the result bound as a rule to apply where it is wanted
            append!(out, implication(_implication(st.args[2])..., st.args[1]))
        elseif _implication(st) !== nothing
            append!(out, implication(_implication(st)...))
        elseif st isa Expr && st.head === :(:=) && st.args[1] isa Symbol && _is_tilde(st.args[2])
            # `name := lhs ~ rhs`: shows lhs (as written) = rhs, and binds name to the equation
            name, eq = st.args
            push!(out, bind(name, valued(eq)), row(eq.args[2], :(rhs($(esc(name))))))
        elseif st isa Expr && st.head === :(:=) && st.args[1] isa Symbol
            # `name := expr`: shows expr (as written) = value, and binds name to the value
            name, expr = st.args
            push!(out, bind(name, valued(expr)), row(expr, esc(name)))
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Expr && st.args[1].head === :call && st.args[1].args[1] isa Symbol
            # `φ(x) = body`: a function of `x`, shown as its definition
            call, body = st.args
            name = call.args[1]
            define = esc(Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@giac_function")), nothing, Expr(:(=), call, _exact_literals(body))))
            push!(out, bind(name, define), row(call, :(_body($(esc(name))))))
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol
            name, rhs_ = st.args
            push!(out, bind(name, valued(rhs_)),
                  :(append!($rows, _binding_rows(written_latex($(QuoteNode(name))), $(esc(name)), $simplifier))))
        elseif _is_tilde(st)
            # `a ~ b ~ c` parses as `a ~ (b ~ c)`
            terms = Any[st.args[2]]
            rest = st.args[3]
            while _is_tilde(rest)
                push!(terms, rest.args[2])
                rest = rest.args[3]
            end
            push!(out, row(terms[1], valued(rest); written = terms[2:end]))
        else
            push!(out, row(st, valued(st)))
        end
    end
    push!(out, :(Derivation($rows)))
    return Expr(:block, out...)
end
