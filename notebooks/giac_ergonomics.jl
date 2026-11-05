try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro title
@md"""
# Giac.jl Ergonomics
## Symbolic math in Julia syntax, with typeset results

Notebook 7 drove Giac.jl hard: symbolic Lyapunov equations, Gröbner reductions, series, typeset derivations. It did that by building Giac commands as strings. This notebook shows the syntax that replaces them, each feature as code beside the result it gives.

The features live in a development branch of Giac.jl (`~/devel/Giac.jl`, loaded here in develop mode), with tests, to be offered upstream.

0. how Giac stores expressions;
1. symbolic vectors and matrices;
2. matrix equations;
3. rewriting in named quantities;
4. typeset derivations;
5. rewrite rules, simplification, and what is known: the identity catalog, properties of functions and matrices;
6. everyday Julia: equality, substitution, linear algebra, solving, calculus, numerics, ODEs, block matrices, assumptions, equations as values, symbolic functions, Taylor polynomials, collecting terms;
7. symbolic matrices: matrices as single symbols, products that do not commute.
"""

#%% code id=setup
using Giac
using Giac.Commands: hold_cmd, release, solve
using Giac.Rules
using GiacSlate
using LinearAlgebra

const C = Giac.Commands
(; giac_jl = pkgdir(Giac), version = pkgversion(Giac))

#%% code id=examples_helper
"Lines of code beside the values they give: rows of `code => value`, or a heading string."
struct Examples
    rows::Vector{Any}
end

# a string followed by a line parses as a docstring for it: split them again
_unwrap_doc(st) = st isa Expr && st.head === :macrocall && st.args[1] == GlobalRef(Core, Symbol("@doc")) ?
                  Any[st.args[3], st.args[4]] : Any[st]

# code as one readable line: no line-number comments, `() -> f(x)` rather than a `begin` block
# (a `do` block keeps its lines)
function _collapse(ex)
    ex isa Expr || return ex
    ex.head === :do && return ex
    ex.head === :macrocall && return Expr(:macrocall, ex.args[1], nothing, map(_collapse, ex.args[3:end])...)
    ex.head === :block && length(ex.args) == 1 && return _collapse(ex.args[1])
    return Expr(ex.head, map(_collapse, ex.args)...)
end
_code(st) = replace(string(_collapse(Base.remove_linenums!(deepcopy(st)))), " ^ " => "^")

# `x = …` or `a, b = …` (not a short function definition `f(x) = …`)
_is_assignment(st) = st isa Expr && st.head === :(=) && !(st.args[1] isa Expr && st.args[1].head in (:call, :where))

# the name `@giac_function f(x) = …` defines, or `nothing`
_defines(st) = st isa Expr && st.head === :macrocall && st.args[1] === Symbol("@giac_function") ?
               st.args[end].args[1].args[1] : nothing

"`@examples begin … end`: show each line of code next to its result; a string line is a heading."
macro examples(block)
    rows = gensym(:rows)
    out = Any[:($rows = Any[])]
    for st in mapreduce(_unwrap_doc, vcat, block.args; init = Any[])
        st isa LineNumberNode && continue
        if st isa String
            push!(out, :(push!($rows, $st)))
        elseif _is_assignment(st) || _defines(st) !== nothing
            lhs, rhs = _is_assignment(st) ? st.args : (_defines(st), st)
            value = gensym(:value)
            # the right side runs inside a `try`; the binding (or destructuring) happens outside
            # it, so later lines can use the names
            push!(out, :($value = try $(esc(rhs)) catch err; err end),
                       :($value isa Exception || ($(esc(lhs)) = $value)),
                       :(push!($rows, $(_code(st)) => $value)))
        else
            push!(out, :(push!($rows, $(_code(st)) => try $(esc(st)) catch err; err end)))
        end
    end
    push!(out, :(Examples($rows)))
    return Expr(:block, out...)
end

_esc_html(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
_value_tex(v) = strip(sprint(show, MIME("text/latex"), v), ['$', '\n', ' '])
_has_tex(v) = showable(MIME("text/latex"), v) && !(v isa AbstractString)
_plain(v) = repr(MIME("text/plain"), v; context = :compact => true)
function _example_value(v)
    v isa Exception && return "<span class=\"ex-err\">" * _esc_html(first(sprint(showerror, v), 160)) * "</span>"
    v isa Tuple && return join(map(_example_value, v), " , ")
    _has_tex(v) && return "\$\\displaystyle " * _value_tex(v) * "\$"
    s = _plain(v)
    # a report over several lines (a certificate, a classification) keeps its lines
    occursin('\n', s) && return "<pre class=\"ex-pre\">" * _esc_html(s) * "</pre>"
    return "<code>" * _esc_html(s) * "</code>"
end

# Slate styles output tables as `.disp.html td`; the selectors carry that prefix to take precedence.
# A result wider than the cell scrolls sideways instead of running past it.
function Base.show(io::IO, ::MIME"text/html", ex::Examples)
    print(io, """<style>
      .disp.html .examples-scroll { overflow-x: auto; max-width: 100%; }
      .disp.html table.examples { border-collapse: collapse; margin: 4px 0; }
      .disp.html table.examples td { padding: 9px 16px; vertical-align: middle; border-bottom: 1px solid rgba(128,128,128,.25); }
      .disp.html table.examples td.code { white-space: pre; font-family: var(--mono, monospace); font-size: .92em; opacity: .9; }
      .disp.html table.examples td.head { padding: 14px 16px 4px; font-weight: 600; border-bottom: none; }
      .disp.html table.examples .ex-err { color: #e06c75; font-family: monospace; font-size: .9em; }
      .disp.html table.examples pre.ex-pre { margin: 0; font-size: .9em; }
    </style><div class="examples-scroll"><table class="examples">""")
    for r in ex.rows
        if r isa String
            print(io, "<tr><td class=\"head\" colspan=\"2\">", _esc_html(r), "</td></tr>")
        else
            print(io, "<tr><td class=\"code\">", _esc_html(first(r)), "</td><td>", _example_value(last(r)), "</td></tr>")
        end
    end
    print(io, "</table></div>")
end

# The same table as markdown, for renderers without HTML (a PDF export typesets this instead of
# taking a picture of the page): code in backticks, results as `$$…$$` display math (so tall
# brackets stretch), headings in bold. A report over several lines is joined with `; `.
_md_cell(s) = replace(s, "|" => "\\|")
_md_code(s) = (t = _md_cell(s); occursin('`', t) ? "`` " * t * " ``" : "`" * t * "`")
function _example_md(v)
    v isa Exception && return _md_code("error: " * first(sprint(showerror, v), 120))
    v isa Tuple && return join(map(_example_md, v), ", ")
    _has_tex(v) && return "\$\$" * replace(_value_tex(v), "|" => "\\vert ") * "\$\$"
    return _md_code(join(strip.(split(_plain(v), '\n')), "; "))
end
function Base.show(io::IO, ::MIME"text/markdown", ex::Examples)
    println(io, "| code | result |\n|:--|:--|")
    for r in ex.rows
        r isa String ? println(io, "| **", _md_cell(r), "** | |") :
                       println(io, "| ", _md_code(first(r)), " | ", _example_md(last(r)), " |")
    end
end

#%% md id=trees_md
@md"""
## 0. What expression trees look like

Arithmetic on `GiacExpr` (`+ - * / ^`) happens in Julia, on trees in a normal form: a sum is a list of distinct terms with numeric coefficients, a product a numeric coefficient and distinct bases with exponents. Like terms merge as the expression is built (`x + x` is `2x`, `x * x` is `x²`); nothing is expanded, and terms keep the order they were written in. GIAC is called for everything else (`simplify`, `factor`, `solve`, …), and each form is made from the other when it is first needed.

Code that walks expressions (rules, the typesetter, derivations) sees the same shapes GIAC uses, with its operation names:

- sums and products are **n-ary and flattened**;
- subtraction is a sum with a negated term (`a + -b`);
- division is multiplication by an inverse (`a * inv(b)`), and a numeric coefficient is a leading factor (`a/2` is `1/2 * a`);
- numbers stay as numeric leaves.

A rule written for `a - b` therefore matches `a + (-b)`.
"""

#%% code id=trees
"The tree of a Giac expression as a Julia `Expr`, using Giac's operation names."
tree(e) = Giac.iscall(e) ? Expr(:call, Symbol(Giac.symb_funcname(e)), tree.(Giac.arguments(e))...) :
          Symbol(string(e))

let
    @giac_var a b c x
    shapes = ["a - b" => a - b, "a + b + c" => a + b + c, "a + (b + c)" => a + (b + c), "-a" => -a,
              "a*b*c" => a * b * c, "a/b" => a / b, "a^2" => a^2, "2a" => 2a,
              "a*b - c*x" => a * b - c * x, "sin(a)" => C.sin(a), "a/2" => a / 2]
    [(typed = s, tree = string(tree(e))) for (s, e) in shapes]
end

#%% md id=matrix_probe_md
@md raw"""
## 1. Symbolic vectors and matrices

Symbolic arrays are ordinary Julia arrays of `GiacExpr`. Julia's `LinearAlgebra` does the bookkeeping, calling `+` and `*` on entries, and Giac evaluates them. `det`, `inv` and `\` hand the whole matrix to Giac.

- `@giac_var x[1:2]` declares the vector `x = [x1, x2]` and binds `x1`, `x2`.
- `@giac_matrix A[2, 2]` declares `A = [a11 a12; a21 a22]`; the entries are named after the lowercased matrix name.
- `symmetric` shares the off-diagonal entries, as a Lyapunov $P$ needs.

What changed in Giac.jl to make this work:
- `A + I`: `I`'s `Bool` scale converts to a number, not a Giac boolean.
- `s*I`: Julia's `UniformScaling` only holds a `Number`, so a small `SymbolicScaling` type stands in.
- `det`/`inv`: they no longer go through Julia's pivoting LU, which compares entries.
- `@giac_var x` no longer picks up a value that some Giac context assigned to `x`.
"""

#%% code id=decl
let
    @giac_var x[1:2] s
    @giac_matrix A[2, 2] P[2, 2] symmetric
    @examples begin
        "declarations bind the array and its entries"
        x
        A
        P
        a12
        "matrix algebra, with Giac doing the entries"
        A'
        A + I
        s * I - A
        P * A + A' * P
        x' * P * x
        Giac.Commands.expand(x' * P * x)
        "determinant, inverse, solve, trace, slices"
        det(A)
        inv(A)
        A \ x
        tr(A)
        A[:, 1]
        A[2, :]'
    end
end

#%% md id=giacmatrix_examples
@md"""
`GiacMatrix`, the matrix type stored inside Giac, is now an `AbstractMatrix{GiacExpr}` as well. Slicing, `A'`, broadcasting and all of the above work on it too. The generated Giac commands no longer take over functions such as `axes` and `copy` that Julia's array code relies on. Operations that Giac implements for whole matrices (`*`, `+`, `det`, `inv`, `transpose`) return a `GiacMatrix`; the generic ones return a `Matrix`:
"""

#%% code id=giacmatrix_demo
let
    @giac_var a b c d u v
    M = GiacMatrix([a b; c d])
    w = GiacMatrix([u, v])
    @examples begin
        M
        M'
        M + I
        M * w
        w' * M * w
        M[:, 2]
        M .^ 2
        det(M)
        inv(M)
        typeof(inv(M)), typeof(M + I)
    end
end

#%% md id=solve_md
@md"""
## 2. Matrix equations

`A ~ B` between arrays builds one equation per entry, including against `I` or a scalar. `solve(eqs, X)` solves for the free entries of a matrix unknown `X` and returns `X` filled in. A symmetric `P` produces the $(1,2)$ and $(2,1)$ equations twice, and repeats count once.
"""

#%% code id=solve_matrix
let
    @giac_matrix A[2, 2] P[2, 2] symmetric
    @giac_var b
    @examples begin
        "the Lyapunov equation, entry by entry"
        P * A + A' * P ~ -I
        "solved for the free entries of P"
        P★ = solve(P * A + A' * P ~ -I, P)
        Giac.Commands.simplify.(P★ * A + A' * P★)
        Giac.Commands.factor(det(P★))
        "a concrete stable A gives a concrete P"
        Ab = GiacExpr[-b -1; 1 0]
        solve(P * Ab + Ab' * P ~ -I, P)
    end
end

#%% md id=express_md
@md raw"""
## 3. Rewriting in named quantities: `express`

`express(e, τ => tr(A), Δ => det(A))` rewrites `e` in terms of the names on the left wherever the definitions allow. It reduces the numerator and denominator by Gröbner basis, eliminating the original variables first, then factors. It broadcasts over arrays.

`Δ` and `Σ` are operators in Giac's language, so `@giac_var Δ` creates the identifier `Delta`, which typesets as $\Delta$.
"""

#%% code id=express_demo
let
    @giac_matrix A[2, 2] P[2, 2] symmetric
    @giac_var τ Δ
    P★ = solve(P * A + A' * P ~ -I, P)
    @examples begin
        express(Giac.Commands.expand((a11 + a22)^2), τ => tr(A), Δ => det(A))
        express(a11 * a22 - a12 * a21 + 1, τ => tr(A), Δ => det(A))
        express(1 / (a11 + a22), τ => tr(A), Δ => det(A))
        "the Lyapunov solution in the invariants"
        express(det(P★), τ => tr(A), Δ => det(A))
        express.(P★, τ => tr(A), Δ => det(A))
    end
end

#%% md id=derivation_md
@md"""
## 4. Derivations: `@derivation`

`@derivation` runs a block of steps and typesets them as aligned equations: the left side as written in the code (typeset by Giac, unevaluated), the right side as computed.

- `name = expr` binds `name` and shows `name = value`;
- a bare expression shows itself `= value`;
- `lhs ~ expr` shows `lhs = value`, for left sides that aren't assignable, like `diff(V, t)`;
- a chain `lhs ~ mid ~ expr` shows every term but the last as written;
- a string is a line of text (use `raw"…"` for `$…$` math).

An optional first argument such as `expand` tidies each displayed value without changing the bound ones. Here is notebook 7's symbolic core in one block:
"""

#%% code id=derivation_demo
@giac_var x[1:2] h[1:2] t τ Δ
@giac_matrix A[2, 2] P[2, 2] symmetric
E = C.expand

@derivation begin
    V = x' * P * x / 2
    V ~ E(V)
    ∇(V, x) ~ E.(∇(V, x))
    diff(V, t) ~ E(∇(V, x) ⋅ (A * x + h))
    P★ = solve(P * A + A' * P ~ -I, P)
    P★ * A + A' * P★ ~ C.simplify.(P★ * A + A' * P★)
    det(P★) ~ express(det(P★), τ => tr(A), Δ => det(A))
end

#%% md id=rules_md
@md"""
## 5. Rewrite rules

`express` covers one kind of rewriting. Everything else (collecting a common factor, applying a trig identity, renaming a combination) is pattern rules, in the submodule `Giac.Rules` (`using Giac.Rules`).

- `@rule lhs => rhs` makes a rule. `~x` matches any subexpression, a slot used twice must match equal ones, and `~x::pred` matches only where `pred` is `true`.
- The right side is plain Julia run with the slots' values. Returning `nothing` declines the match.
- Patterns are evaluated by Giac with placeholders for the slots, so they have the same normal form as the expressions they meet (`~a - ~b` is stored as `a + -b`).
- Sums and products match in any order. A slot with nothing left to match stands for $0$ or $1$. Terms a rule doesn't use are kept, and a sum also matches up to sign.
- `rewrite(e, rules...)` applies rules bottom-up until nothing changes, over arrays too. `@rules begin … end` collects several.
"""

#%% code id=rules_demo
let
    @giac_var a b x y z w
    factor_out = @rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
    pythagoras = @rule sin(~t)^2 + cos(~t)^2 => 1
    merge_exp = @rule exp(~p) * exp(~q) => exp(~p + ~q)
    names_only = @rule exp(~p::is_identifier) * exp(~q::is_identifier) => exp(~p + ~q)
    @examples begin
        "a rule displays as what it does"
        factor_out
        pythagoras
        "common factors, with leftover terms kept"
        rewrite(x * y + x * z + w, factor_out)
        rewrite(x - x * y, factor_out)
        rewrite(2x * y + 2x * z, factor_out)
        "identities apply inside longer sums, inside functions, and up to sign"
        rewrite(sin(a)^2 + cos(a)^2 + b, pythagoras)
        rewrite(exp(sin(a)^2 + cos(a)^2), pythagoras)
        rewrite(-sin(a)^2 - cos(a)^2, pythagoras)
        rewrite(sin(a)^2 + cos(b)^2, pythagoras)
        "products in any order; predicates restrict a slot"
        rewrite(exp(x) * exp(y) * z, merge_exp)
        rewrite(exp(x) * exp(y) + exp(x) * exp(a + b), names_only)
    end
end

#%% md id=rules_lyap_md
@md raw"""
### Rules and `express` together: notebook 7's $P^\star$ in $\tau$ and $\Delta$

Rules match **syntax**, with one piece of algebra built in: a sum matches **up to sign**. `factor` writes the determinant factor as $a_{12}a_{21} - a_{11}a_{22}$, the negative of the $\Delta$ rule's pattern, and the rule still applies, giving $-\Delta$. (If a rule rewrites $-e$ to $v$, then $e$ is $-v$, so this is always sound.)

The numerator of $\det P^\star$ is a different job: it hides $\tau^2$ inside a polynomial, and finding it takes algebra, which is what `express` does (a Gröbner reduction). A perfect-square rule set then finishes it. Each tool does what it is good at.

Rule sets display as the rules they contain:
"""

#%% code id=rules_lyap
invariant_rules, square_rules = let
    @giac_matrix A[2, 2]
    @giac_var τ Δ
    (@rules begin
         a11 + a22 => τ
         a11 * a22 - a12 * a21 => Δ
     end),
    (@rules begin
         ~p^2 + 2 * ~p * ~q + ~q^2 => (~p + ~q)^2
         ~p^2 - 2 * ~p * ~q + ~q^2 => (~p - ~q)^2
     end)
end
[invariant_rules; square_rules]

#%% code id=rules_lyap_derivation
let
    @giac_matrix A[2, 2] P[2, 2] symmetric
    @giac_var τ Δ
    @derivation begin
        P★ = solve(P * A + A' * P ~ -I, P)
        "naming the invariants in the factored entries; the sign-flipped determinant matches too"
        P★ ~ rewrite(C.factor.(P★), invariant_rules)
        raw"$\tau^2$ is hidden in the numerator: express finds it, the square rules tidy the rest"
        det(P★) ~ rewrite(express(det(P★), τ => tr(A), Δ => det(A)), square_rules)
    end
end

#%% code id=0ebb7a
@examples begin
    det(P★) 
    express(det(P★), τ => tr(A), Δ => det(A))
    P == P'
    λ, V = eigen(P)
    λ
    V
    C.simplify.(V * Diagonal(λ) * inv(V))
    tr(A)
    A / P
    A \ P
     [A I; I A']
    rewrite(express(det(P★), τ => tr(A), Δ => det(A)), square_rules)
end

#%% md id=rules_status
@md"""
**In Giac.jl:** the submodule `Giac.Rules` (`src/rules/`), tests in `test/test_rules.jl`.

Rules typeset with their slots as plain names, and a `where` condition shows with the rule. A right side that is Julia logic (a conditional, a comparison) is shown as code. The sections below add segments, conditions, strategies, matching up to a common factor, queries, the identity catalog, and the search for the simplest form.
"""

#%% md id=rules_more_md
@md"""
### Segments, conditions, and traced rewriting

- **Segments.** `~~xs` matches any number of terms of a sum or product, or a run of a call's arguments, and its value is a vector. `+(~~ts)` matches a whole sum (of two or more terms), so one rule distributes over sums of any length.
- **Conditions.** `where` after the right side is a condition on the slots, shown with the rule. Comparisons Giac can decide count (`~n > 3` for a number), and `is_free(~a, ~x)` says `a` doesn't involve `x`.
- **Strategies.** `rewrite(…; order = :topdown)` rewrites from the outside in, and `once = true` makes a single pass. `rewrite(rules)` without an expression is a function, so rule sets compose with `|>`.
- **Traces.** `trace = true` returns the rewriting as steps, one rule in one place per step, each labelled with its rule.
"""

#%% code id=rules_more_demo
let
    @giac_var x y z t a
    distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
    complete_square = @rule ~a * ~x^2 + ~b * ~x + ~~rest =>
        ~a * (~x + ~b / (2 * ~a))^2 - ~b^2 / (4 * ~a) + +(~~rest) where is_free(~a, ~x) && is_free(~b, ~x)
    above_cubic = @rule ~x^~n => 0 where ~n > 3
    pythagoras = @rule sin(~t)^2 + cos(~t)^2 => 1
    @examples begin
        "rules"
        distribute
        complete_square
        above_cubic
        "segments and conditions"
        distribute(a * (x + y + z))
        complete_square(2t^2 + 3t + 7)
        complete_square(a * t^2 + t)
        complete_square(sin(t) * t^2 + t)
        rewrite(1 + t + t^2 / 2 + t^3 / 6 + t^4 / 24 + t^5 / 120, above_cubic)
        "composing rule sets"
        (sin(t)^2 + cos(t)^2 + a * (x + y)) |> rewrite(pythagoras) |> rewrite(distribute)
    end
end

#%% code id=rules_trace
# every step of a rewrite, with the rule that made it
let
    @giac_var x y t a
    distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
    pythagoras = @rule sin(~t)^2 + cos(~t)^2 => 1
    rewrite((sin(t)^2 + cos(t)^2) * x + a * (x + y), distribute, pythagoras; trace = true)
end

#%% md id=rules_power_md
@md"""
### Stronger matching, queries, identities, search

- **Common factors.** A sum matches up to a common factor, using a term in part when its coefficient is larger than needed: `sin²t + cos²t → 1` also turns `3sin²t + 2cos²t` into `2 + sin²t`. A product matches up to its numeric coefficient, so `2 sin t cos t → sin 2t` also handles `sin t cos t`.
- **Defaults.** `(~x = d)` stands for `d` where the subject has nothing for it. `~v^(~n = 1)` matches `v` itself, and a power also matches `1/v` as `v⁻¹`. A sum term written `(~b = 0) * x` may be absent.
- **Segments with a test.** `~~lin::of_degree(1, x) + ~~rest` splits a sum by a property.
- **Queries.** `@pattern`, `match`, `occursin`, `findall`, and `@match` for branching on an expression's shape.
- **Identities with conditions.** `log_expand()`, `power_combine()`, `trig_simplify()` and the others hold only under conditions, which Giac checks using declared assumptions.
- **Search.** `simplest(e, rules…, expand, factor)` looks for the shortest form, even when the way there passes through a longer one. `trace = true` gives the steps it took.
"""

#%% code id=rules_power_demo
let
    @giac_var x y t a
    @giac_var q[1:2]
    pythagoras = @rule sin(~t)^2 + cos(~t)^2 => 1
    merge = @rule ~v^(~n = 1) * ~v^(~m = 1) => ~v^(~n + ~m)
    linear_part = @rule +(~~lin::of_degree(1, q)) + ~~rest => +(~~lin)
    quadratic = @pattern ~a * x^2 + (~b = 0) * x + ~c
    shape(e) = @match e begin
        ~a * x^2 + (~b = 0) * x + ~c => "quadratic, leading coefficient $(~a)" where is_free(~a, x) && is_free(~b, x) && is_free(~c, x)
        ~a * x + ~b => "linear, slope $(~a)" where is_free(~a, x) && is_free(~b, x)
        _ => "something else"
    end
    @examples begin
        "common factors"
        pythagoras(3sin(t)^2 + 3cos(t)^2)
        pythagoras(3sin(t)^2 + 2cos(t)^2)
        rewrite(sin(t) * cos(t), trig_simplify())
        "defaults and segments with a test"
        merge(x * x^2)
        linear_part(q1 + 2q2 + q1 * q2 + 7)
        "queries"
        m = match(quadratic, 3x^2 + 1)
        m[:a], m[:b], m[:c]
        findall(@pattern(sin(~u)), sin(x) + cos(sin(y)))
        shape(a * x^2 + 1), shape(2x + y), shape(x^3)
    end
end

#%% code id=rules_identities
let
    @giac_var x y p::positive w::positive
    demo = @examples begin
        "identities that hold under conditions"
        log_expand()
        rewrite(log(p * w), log_expand()), rewrite(log(x * y), log_expand())
        rewrite(sqrt(p) * sqrt(w), power_combine()), rewrite(sqrt(x) * sqrt(y), power_combine())
        rewrite(2log(p) + 2log(w), log_combine())
    end
    @giac_var p w      # assumptions are global per name: drop them for the other cells
    demo
end

#%% code id=rules_simplest
# no rule applies to sin⁴t − cos⁴t as written; the search finds a way through a longer form
let
    @giac_var t
    simplest(sin(t)^4 - cos(t)^4, trig_simplify(), C.factor, C.expand; trace = true)
end

#%% md id=toolkit_md
@md"""
### A simplification toolkit

- **`simplest(e)`** with no further arguments searches `standard_transforms()`. That set holds Giac's rewriters that are valid for every value (normal form, factoring, partial fractions, the trig forms `tlin`/`tcollect`/`trigsin`/…, `exp(n log x) = xⁿ`) and the rule sets whose identities hold only under conditions. Giac's `simplify` and `lnexpand` are left out, because they combine and split logarithms without regard to sign.
- **`goal`** says what simplest means: `:compact`, `:expanded`, `:factored`, `:trig_linear`. A `cost` function can replace it.
- **Identities as rules.** `rule(sin(2x) ~ 2sin(x)cos(x), x)` makes `x` a slot, and `inverse(r)` turns a rule around.
- **Checking.** `equivalent(a, b)` compares symbolically, then at random points chosen within the declared assumptions. `verify(r)` checks that a rule rewrites to an equal expression. `simplest(…; verify = true)` takes only checked steps.
- **Substituting a relation.** `substitute(e, x^2 + y^2 => r^2)` works where Giac's own substitution only replaces variables. It matches occurrences up to a common factor, and falls back to a Gröbner reduction when `e` needs algebra to be written in terms of `r`.
"""

#%% code id=toolkit_demo
let
    @giac_var x y z t r
    double = rule(sin(2x) ~ 2sin(x) * cos(x), x)
    @examples begin
        "simplest with no further arguments"
        simplest((x^2 - 1) / (x^2 + 2x + 1))
        simplest(exp(2log(x)) + exp(x) * exp(y))
        simplest(2sin(t) * cos(t) + 3sin(t)^2 + 3cos(t)^2)
        "goals"
        simplest(sin(t)^3; goal = :trig_linear)
        simplest((x + 1)^2 * (x - 1); goal = :expanded)
        simplest(x^2 - 1; goal = :factored)
        "the path, when you want it"
        path = simplest(sin(t)^4 - cos(t)^4; trace = true)
        first(path), path[2], last(path)
        length(path), path.by[1]
        "identities as rules"
        double
        inverse(double)
        inverse(double)(sin(t) * cos(t))
        "checking"
        equivalent(sin(t)^2 + cos(t)^2, 1), equivalent(log(x * y), log(x) + log(y))
        verify(double), verify(rule(sin(2x) ~ sin(x) * cos(x), x))
        "substituting a relation"
        substitute(2x^2 + 2y^2 + z, x^2 + y^2 => r^2)
        substitute(x^4 + 2x^2 * y^2 + y^4, x^2 + y^2 => r^2)
    end
end

#%% code id=toolkit_derivation
# a search's steps continue a derivation
let
    @giac_var t
    E = sin(t)^4 - cos(t)^4 + 2sin(t) * cos(t)

    @derivation begin
        "the energy-like term, simplified by search"
        E ~ simplest(E; trace = true)
    end
end

#%% md id=facts_md
@md"""
### What is known: identities, function properties, matrix properties

- **An identity catalog.** `identities(topics...)` lists identities Giac doesn't apply on its own, grouped by topic: trigonometric and hyperbolic (one generator writes both, by Osborn's rule), their linear forms, inverse-trig compositions, exp/log/power, integer multiples. Each holds under the condition shown with it, and the test suite checks every entry with `verify`. Identities Giac already applies, given their conditions, are left out. `direction = :expand` turns them around.
- **Function properties.** `@giac_var φ(x)::odd ψ(t)::(even, periodic(2π)) L(s)::linear g(x)::inverse_of(f)` records what's known about a function. `function_identities()` gives the rules, and `simplest` uses them.
- **Matrix structure and properties.** `@giac_matrix W[3,3]::skew`, `::diagonal`, `::upper`, `::lower` shape the entries. `::orthogonal` records the relations QᵀQ = QQᵀ = I. `::posdef` assumes the diagonal entries positive and records the leading minors as positive. `::invertible` records det ≠ 0. Relations are applied by Gröbner reduction (`apply_relations`, and in `simplest` and `equivalent`). Signs feed `is_positive`, `is_nonzero` and the other tests.
"""

#%% code id=facts_catalog
# the catalog, by topic (trig_linear and the other lengthening topics on request)
identities(:trig, :hyperbolic, :inverse_trig, :exp, :log, :power, :integer)

#%% code id=facts_demo
let
    @giac_var t a x y
    @giac_var φ(t)::odd ψ(t)::(even, periodic(2π))
    @giac_matrix Q[2, 2]::orthogonal P[2, 2]::posdef W[3, 3]::skew
    @giac_var z[1:3]
    φ_, ψ_ = Giac.giac_identifier("φ"), Giac.giac_identifier("ψ")
    π_ = convert(GiacExpr, π)
    demo = @examples begin
        "the catalog at work"
        simplest(sin(x) * cos(y) + cos(x) * sin(y))
        simplest(cosh(x)^2 + sinh(x)^2)
        simplest(cosh(asinh(x)))
        "an odd nonlinearity and a periodic potential"
        function_identities()
        simplest(φ_(-(t^2 + 1)) + φ_(t^2 + 1))
        simplest(ψ_(-t - 2π_) - ψ_(t))
        "matrices"
        W
        simplest(z' * W * z)
        equivalent(Q' * Q, I), simplest(det(Q)^2)
        is_positive(det(P)), is_positive(p12)
    end
    @giac_var φ(t) ψ(t)                       # drop the properties for the other cells
    @giac_matrix Q[2, 2] P[2, 2]
    demo
end

#%% md id=idioms_md
@md"""
## 6. Everyday Julia

The idioms any Julia user reaches for without thinking: equality and `Dict`s, substitution, numbers, `LinearAlgebra`, solving and calculus.

- `==` is mathematical equality, decided by simplifying the difference. `isequal` and `hash` are structural, so `Dict`, `Set` and `unique` behave.
- `substitute` takes a vector of variables with a vector or tuple of values. A tuple keeps exact constants: `[π, 0]` is a `Vector{Float64}` before Giac sees it.
"""

#%% code id=idioms_probe
let
    @giac_var x y
    @giac_var q[1:2]
    A = GiacExpr[0 1; -2 -3]
    e = (x + 1)^2
    @examples begin
        "equality"
        x + 1 == 1 + x
        sin(x)^2 + cos(x)^2 == 1
        isequal(x + 1, x + 1), isequal(x + 1, 1 + x)
        unique([x, x, y])
        Dict(x => 1)[x]
        "numbers"
        Float64(sqrt(convert(GiacExpr, 2)))
        Rational(giac_eval("3/4"))
        "substitution"
        substitute(e, x => 2)
        e(x => y)
        substitute(q1 + q2^2, q => [1, 2])
        substitute(sin(q1) + q2, q => (π, 0))
        substitute(A * q, q => [1, 1])
    end
end

#%% code id=idioms_probe2
let
    @giac_var s t k
    A = GiacExpr[0 1; -2 -3]
    @examples begin
        A
        "spectra, from Giac's Jordan form"
        eigvals(A)
        eigvals(GiacExpr[0 1; -k -3])
        eigen(A)
        det(s * I - A)
        "rank and null space"
        rank(GiacExpr[1 2; 2 4])
        nullspace(GiacExpr[1 2; 2 4])
        "the state-transition matrix and the resolvent"
        exp(A * t)
        inv(s * I - A)
        [1 0] * inv(s * I - A) * [0, 1]
    end
end

#%% code id=solve_forms_probe
let
    @giac_var x s t k
    @giac_var q[1:2]
    F = [q2, -sin(q1) - q2]                                  # damped pendulum
    G = [q1 * (1 - q1 - q2), q2 * (q1 - 1 // 2)]             # a polynomial system
    @examples begin
        "solving: solution lists display as sets"
        solve(x^2 ~ 4, x)
        solve(G ~ 0, q)
        solve(F ~ 0, q)
        "assumptions, scoped to a block"
        assuming(() -> sqrt(k^2), k > 0)
        sqrt(k^2)
        "calculus"
        diff((x + 1)^2, x)
        Giac.Commands.integrate((x + 1)^2, x, 0, 1)
        Giac.Commands.limit(sin(x) / x, x, 0)
        Giac.Commands.series(exp(t), t, 0, 3)
        Giac.Commands.laplace(exp(-t), t, s)
        Giac.Commands.ilaplace(1 / (s^2 + 3s + 2), s, t)
        Giac.Commands.partfrac(1 / (s^2 + 3s + 2), s)
    end
end

#%% md id=idioms_status
@md raw"""
### From model to numbers

A damped pendulum, $\ddot q = -2\sin q - 3\dot q + u$, in control-affine form. One block covers its equilibria, linearisations and eigenvalues (a sink at the bottom, a saddle at the top), and the state-transition matrix of the stable one:
"""

#%% code id=pendulum_showcase
@giac_var q[1:2] u t
pend_f = [q2, -2sin(q1) - 3q2 + u]

@derivation begin
    f = pend_f
    "equilibria with no input"
    q̄ = solve(substitute(f, u => 0) ~ 0, q)
    J = jacobian(f, q)
    A₀ = substitute(J, q => (0, 0))
    A₁ = substitute(J, q => (π, 0))
    "a sink at the bottom, a saddle at the top"
    λ₀ = eigvals(A₀)
    λ₁ = eigvals(A₁)
    raw"the state-transition matrix at the bottom, solving $\dot{x} = A_0 x$"
    Φ = exp(A₀ * t)
    diff(Φ, t) ~ A₀ * Φ ~ C.expand.(A₀ * Φ)
end

#%% code id=pendulum_compiled
# the same model as a fast numeric right-hand side, and the code it compiles to
let
    F = build_function(pend_f, q, u; backend = :julia)
    Fg = build_function(pend_f, q, u)                          # through Giac on every call
    x0 = [0.1, 0.0]
    F(x0, 0.0); Fg(x0, 0.0)
    per_call(g, n) = 1e6 * (@elapsed for _ in 1:n; g(x0, 0.0); end) / n
    (code = F.code,
     value = F(x0, 0.0),
     same_as_giac_backend = F(x0, 0.0) ≈ Fg(x0, 0.0),
     μs_per_call = (julia = round(per_call(F, 100_000), sigdigits = 2), giac = round(per_call(Fg, 200), sigdigits = 2)))
end

#%% code id=showcase_edges_probe
let
    @giac_var A₀ x̄ P★ q′ θ̇ τ Δ x t
    @examples begin
        "names typeset as they read"
        A₀ + x̄
        P★ * q′
        θ̇^2 + τ * Δ
        "series remainders and exact constants"
        Giac.Commands.series(sin(x) / x, x, 0, 4)
        convert(GiacExpr, π) * x
    end
end

#%% code id=ode_probe
let
    @giac_var t
    @giac_var y(t)
    ẏ = Differential(t)(y)
    ode = diff(y, t, 2) + 3diff(y, t) + 2y ~ 0
    @examples begin
        "a function of t, its derivatives, and an ODE"
        y
        diff(y, t)
        ode
        "general solution, and with initial conditions"
        Giac.Commands.desolve(ode, t, y)
        Giac.Commands.desolve([ode, y(0) ~ 1, ẏ(0) ~ 0], y)
    end
end

#%% md id=desolve_status
@md"""
`@giac_var y(t)` binds `y` to `y(t)`. `desolve` passes such an unknown to Giac by its name `y`, which Giac needs to apply initial conditions, and reads the variable `t` from it, where Giac would otherwise assume `x`. Initial conditions on derivatives use `Differential(t)(y)(0) ~ 0`, which stays unevaluated until `desolve` sees it.

The same in a derivation, checked, and through the Laplace transform:
"""

#%% code id=ode_demo
let
    @giac_var t s
    @giac_var y(t)
    ẏ = Differential(t)(y)
    @derivation begin
        raw"a damped oscillator released from rest at $y = 1$"
        ode = diff(y, t, 2) + 3diff(y, t) + 2y ~ 0
        y★ = C.desolve([ode, y(0) ~ 1, ẏ(0) ~ 0], y)
        "check: the residual and the initial value"
        r = C.simplify(diff(y★, t, 2) + 3diff(y★, t) + 2y★)
        y★0 = substitute(y★, t => 0)
        "the same solution through the Laplace transform"
        Y = C.laplace(y★, t, s)
        C.partfrac(Y, s)
    end
end

#%% md id=blocks_md
@md"""
### Block matrices

Concatenation (`[A B; C D]`) gives an ordinary matrix: the entries are all there, but the blocks are forgotten. [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) keeps the partition: a `BlockMatrix` is still an `AbstractMatrix`, so the symbolic algebra works on it, and it also knows its blocks. Blocks are addressed as `M[Block(i, j)]`, and products of conforming block matrices stay partitioned.

Giac.jl has an extension that loads with BlockArrays. It draws the partition lines and lets block products start from symbolic zeros. Two pieces of Giac.jl itself were needed too: an expression broadcasts as a scalar (`t .* A`), and the 5-argument `mul!(C, A, B, α, β)` takes symbolic `α` and `β`.
"""

#%% code id=blocks_using
using BlockArrays          # loads Giac.jl's BlockArrays extension: partition lines, block products

#%% code id=blocks_demo
let
    @giac_matrix A[2, 2] B[2, 2]
    @giac_var b[1:2] c[1:2] d
    @examples begin
        "a partition of an assembled matrix"
        M = BlockArray([A I; I A'], [2, 2], [2, 2])
        M[Block(1, 2)]
        blocksize(M)
        "products of conforming block matrices stay partitioned"
        M * M
        M * M isa BlockMatrix
        "a bordered matrix: a 2×2 block, a column, a row and a scalar"
        K = BlockArray([A b; c' d], [2, 1], [2, 1])
        K[Block(2, 1)]
        det(K)
    end
end

#%% md id=assume_md
@md raw"""
### Assumptions

What is known about a variable is declared with it: `@giac_var k::positive n::integer x::real y::(0, 1)`, and `@giac_matrix M[2, 2]::real` for every entry. Giac then simplifies with that knowledge: $\sqrt{k^2} = k$, $\sin n\pi = 0$, and integrals that converge only for $k > 0$ evaluate. A declaration replaces what was assumed about the name before, so a plain `@giac_var k` forgets it. `assuming(k < 1) do … end` adds a condition for one block and restores the declaration afterwards.
"""

#%% code id=assume_demo
let
    @giac_var k::positive n::integer x::real y::(0, 1) t ω::positive
    demo = @examples begin
        "declared with the variable"
        sqrt(k^2), sign(k)
        sin(n * π), cos(n * π)
        conj(x), abs(x)^2
        sqrt(y^2), sqrt((y - 1)^2)
        "integrals that converge only for a sign"
        C.integrate(exp(-k * t), t, 0, Inf)
        C.integrate(sin(ω * t) / t, t, 0, Inf)
        "assuming adds a condition for one block"
        assuming(k < 1) do
            sqrt((k - 1)^2), C.integrate(exp(-k * t), t, 0, Inf)
        end
        sqrt((k - 1)^2)
        "a plain declaration forgets"
        @giac_var k
        sqrt(k^2), C.integrate(exp(-k * t), t, 0, Inf)
    end
    @giac_var n x y ω     # assumptions are global per name: drop them for the other cells
    demo
end

#%% md id=equations_md
@md"""
### Equations as values

An equation `a ~ b` is a value to work with, as on paper. Arithmetic applies to both sides, `lhs` and `rhs` take it apart, `isolate(eq, x)` solves it for `x` as an equation `x ~ …`, and an equation can be the rule in `substitute`. Arithmetic does not simplify (no Giac arithmetic does); `simplify` an equation to tidy both sides.
"""

#%% code id=equations_demo
let
    @giac_var x y m g L θ v
    @examples begin
        "both sides at once"
        eq = 2x + 3 ~ y
        eq - 3
        C.simplify((eq - 3) / 2)
        eq^2
        eq + (x ~ 1)
        lhs(eq), rhs(eq)
        "solving for a variable"
        isolate(eq, x)
        isolate(x^2 ~ y, x)
        "an equation as a substitution rule"
        substitute(x * y, isolate(eq, y))
        "a pendulum released from horizontal: its speed at angle θ from the bottom"
        energy = m * v^2 / 2 + m * g * L * (1 - cos(θ)) ~ m * g * L
        isolate(energy, v)
    end
end

#%% md id=functions_md
@md"""
### Symbolic functions

`@giac_function f(x) = x^2 * sin(x)` defines a function of a symbolic argument, local to the definition. Called on an expression it substitutes, on exact numbers it stays exact, and on floats it runs a compiled version, so it plots as fast as a hand-written function. `f'` is its derivative as a function. A vector argument `x[1:n]` takes a vector, and `∇`, `jacobian` and `hessian` give functions of the same arguments. A body may use outer variables such as a matrix `P`; it stays symbolic in them.
"""

#%% code id=functions_demo
let
    @giac_var y
    @giac_matrix P[2, 2] symmetric
    @examples begin
        "a function of one variable"
        @giac_function f(x) = x^2 * sin(x)
        f(y + 1)
        f(π / 2), f(π)
        f(0.5)
        f', f''
        "the pendulum as a function of state and input"
        @giac_function F(q[1:2], u) = [q2, -sin(q1) + u]
        F([0.1, 0.0], 0.5)
        jacobian(F)
        jacobian(F)([0, 0], 0)
        "a quadratic form"
        @giac_function V(z[1:2]) = z' * P * z
        ∇(V)
        hessian(V)
    end
end

#%% code id=functions_plot
using CairoMakie
# f, f′ and f″ called on floats run compiled code
let
    @giac_function f(x) = x^2 * sin(x)
    xs = range(0, 2π, 400)
    set_theme!(theme_dark())
    fig = Figure(size = (720, 340))
    ax = Axis(fig[1, 1]; xlabel = L"x", title = L"f(x) = x^2 \sin x", titlesize = 18)
    lines!(ax, xs, f.(xs); label = L"f")
    lines!(ax, xs, f'.(xs); label = L"f'")
    lines!(ax, xs, f''.(xs); label = L"f''")
    axislegend(ax; position = :lb)
    fig
end

#%% md id=taylor_md
@md"""
### Taylor polynomials

`taylor(f, x => a, n)` is the Taylor polynomial of `f` about `x = a`, with every term up to total degree `n`. `x` can be a vector of variables, with the point as a tuple to keep `π` exact, and an array `f` expands entry by entry. With `n = 1` that is the linearization of a vector field about an equilibrium.
"""

#%% code id=taylor_demo
let
    @giac_var x[1:2] u t
    f = [x2, -sin(x1) + u * cos(x1)]
    @examples begin
        "one variable"
        taylor(sin(t), t, 7)
        taylor(exp(t), t => 1, 2)
        "several variables: total degree"
        taylor(exp(x1 * x2) + log(1 + x1), x, 4)
        "the pendulum, linearized about hanging and upright"
        taylor(f, [x; u], 1)
        taylor(f, [x; u] => (π, 0, 0), 1)
        "and to third order about hanging"
        taylor(f, [x; u], 3)
    end
end

#%% md id=collect_md
@md raw"""
### Collecting by monomials

`collect_terms(e, x)` writes `e` as a sum over monomials in the variables `x`, each with its coefficient collected: higher degrees first, then higher powers of the earlier variables. A factor that depends on `x` without being a power of one, like `sin(x1)`, counts as part of the monomial. `Giac.coefficients(e, x)` gives the same pieces as a `Dict` from monomial to coefficient. It is not exported, because Nemo exports a `coefficients` too. For a Lyapunov candidate this reads off $\dot V$ as a quadratic form.
"""

#%% code id=collect_demo
let
    @giac_var x[1:2] a b
    @giac_matrix A[2, 2] P[2, 2] symmetric
    @examples begin
        "a polynomial in x with parameters"
        collect_terms((x1 + x2 + a)^2, x)
        collect_terms(a * sin(x1) * x2 + b * sin(x1) * x2 + x2^2, x)
        "V = xᵀPx along ẋ = Ax"
        Vdot = 2 * (x' * P * (A * x))
        collect_terms(Vdot, x)
        c = Giac.coefficients(Vdot, x)
        c[x1 * x2]
    end
end

#%% md id=typeset_md
@md"""
### Written as a textbook would

Giac stores an expression in its own order. How it is written is decided separately, by conventions:

- **Sums** run in descending powers of the variable, constant last, or ascending for a series. A difference of two terms leads with its positive one: 1 − x².
- **Products** run numbers, constants, parameters, variables, exponentials, other functions, sums. Parameters go Latin before Greek, each in the order declared, so `@giac_var m g h` writes mgh.
- **Fractions:** negative powers become fractions, a sign is pulled out of a fraction, and fractions in exponents are written inline (e^{−t/τ}).
- **Which symbols are variables** comes from `by`, from the expression (different powers across a sum, a square, the argument of y(t)), or from the letters textbooks use for variables (x, y, z, t, s, u, q, θ, φ).

The defaults can be changed for everything (`typeset!(…)`), for a block (`typeset(…) do … end`) or for one value (`typeset(e; by = s)`), including a derivation's rows (`@derivation typeset(fractions = :front) begin … end`).
"""

#%% code id=typeset_demo
let
    @giac_var m g h v x a b c t τ ζ ω s K
    G = K * ω^2 / (s^2 + 2ζ * ω * s + ω^2)
    @examples begin
        "the defaults"
        c + b * x + a * x^2
        -x^2 + 1, -a + b
        h * g * m, x * a
        a * exp(-t / τ) * cos(ω * t)
        sin(x)^2 + 3cos(x)^2
        m * v^2 / 2
        G
        "changed for one value"
        typeset(G; by = s)
        typeset(m * v^2 / 2; fractions = :front)
        typeset(x^2 + 1 + x; order = :ascending)
        typeset(exp(x); exp = :exp)
    end
end

#%% md id=in_form_md
@md"""
### Written in a given form: `in_form`

`in_form(e, template, params)` writes `e` in the form of `template` by solving for its parameters. The template's other symbols are its variables. Coefficients in them are matched, and the resulting equations are solved. The result is a dictionary of `parameter => value` and shows as the template with the values. `substitute(template, f)` fills the values in.

When several choices fit (ω = ±2), the one with fewest negative values is kept, and the others are in `f.alternatives`. A sign declared for a parameter (`@giac_var ω::positive`) rules out choices that contradict it.
"""

#%% code id=in_form_demo
let
    @giac_var s K ζ ω x a h k A B b₀ a₀ a₁
    second_order = K * ω^2 / (s^2 + 2ζ * ω * s + ω^2)
    vertex = a * (x - h)^2 + k
    fractions = A / (s + 1) + B / (s + 2)
    @examples begin
        "a second-order system, numbers and symbols"
        in_form(5 / (s^2 + 3s + 4), second_order, (K, ζ, ω))
        only(in_form(5 / (s^2 + 3s + 4), second_order, (K, ζ, ω)).alternatives)
        in_form(b₀ / (s^2 + a₁ * s + a₀), second_order, (K, ζ, ω))
        "completing the square, partial fractions"
        in_form(2x^2 - 4x + 7, vertex, (a, h, k))
        substitute(vertex, in_form(2x^2 - 4x + 7, vertex, (a, h, k)))
        substitute(fractions, in_form((3s + 5) / (s^2 + 3s + 2), fractions, (A, B)))
    end
end

#%% md id=names_md
@md"""
### Names and derivatives, written as typed

A name is written the way it is typed. Subscripts can be typed as `x_1`, `x₁` or `x1`; words in a subscript are upright (`x_ref`), and subscript letters work too (`kₚ`, `ωₙ`). Accents (`x̂`, `V̇`, `ẍ`), superscripts (`Kᵀ`, `x⁺`, `P★`) and primes come out as marks, and `Δx` is Δx. Anything else can be given its LaTeX with the name: `@giac_var m_c => "m_{\text{cart}}"`.

After a function of time, its name with dots declares its derivatives: `@giac_var q(t) q̇ q̈` makes `q̇` the derivative of `q`. Derivatives in time are written with dots, partial derivatives as ∂f/∂x, and a function of time alone without its argument. The styles `derivatives = :prime | :leibniz` and `arguments = :shown` change that.
"""

#%% code id=names_demo
let
    @giac_var t q(t) q̇ q̈ u x y x_ref kₚ e₁ b_0 ζ ωₙ Δx Kᵀ x̂₁ m_c => "m_{\\text{cart}}"
    f = giac_eval("f(x,y)")
    @examples begin
        "names"
        kₚ * e₁ + x_ref, b_0 * ωₙ
        Δx^2, Kᵀ * x̂₁, m_c
        "derivatives"
        q̈ + 2ζ * ωₙ * q̇ + ωₙ^2 * q ~ ωₙ^2 * u
        q̇^2 + sin(q), q(0)
        diff(diff(f, x), y)
        "other styles"
        typeset(q̈ + q; derivatives = :prime)
        typeset(q̇^2; derivatives = :leibniz)
        typeset(q̇ + q; arguments = :shown)
    end
end

#%% md id=matsym_md
@md"""
## 7. Symbolic matrices

`@matsym A[n, n] P[n, n]::posdef Q[n, n]::orthogonal x[n]` declares matrices as single symbols, of a size given by numbers or names, whose products do not commute. An expression is kept as a sum of scalar coefficients times products of factors, and the identities of matrix algebra hold as it is built:

- (AB)ᵀ = BᵀAᵀ, (AB)⁻¹ = B⁻¹A⁻¹, AA⁻¹ = I;
- Pᵀ = P for a symmetric P, Wᵀ = −W for a skew W, Q⁻¹ = Qᵀ for an orthogonal Q;
- the inverse of a sum cancels against the sum;
- a 1×1 term is its own transpose, and a 1×1 factor that does not fit as a matrix is a scalar: (xᵀx) b.

`tr` and `det` give scalars, with `tr(AB) = tr(BA)` and `det(cAB) = cⁿ det A det B`. What's known about them, such as det P > 0 and (det Q)² = 1, goes to the same facts the `Rules` sign tests and relation reductions read.

`differential`, `∇`, `jacobian`, `hessian` and `derivative` do matrix calculus (below), and `quadratic_form` takes a quadratic form apart. `substitute(M, A => entries)` gives the explicit matrix. `[A B; C D]` is a block matrix with products, `det` and `inv` (by the Schur complement). `simplest`, `rewrite` and `equivalent` work on the scalar coefficients.
"""

#%% code id=matsym_demo
let
    @matsym A[n, n] B[n, n] C[n, n] D[n, n] Q[n, n]::orthogonal W[n, n]::skew
    @giac_var t
    @examples begin
        "the algebra, as expressions are built"
        (A * B)', inv(A * B)
        inv(A + B) * (A + B) * C
        Q' * Q, inv(Q), W'
        (A + B)^2
        "scalars"
        tr(A * B * C), tr(C' * B' * A')
        tr(A * B * inv(A)), tr(W)
        det(2 * A * B), det(inv(A))
        "coefficients simplify"
        simplest((sin(t)^2 + cos(t)^2) * A + 2sin(t) * cos(t) * B)
        "blocks"
        X = [A B; C D]
        inv([A B; 0 D])
        inv(X) * X
        det(X)
    end
end

#%% code id=matsym_identities
# identities beyond the canonical form, each checked on random numeric matrices
let
    @matsym A[n, n] B[n, n] u[n] v[n] M[3, 2] N[2, 3]
    ids = matrix_identities()
    by(name) = only(filter(i -> i.name == name, ids))
    demo = @examples begin
        "the catalog"
        ids
        all(verify, ids)
        "used where they help"
        simplest(inv(inv(A) + inv(B)))
        rewrite(inv(I + A * B) * A, ids...)
        by("Sherman–Morrison")(inv(A + u * v'))
        by("determinant lemma")(det(A + u * v'))
        by("Sylvester")(det(I + M * N))
    end
    demo
end

#%% code id=matsym_lyapunov
# the Lyapunov argument for ẋ = Ax at the level of matrices, then in the entries for n = 2
let
    @matsym A[n, n] P[n, n]::posdef x[n]
    A₂ = Giac.giac_array(:a, 2, 2)                          # entries a11 … a22
    P₂ = Giac.giac_array(:p, 2, 2; structure = :symmetric)
    @derivation begin
        V = x' * P * x
        g = ∇(V, x)
        V̇ = g' * (A * x)
        S = quadratic_form(V̇, x)
        "for n = 2"
        S₂ = substitute(S, A => A₂, P => P₂)
    end
end

#%% md id=matcalc_md
@md"""
### Matrix calculus

`differential(M)` applies the product rule to a matrix expression, in the differentials `dX` of its symbols: d(XY) = dX Y + X dY, d(X⁻¹) = −X⁻¹ dX X⁻¹. Traces and determinants follow: d tr W = tr dW and d det M = det M tr(M⁻¹ dM). Given symbols, `differential(M, X)` lets only those vary.

The rest is read off the differential:
- `∇(f, X)` is the gradient of a scalar in any matrix or vector symbol, the G with df = tr(Gᵀ dX);
- `jacobian(F, x)` is J with dF = J dx, and `hessian(f, x)` is the Jacobian of the gradient;
- `derivative(f, x => ẋ)` is the derivative along a direction: each dx replaced by ẋ. Along the trajectories of ẋ = Ax, `derivative(x'*P*x, x => A*x)` is the V̇ of a Lyapunov argument.
"""

#%% code id=matcalc_demo
let
    @matsym X[n, n] A[n, n] W[n, n] P[n, n]::posdef x[n] b[n]
    @examples begin
        "differentials"
        differential(X * A * X, X)
        differential(inv(X))
        differential(det(X))
        "gradients in matrices and vectors"
        ∇(tr(X' * A * X), X)
        ∇(log(det(X)), X)
        ∇(x' * W * x + b' * x, x)
        "Jacobians and Hessians"
        jacobian(inv(X) * x, x), jacobian((x' * x) * b, x)
        hessian(x' * W * x, x)
        "along ẋ = Ax"
        derivative(x' * P * x, x => A * x)
        quadratic_form(derivative(x' * P * x, x => A * x), x)
    end
end

#%% md id=kron_md
@md raw"""
### Kronecker products and vectorization

`kron(A, B)` is the Kronecker product A ⊗ B, of size given by products of names (`m*n`, which `@matsym` also accepts). It follows the rules as expressions are built: (A⊗B)(C⊗D) = AC⊗BD, (A⊗B)ᵀ = Aᵀ⊗Bᵀ, (A⊗B)⁻¹ = A⁻¹⊗B⁻¹, tr(A⊗B) = tr A tr B, and det(A⊗B) = (det A)ᵐ (det B)ⁿ.

`vectorize(E, X)` writes an expression linear in `X` as the linear system its entries make, returned as `(K = …, R = …)`: vec E = K vec X + vec R, where R is the part of E without X. It uses vec(UXV) = (Vᵀ ⊗ U) vec X and, for a transposed unknown, vec(UXᵀV) = (Vᵀ ⊗ U) K₍ₘ,ₙ₎ vec X, with the commutation matrix K₍ₘ,ₙ₎. The Lyapunov equation AᵀP + PA = −Q becomes (Aᵀ⊗I + I⊗Aᵀ) vec P = −vec Q.

For a symmetric unknown, `unknowns = :vech` counts each entry once: vech E = Dₙ⁺ K Dₙ vech P, a system of n(n+1)/2 equations, with the duplication matrix Dₙ (vec P = Dₙ vech P) and its left inverse Dₙ⁺. The constants have their algebra too: K₍ₘ,ₙ₎K₍ₙ,ₘ₎ = I, Dₙ⁺Dₙ = I, K₍ₙ,ₙ₎Dₙ = Dₙ, and K moves past a Kronecker product, K(A⊗B) = (B⊗A)K. `substitute` gives everything explicitly, and `\` solves it.
"""

#%% code id=kron_demo
let
    @matsym A[n, n] B[m, m] C[n, n] D[m, m] P[n, n]::symmetric X[n, m] E[n, n] F[m, m] Z[n, n] G[m, n] H[p, q]
    K(a, b) = Giac.commutation(a, b)
    A₂ = GiacExpr[0 1; -2 -3]
    L₂ = substitute(vectorize(A' * P + P * A, P; unknowns = :vech).K, A => A₂)
    @examples begin
        "the algebra"
        kron(A, B) * kron(C, D), inv(kron(A, B))
        tr(kron(A, B)), det(kron(A, B))
        "linear matrix equations as linear systems: vec E = K vec X + vec R"
        vectorize(A' * P + P * A, P).K
        vectorize(E * X + X * F, X).K
        vectorize(A * Z' * C, Z).K
        vectorize(A * P + C, P).R
        "commutation and duplication matrices"
        K(:p, :m) * kron(G, H), K(:p, :m) * kron(G, H) * K(:n, :q)
        vectorize(A' * P + P * A, P; unknowns = :vech).K
        "the Lyapunov equation for A = [0 1; -2 -3], Q = I, in vech P"
        L₂
        L₂ \ (-GiacExpr[1, 0, 1])
    end
end

#%% md id=held_md
@md"""
## 8. Calculus as it is written

### Held commands

`hold_cmd(cmd, args...)` is a command left unevaluated, written in its notation: ∫, d/dx, lim, Σ, ℒ, 𝒵, and `name(args)` for the rest. It is an expression like any other: it goes into sums and products, stands on either side of `~`, and `substitute` works inside it. `release` evaluates every held command in an expression, so `h ~ release(h)` puts a computation next to its result.
"""

#%% code id=held_demo
let
    @giac_var x t s n z a ω
    I = hold_cmd(:integrate, x^2 * exp(-x), x, 0, Inf)
    ℒ = hold_cmd(:laplace, exp(-a * t) * sin(ω * t), t, s)
    Σ = hold_cmd(:sum, 1 / n^2, n, 1, Inf)
    @examples begin
        "written, not evaluated"
        hold_cmd(:diff, sin(ω * t), t), hold_cmd(:diff, x^3, x, 2)
        hold_cmd(:limit, sin(x) / x, x, 0), hold_cmd(:product, n, n, 1, 5)
        hold_cmd(:ztrans, a^n, n, z), hold_cmd(:factor, x^2 - 1)
        "next to their values"
        I ~ release(I)
        ℒ ~ release(ℒ)
        Σ ~ release(Σ)
        "in expressions"
        2 * hold_cmd(:limit, sin(x) / x, x, 0) + 1
        substitute(hold_cmd(:sum, a * n, n, 1, 3), a => 2)
        release(substitute(hold_cmd(:sum, a * n, n, 1, 3), a => 2))
    end
end

#%% md id=coords_md
@md"""
### Coordinates and velocities

With `@giac_var q(t) q̇`, `diff` treats a coordinate and its velocity as independent variables, as mechanics does: ∂L/∂q̇ differentiates in q̇ alone, and `diff(·, t)` applies the chain rule through q(t). Derivatives of unknown functions are written ∂F/∂q̇, or dM/dq for a function of one variable. `partial(L, t)` is the explicit dependence on time alone.

`differential(f, x, y, …)` is ∂f/∂x dx + ∂f/∂y dy + …, written in the order the variables are given; with no variables it takes all of them. A coefficient is read off with `diff(df, differential(x))`. The same function takes matrix symbols (section 7), and both kinds can be mixed.
"""

#%% code id=coords_demo
let
    @giac_var t m g ℓ k Ω q(t) q̇ q̈ θ(t) θ̇ M(x) U(x) p x y F(x, v, s) ω::positive q₀ v₀
    @matsym X[n, n] A[n, n]
    Lp = m * ℓ^2 * θ̇^2 / 2 + m * g * ℓ * cos(θ)
    Ld = m * q̇^2 / 2 - k * q^2 / 2 + g * q * cos(Ω * t)
    demo = @examples begin
        "a pendulum: ∂L/∂θ̇, ∂L/∂θ, and Lagrange's equation"
        diff(Lp, θ̇), diff(Lp, θ)
        diff(diff(Lp, θ̇), t) - diff(Lp, θ) ~ 0
        "unknown functions"
        diff(M(q) * q̇^2 / 2 - U(q), q)
        diff(diff(F(q, q̇, t), q̇), t)
        "explicit dependence on time, and the total derivative"
        partial(Ld, t)
        diff(Ld, t)
        "differentials"
        differential(p * y - F(x, y, t), x, y, p, t)
        differential(q^2 * q̇)
        differential(x^2 + tr(A * X), x, X)
        "initial velocities"
        C.desolve([q̈ + ω^2 * q ~ 0, q(0) ~ q₀, q̇(0) ~ v₀], q)
    end
    @giac_var ω     # assumptions are global per name
    demo
end

#%% md id=held_forms_md
@md"""
### Held integrals, evaluations, and integration by parts

`integral(f, t, a, b)` is ∫ f dt, held, and `evaluated(f, t, a, b)` is f|ₐᵇ (`evaluated(f, x => a)` is f at a point). `substitute`, `factor`, `expand` and `normal` work inside them, and a substitution that fixes an evaluation's value replaces the bar with that value. `by_parts(I, p)` integrates by parts with `p` as the factor that is differentiated, and leaves the other terms under the integral.

Together they carry the calculus of variations as it is written on paper: vary the path q → q + εη, differentiate at ε = 0, integrate by parts, and the endpoint term goes away when η(t₁) = η(t₂) = 0. `integrate` also gives the antiderivative of an exact derivative of unknown functions, which GIAC alone cannot.
"""

#%% code id=held_forms_demo
let
    @giac_var t t₁ t₂ ε q(t) q̇ η(t) p(t) F(x, v, s)
    δL = substitute(diff(substitute(F(q, q̇, t), q => q + ε * η), ε), ε => 0)
    δS = integral(substitute(δL, diff(F(q, q̇, t), q̇) => p), t, t₁, t₂)
    @examples begin
        "held"
        integral(F(q, q̇, t), t, t₁, t₂)
        evaluated(p * η, t, t₁, t₂), evaluated(cos(ε * t), ε => 0)
        "the first variation, with p = ∂F/∂q̇"
        δL
        δS
        by_parts(δS, p)
        C.factor(substitute(by_parts(δS, p), η(t₁) => 0, η(t₂) => 0))
        "exact derivatives integrate"
        C.integrate(diff(η * p, t), t)
        C.integrate(diff(η^2 * p, t) + t^2, t)
    end
end

#%% md id=physics_md
@md"""
### Physics notation

`im` is the imaginary unit i, written as a constant: iħ ∂ψ/∂t, e^{i(kx − ωt)}. A derivative written as a fraction stands after its coefficient's fraction (−ħ²/2m ∂²ψ/∂x²), and a second or higher derivative leads its sum. `substitute(e, ψ => φ * T)` for a function of several variables also replaces its derivatives, which is separation of variables.

A root of a product takes out the factors known not to be negative, from declarations like `m::positive`, and gathers the rest under one radical: √(ħ²k²/2m) = ħ|k|/√(2m).
"""

#%% code id=physics_demo
let
    @giac_var x t k ω ħ::positive m::positive ψ(x, t) V(x) φ(x) T(t)
    SE = im * ħ * diff(ψ, t) ~ -ħ^2 / (2m) * diff(ψ, x, 2) + V * ψ
    demo = @examples begin
        "a plane wave"
        exp(im * (k * x - ω * t))
        im * ħ * diff(exp(im * (k * x - ω * t)), t)
        "the Schrödinger equation, and ψ = φ(x) T(t)"
        SE
        C.expand(substitute(SE, ψ => φ * T) / (φ * T))
        "roots"
        sqrt(ħ^2 * k^2 / (2m)), sqrt(m^2 * ω^2), sqrt(8m^3)
    end
    @giac_var ħ m     # assumptions are global per name
    demo
end

#%% md id=deriv_rows_md
@md"""
### Derivations: `:=`, `=>` and functions

The rows of `@derivation` (section 4) say what they bind and what they apply:

- `name = expr` shows `name = value` and binds `name`; `φ(x) = expr` binds a function, so `φ(0)` and `φ(a)` work in later rows.
- `name := expr` shows the expression *as written* = its value, and binds `name` to the value. `name := lhs ~ rhs` shows the equation and binds it, as a rule for `substitute`.
- `lhs ~ expr` shows `lhs` as written = its value, and binds nothing.
- `cond => result` shows cond ⇒ result, and substitutes the result into every later row and every name bound so far. The result can be an equation, or a name to solve the condition for. Conditions follow after commas (`n::integer, n != 0`): they are written, and assumed from that row on. `name := cond => result` binds the result as a rule without applying it.

The particle in a box: φ(0) = 0 fixes A, φ(a) = 0 quantizes k, the energy follows, and normalization gives B. Then a mass on a spring: Lagrange's expression as written, and the frequency applied as a rule.
"""

#%% code id=deriv_rows_box
let
    @giac_var x k a::positive A B::positive E ħ::positive m::positive n
    d = @derivation begin
        φ(x) = A * cos(k * x) + B * sin(k * x)
        φ(0) ~ 0 => A
        φ(a) ~ 0 => k ~ n * π / a, n::integer, n != 0
        E = ħ^2 * k^2 / (2m)
        C.integrate(φ(x)^2, x, 0, a) ~ 1 => B
        φ(x) ~ φ(x)
    end
    @giac_var ħ m n a B     # assumptions are global per name
    d
end

#%% code id=deriv_rows_mech
let
    @giac_var t m k ω q(t) q̇ q̈
    @derivation begin
        L = m * q̇^2 / 2 - k * q^2 / 2
        EL := diff(diff(L, q̇), t) - diff(L, q)
        r := k ~ m * ω^2
        EL ~ substitute(EL, r)
        q̈ ~ rhs(isolate(substitute(EL, r) ~ 0, q̈))
    end
end

#%% md id=exact_md
@md"""
## 9. Exact answers: signs, proofs, polynomial systems

Giac writes and rewrites expressions. Some questions about them are better answered by a specialist engine and then checked exactly: the sign of an expression, a proof that a polynomial is nonnegative, every real solution of a polynomial system. Loading the engine turns on the matching Giac extension:

- **Nemo** (FLINT): exact polynomial algebra; `positivity`, `decompose`, and the sign decisions below.
- **JuMP + HiGHS**: linear programs for `certify`.
- **JuMP + Clarabel**: semidefinite programs for sums of squares, Lyapunov functions and regions of attraction.
- **AlgebraicSolving** (msolve): Gröbner bases, real solutions, parametrizations, elimination, classification by parameter.

A floating-point solver only proposes. Every certificate below has been rounded to rationals and checked exactly. The packages are loaded with `import`, because Nemo exports names (`coefficients`, `degree`) that this notebook uses from Giac.
"""

#%% code id=exact_setup
import Nemo, JuMP, Clarabel, HiGHS     # load Giac's extensions; their own names stay out of the way
import AlgebraicSolving
using AlgebraicSolving: real_solutions, rational_solutions, rational_parametrization, eliminate,
                        groebner_basis, real_root_classification
using Giac.Control
[ext => Base.get_extension(Giac, ext) !== nothing for ext in
 (:GiacNemoExt, :GiacJuMPExt, :GiacClarabelExt, :GiacAlgebraicSolvingExt)]

#%% md id=signs_md
@md"""
### Signs

`sign_of(e)` decides the sign of an expression as far as it can: `:positive`, `:negative`, `:zero`, `:nonnegative`, `:nonpositive` or `:unknown`. It reads what is known of each variable from its declaration (`@giac_var k::positive`), from assumptions in force (`assuming`), and from `positive`/`negative` hints. It works through sums, products, powers and functions of known sign (`exp`, `abs`, `sqrt`, `ln`, odd functions). When that leaves it open, the polynomial skeleton, with each function call standing for a variable of the sign found for it, goes to `positivity`: coefficient signs, then a sum of squares. `simplify_signs(e)` uses it to resolve `abs`, `sign`, `max` and `min`.
"""

#%% code id=signs_demo
let
    @giac_var x y k::positive a::positive b::positive
    @examples begin
        "from the structure: functions of known sign, declared variables"
        sign_of(exp(x) + k), sign_of(log(1 + k)), sign_of(-k^3 - exp(y))
        "mixed coefficient signs: a sum of squares decides"
        sign_of(x^2 - 2x * y + y^2), sign_of(sin(x)^2 - 2sin(x) + 1)
        sign_of(x - y)
        "an assumption in force"
        assuming(x > 0) do
            sign_of(x + 1)
        end
        "simplifying by signs"
        simplify_signs(abs(a * b + 1)), simplify_signs(abs(-a - 1))
        simplify_signs(max(a^2 + b^2, 2a * b))
        simplify_signs(abs(x - y))
    end
end

#%% md id=positivity_md
@md"""
### Positivity under an ordering, and decompositions

`positivity(e; order, positive)` writes each variable of an ordering as the one before it plus a positive gap `Δ`, and reads the sign off the coefficients: a polynomial in positive variables with positive coefficients is positive. When the signs are mixed it reports the terms against the majority, and with Clarabel loaded it then looks for a sum of squares (each positive variable written as a square), which proves the sign nonnegative.

`decompose(target, blocks)` writes a target as `Σ Mₖ Bₖ` with unknown polynomial multipliers, by exact coefficient matching, and returns one solution with the free directions of the whole family. `certify` picks a member whose multipliers have the signs wanted, by a linear program whose weights are made rational and checked exactly.
"""

#%% code id=positivity_demo
let
    @giac_var u t₁ t₂ A B p r L x y
    @examples begin
        "positive when each variable is the one before plus a positive gap"
        positivity((t₂ - u) * (t₁ - u) * A; order = (u, t₁, t₂), positive = (A,))
        "mixed coefficient signs, decided by a sum of squares"
        positivity(A^2 - A * B + B^2; positive = (A, B))
        "without the sum of squares: the terms against the majority sign"
        positivity(A^2 - 3A * B + B^2; positive = (A, B), sos = false)
        "a target as a combination of blocks with unknown multipliers (another package here exports `decompose`)"
        Giac.decompose((p * (L - r) + (L - p) * r)^2 - (p - r)^2 * L^2, [:Q => p * (L - p) * r * (L - r)])
        "the member of a family whose multipliers have the signs wanted: every product of up to 3 blocks is a block"
        d = Giac.decompose(x^3 - x^2 * y - x * y^2 + y^3, [:x => x, :y => y, :s => (x - y)^2]; products = 3, degree = 0)
        Giac.certify(d; positive = (x, y), block_signs = (1, 1, 1))
    end
end

#%% md id=sos_md
@md"""
### Sums of squares

`sos_decomposition(p)` writes a polynomial as `Σ wₖ qₖ²` with positive rational weights, which proves `p ≥ 0` everywhere. Clarabel finds a Gram matrix well inside the semidefinite cone. It is rounded to rationals, moved by the least amount that matches the polynomial's coefficients exactly, and factored over the rationals; the identity is checked in Nemo. It returns `nothing` when none is found: the polynomial may take negative values, or, like Motzkin's, be nonnegative without being a sum of squares.
"""

#%% code id=sos_demo
let
    @giac_var x y
    @examples begin
        c = sos_decomposition(x^2 - 2x * y + 2y^2)
        "the certificate written out, and checked"
        GiacExpr(c), C.expand(GiacExpr(c) - c.polynomial)
        "a quartic with cross terms"
        sos_decomposition(2x^4 + 2x^3 * y - x^2 * y^2 + 5y^4)
        "no certificate: indefinite, and Motzkin's nonnegative polynomial"
        sos_decomposition(x^2 - y^2)
        sos_decomposition(x^4 * y^2 + x^2 * y^4 - 3x^2 * y^2 + 1)
    end
end

#%% md id=systems_md
@md"""
### Polynomial systems

With AlgebraicSolving loaded, its functions take systems of Giac expressions: polynomials (each `= 0`) or equations `a ~ b`, with rational coefficients, in the variables given or else all of them in the order declared.

- `real_solutions` isolates every real solution in intervals with rational ends guaranteed to contain it. `s[x]` is the midpoint, and `s.lower`, `s.upper` the ends.
- `rational_parametrization` gives every complex solution exactly: each variable as a rational function of a new `T` at the roots of one polynomial.
- `eliminate` gives the relations among the remaining variables, and `normal_form` reduces modulo the system.
- `real_root_classification` divides parameter space by the signs of a few polynomials, so that the number of real solutions is the same throughout each part.
"""

#%% code id=systems_demo
let
    @giac_var x y t
    sys = [x^2 + y^2 - 1, y - x^2]
    @examples begin
        "every real solution, isolated exactly"
        sols = real_solutions(sys)
        Float64(sols[1][y]), Float64(sols[1].upper[2] - sols[1].lower[2])
        "every complex solution, exactly"
        rational_parametrization(sys)
        rational_solutions([x^2 - 1, y - x])
        "a parametrized curve's implicit equation"
        eliminate([x - (1 - t^2), y - 2t], t)
        "reduced modulo the system"
        AlgebraicSolving.normal_form(x^4 + 2x^2 * y^2 + y^4, [x^2 + y^2 - 1])
        groebner_basis(sys)
    end
end

#%% code id=classify_demo
let
    @giac_var x a b r X Y Z
    @examples begin
        "the equilibria of ẋ = x³ − a x + b: three inside the cusp 4a³ = 27b², one outside"
        real_root_classification([x^3 - a * x + b], [x]; params = [a, b])
        "Lorenz, by ρ: the pitchfork at ρ = 1"
        real_root_classification([10 * (Y - X), X * (r - Z) - Y, X * Y - 8 // 3 * Z], [X, Y, Z]; params = [r])
    end
end

#%% md id=control_md
@md"""
## 10. Nonlinear control: `Giac.Control`

The submodule `Giac.Control` has the tools of a nonlinear control course for systems `ẋ = f(x) + g(x) u`, with vector fields as vectors of expressions:

- Lie derivatives and brackets, `ad`, the accessibility rank and involutivity;
- relative degree and input-output linearization;
- Hurwitz conditions, and the stability they give, decided under the signs declared;
- every real equilibrium with its linearization's stability, certain;
- with Clarabel: certified Lyapunov functions and regions of attraction;
- center manifolds, where the linearization does not decide.
"""

#%% code id=lie_demo
let
    @giac_var θ ω px py φ
    f = [ω, -sin(θ)]
    g = [[cos(φ), sin(φ), 0], [0, 0, 1]]          # the unicycle's two input fields
    @examples begin
        "Lie derivatives along the pendulum"
        lie_derivative(θ^2, f, [θ, ω]), lie_derivative(θ^2, f, [θ, ω]; order = 2)
        "the unicycle reaches three states with two inputs, through a bracket"
        lie_bracket(g[1], g[2], [px, py, φ])
        accessibility_rank([0, 0, 0], g, [px, py, φ]), involutive(g, [px, py, φ])
        "the pendulum's angle as output: y'' = v with u = α + β v"
        io_linearization(θ, f, [0, 1], [θ, ω])
    end
end

#%% code id=stability_demo
let
    @giac_var s k a q::positive x y z
    @examples begin
        "Hurwitz conditions: every one positive, every root in the left half-plane"
        hurwitz_conditions(s^3 + 2s^2 + k * s + a, s)
        hurwitz_stable(s^3 + 2s^2 + q * s + q, s), hurwitz_stable(s^2 + k * s + 1, s)
        hurwitz_stable(GiacExpr[0 1; -2 -3], s)
        "every real equilibrium, with the count of unstable eigenvalues certain"
        equilibria([y, x - x^3 - y], [x, y])
        equilibria([10 * (y - x), x * (28 - z) - y, x * y - 8 // 3 * z], [x, y, z])
        "a center: the linearization does not decide"
        equilibria([y, -x], [x, y])
    end
end

#%% code id=lyapunov_demo
let
    @giac_var x y x₁ x₂ c::positive
    @examples begin
        "a polynomial Lyapunov function, both conditions certified exactly"
        lyapunov([-x₁ - x₁ * x₂^2, -x₂ + x₁^2 * x₂], [x₁, x₂])
        lyapunov([-x₁^3 + x₂, -x₁ - x₂], [x₁, x₂]; degree = 4, strict = false)
        "a region of attraction: ẋ = −x + x³ is attracted for |x| < 1, V = x²/2 < 1/2"
        region_of_attraction([-x + x^3], [x])
        "reversed-time Van der Pol"
        region_of_attraction([-x₂, x₁ + (x₁^2 - 1) * x₂], [x₁, x₂])
        "where the linearization does not decide: the dynamics on the center manifold"
        center_manifold([-x * y, -y + x^2], [x, y])
        center_manifold([-x + y - x^3, x - y], [x, y])
        "a weak focus, eigenvalues ±iω: the first Lyapunov coefficient (sin is fine)"
        lyapunov_coefficient([sin(x₂) - c * x₁^3, x₁], [x₁, x₂]; at = (x₁ => 0, x₂ => π))
    end
end

#%% md id=timeout_md
@md"""
## 11. Stopping a long computation

GIAC checks an interruption flag in its long loops (Gröbner bases, factoring, integration). `Giac.with_timeout(f, seconds)` stops a GIAC computation in `f` that runs longer, with a `GiacTimeout`, and `Giac.interrupt()` stops the one running now. A cancelled Kaimon job stops its GIAC work too. Both need Julia with more than one thread: something has to watch beside the computation. GIAC is usable again straight after.
"""

#%% code id=timeout_demo
let
    slow() = giac_eval("gbasis([x^7+y^6+z^5+w^3-1, x^5+y^7+z^6+w^4-1, x^6+y^5+z^7+w^5-1, x+y+z+w^2-2], [x,y,z,w], plex)")
    @examples begin
        Threads.nthreads()
        "a Gröbner basis that runs for minutes, stopped after 2 s"
        @elapsed(try
            Giac.with_timeout(slow, 2)
        catch e
            e
        end)
        try
            Giac.with_timeout(slow, 1)
        catch e
            e
        end
        "GIAC works again straight after"
        C.factor(giac_eval("x^4 - 1"))
    end
end

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   AlgebraicSolving 0.13.0 66b61cbe-0446-4d5d-9090-1ff510639f9d
#   BlockArrays 1.10.3 8e7c35d0-a365-5155-bbbb-fb81a777f24e
#   CairoMakie 0.15.16 13f3f980-e62b-5c42-98c6-ff1f3baf88f0
#   Clarabel 0.11.1 61c947e1-3e6d-4ee4-985a-eec8c727bd6e
#   Giac 0.14.3 e4421f97-9838-4fd0-9fa5-94f11373bf78 path=~/devel/Giac.jl
#   GiacSlate 0.1.2 f4c9f34b-fa6a-4506-9309-e8c0bac809fa path=~/devel/GiacSlate.jl
#   HiGHS 1.26.1 87dc4568-4c63-4d18-b0c0-bb2238e4078b
#   JuMP 1.32.1 4076af6c-e467-56ae-b986-b466b2749572
#   Nemo 0.57.1 2edaba10-b0f1-5616-af89-8c11ac63239a
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 51bc1aac-90d8-4a83-bd1e-0725fb1b9ea0
# ╚═╡
