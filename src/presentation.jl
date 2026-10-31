# Typesetting expressions as a textbook writes them. Giac stores an expression in an order of its
# own; how it is written is decided here, from conventions: a polynomial in descending powers of
# its variable (ascending for a series), a difference leading with its positive term, a product
# from numbers through parameters and variables to functions, negative powers as fractions.
# The conventions are defaults that can be changed for everything (`typeset!`), for a block of code
# (`typeset(…) do … end`) or for one expression (`typeset(e; by = x)`).

"""
    TypesetStyle

How expressions are written; see [`typeset!`](@ref) for the options.
"""
Base.@kwdef struct TypesetStyle
    by::Vector{String} = String[]
    order::Symbol = :auto
    lead::Symbol = :auto
    fractions::Symbol = :stacked
    exp::Symbol = :e
    roots::Symbol = :radical
    products::Symbol = :conventional
    trig_powers::Symbol = :superscript
    derivatives::Symbol = :dot
    arguments::Symbol = :auto
    time::String = "t"
    variable_letters::String = "xyztsuqθφ"
end

const _TYPESET_OPTIONS = Dict(
    :order => (:auto, :descending, :ascending, :as_is),
    :lead => (:auto, :positive, :as_is),
    :fractions => (:stacked, :front, :inline),
    :exp => (:e, :exp),
    :roots => (:radical, :power),
    :products => (:conventional, :as_is),
    :trig_powers => (:superscript, :parenthesized),
    :derivatives => (:dot, :prime, :leibniz),
    :arguments => (:auto, :shown))

# The default style, set by `typeset!`; a style for a block of code or one value is the running
# task's own (in its task-local storage), so concurrent tasks never see each other's.
const _TYPESET = Ref(TypesetStyle())
const _STYLE_KEY = :giac_typeset_style
_current_style() = get(task_local_storage(), _STYLE_KEY, _TYPESET[])::TypesetStyle
_in_style(f, style::TypesetStyle) = task_local_storage(f, _STYLE_KEY, style)

_names(v::Union{GiacExpr, Symbol, AbstractString}) = [v isa GiacExpr && iscall(v) && symb_funcname(v) == "of" ?
                                                       string(first(arguments(v))) : string(v)]
_names(vs::Union{Tuple, AbstractVector}) = reduce(vcat, map(_names, collect(vs)); init = String[])

function _with(style::TypesetStyle; kwargs...)
    fields = Dict{Symbol, Any}(f => getfield(style, f) for f in fieldnames(TypesetStyle))
    for (k, v) in kwargs
        if k === :by
            fields[:by] = _names(v)
        elseif k === :variable_letters
            fields[k] = string(v)
        elseif k === :time
            fields[k] = only(_names(v))
        elseif haskey(_TYPESET_OPTIONS, k)
            v in _TYPESET_OPTIONS[k] || throw(ArgumentError("typeset: $k is one of $(join(_TYPESET_OPTIONS[k], ", ")), got $v"))
            fields[k] = v
        else
            throw(ArgumentError("typeset: unknown option `$k`; the options are by, time, variable_letters, $(join(keys(_TYPESET_OPTIONS), ", "))"))
        end
    end
    return TypesetStyle(; fields...)
end

"""
    typeset!(; options...)
    typeset!(; reset = true)

Set how expressions are written from now on (the options given change; the rest stay):

- `by = x` or `by = (x, t)`: the variables, in order of priority. Sums are written in descending
  powers of them and products put them after the parameters. Without it, a symbol counts as a
  variable where the expression shows it as one (different powers in the terms of a sum, a square,
  the argument of a function `y(t)`), or where its letter is one used for variables.
- `variable_letters = "xyztsuqθφ"`: the letters that mark a variable by convention (`x₁`, `ẋ` and
  `θ₂` count by their first letter); `""` for none.
- `order = :auto | :descending | :ascending | :as_is`: the order of the terms of a sum. `:auto` is
  descending, and ascending for a series (a sum with a remainder `O(…)`).
- `lead = :auto | :positive | :as_is`: `:auto` writes a difference of two terms with the positive one
  first (`1 - x^2`); `:positive` leads every sum with a positive term.
- `fractions = :stacked | :front | :inline`: `:front` writes a numeric fraction in front (`½mv²`);
  `:inline` writes `a/b`. Fractions in exponents are always inline (`e^{-t/τ}`).
- `exp = :e | :exp`, `roots = :radical | :power`, `trig_powers = :superscript | :parenthesized`.
- `products = :conventional | :as_is`: a product as numbers, constants, parameters (Latin letters
  before Greek, each in the order they were declared, so `@giac_var m g h` writes `mgh`), variables,
  exponentials, other functions, sums.
- `derivatives = :dot | :prime | :leibniz`: a derivative in time of a function of time is written
  with dots (`q̇`, `q̈`, up to the third), with primes (`q′`), or as `dq/dt`. Derivatives in other
  variables, and partial derivatives, are written `dy/dx` and `∂f/∂x`. A sum puts higher
  derivatives first (`q̈ + 3q̇ + 2 sin q`).
- `time = t`: the variable that dots differentiate by.
- `arguments = :auto | :shown`: `:auto` writes a function of time alone without its argument
  (`q(t)` is `q`); its other values (`q(0)`) and other functions' arguments are always written.
"""
function typeset!(; reset::Bool = false, kwargs...)
    reset && (_TYPESET[] = TypesetStyle())
    _TYPESET[] = _with(_TYPESET[]; kwargs...)
    return _TYPESET[]
end

"""
    typeset(e; options...)
    typeset(; options...)
    typeset(f::Function; options...)

`e` to be written with the options of [`typeset!`](@ref) changed for it alone (an expression, an
array, a symbolic matrix). Without an expression, a function that does that, for a derivation:
`@derivation typeset(by = s) begin … end`. With a function first (`typeset(by = x) do … end`), the
options hold while it runs.
"""
typeset(e; kwargs...) = Typeset(e, _with(_current_style(); kwargs...))
typeset(; kwargs...) = (s = _with(_current_style(); kwargs...); e -> Typeset(e, s))
typeset(f::Function; kwargs...) = _in_style(f, _with(_current_style(); kwargs...))

"A value with the style it is written in; see [`typeset`](@ref)."
struct Typeset
    value::Any
    style::TypesetStyle
end
giac_latex(t::Typeset) = _in_style(() -> giac_latex(t.value), t.style)
_entry_latex(t::Typeset) = _in_style(() -> _entry_latex(t.value), t.style)
Base.show(io::IO, ::MIME"text/latex", t::Typeset) = print(io, "\$\$", giac_latex(t), "\$\$")
Base.show(io::IO, t::Typeset) = show(io, t.value)
Base.string(t::Typeset) = string(t.value)

# ---------------------------------------------------------------------------------------------
# An expression as a tree to write
# ---------------------------------------------------------------------------------------------

abstract type _PNode end
struct _PNum <: _PNode; v::Real; end
struct _PAtom <: _PNode; tex::String; name::String; constant::Bool; end
struct _PSum <: _PNode; terms::Vector{_PNode}; end
struct _PProd <: _PNode; factors::Vector{_PNode}; end
struct _PPow <: _PNode; base::_PNode; exp::_PNode; end
struct _PCall <: _PNode; f::String; args::Vector{_PNode}; user::Bool; end
struct _PEq <: _PNode; lhs::_PNode; rhs::_PNode; end
struct _POrder <: _PNode; arg::_PNode; end
struct _PRaw <: _PNode; tex::String; end
# the derivative of the function `f` (of `args`) in the variables `wrt`, one per order
struct _PDeriv <: _PNode; f::String; args::Vector{_PNode}; wrt::Vector{String}; end
# the derivative of an expression
struct _PDerivOf <: _PNode; arg::_PNode; wrt::Vector{String}; end
# the derivative of the function `f` at arguments that are not names, in the arguments at the
# positions `idx` (from 0): f′(x + h), ∂₁f(x + h, y)
struct _PDerivAt <: _PNode; f::String; args::Vector{_PNode}; idx::Vector{Int}; end
# an integral of `f` in `var`, between `limits` if there are two
struct _PIntegral <: _PNode; f::_PNode; var::_PNode; limits::Vector{_PNode}; end
# `f` evaluated: at `lower` (x = a), or between `lower` and `upper`
struct _PBar <: _PNode; f::_PNode; lower::_PNode; upper::Union{Nothing, _PNode}; end

const _CALL_TEX = Dict(
    "sin" => "\\sin", "cos" => "\\cos", "tan" => "\\tan", "cot" => "\\cot", "sec" => "\\sec", "csc" => "\\csc",
    "sinh" => "\\sinh", "cosh" => "\\cosh", "tanh" => "\\tanh", "coth" => "\\coth",
    "asin" => "\\arcsin", "acos" => "\\arccos", "atan" => "\\arctan",
    "asinh" => "\\operatorname{arsinh}", "acosh" => "\\operatorname{arcosh}", "atanh" => "\\operatorname{artanh}",
    "ln" => "\\ln", "log10" => "\\log_{10}", "exp" => "\\exp", "sign" => "\\operatorname{sgn}",
    "re" => "\\operatorname{Re}", "im" => "\\operatorname{Im}", "arg" => "\\arg",
    "trace" => "\\operatorname{tr}", "det" => "\\det")
const _TRIG = ("sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth")

function _pnode(e::GiacExpr)
    # numbers, names, sums and products are read off the tree directly
    n = getfield(e, :node)
    if n !== nothing && (getfield(e, :native) || !(n isa GenNode || n isa CallNode))
        n isa CallNode || n isa GenNode || return _pnode(n)
    end
    if !iscall(e)
        t = giac_type(e)
        (t == INT || t == ZINT || t == FRAC || t == DOUBLE || t == REAL) && return _PNum(to_julia(e))
        if t == CPLX
            v = to_julia(e)
            v isa Complex && return _complex_pnode(v)
        end
        s = string(e)
        s == "pi" && return _PAtom("\\pi", s, true)
        s == "infinity" && return _PAtom("\\infty", s, true)
        if t == IDNT
            haskey(_ATOMS, s) && return _PAtom("{" * _ATOMS[s][4] * "}", s, false)
            return _PAtom(_tex_name(s), s, false)
        end
        return _PRaw(tidy_latex(_giac_tex(e)))
    end
    h = symb_funcname(e)
    a = arguments(e)
    h == "+" && return _PSum(_PNode[_pnode(x) for x in a])
    if h == "*"
        # a product read as the core has it, so its roots are gathered
        m = _node(e)
        m isa MulNode && return _pnode(m)
        return _PProd(_PNode[_pnode(x) for x in a])
    end
    if h == "-" || h == "inv"
        x = _pnode(only(a))
        # a number stored as -(6) or inv(6), as division by a number is, is that number
        x isa _PNum && return _PNum(h == "-" ? -x.v : x.v isa Integer ? 1 // x.v : inv(x.v))
        return h == "-" ? _PProd(_PNode[_PNum(-1), x]) : _PPow(x, _PNum(-1))
    end
    if h == "^"
        b, x = _pnode(a[1]), _pnode(a[2])
        # Giac writes x^-2 as (1/x)^2
        b isa _PPow && b.exp isa _PNum && b.exp.v == -1 && x isa _PNum && return _PPow(b.base, _PNum(-x.v))
        return _PPow(b, x)
    end
    h == "=" && return _PEq(_pnode(a[1]), _pnode(a[2]))
    h == "exp" && string(only(a)) == "1" && return _PAtom("e", "e", true)
    h == "order_size" && return _POrder(_pnode(only(a)))
    h == "abs" && return _PCall("abs", _PNode[_pnode(only(a))], false)
    h == "conj" && return _PCall("conj", _PNode[_pnode(only(a))], false)
    if h == "diff" && length(a) >= 2 && is_identifier(a[2])
        k = length(a) >= 3 && _is_plain_number(a[3]) ? Int(to_julia(a[3])) : 1
        f, wrt = a[1], fill(string(a[2]), k)
        # f(x) in x is f's derivative; f(x + h) in h is the derivative of an expression
        if iscall(f) && symb_funcname(f) == "of" && any(x -> isequal(x, a[2]), arguments(f)[2:end])
            fa = arguments(f)
            return _PDeriv(string(fa[1]), _PNode[_pnode(x) for x in fa[2:end]], wrt)
        end
        return _PDerivOf(_pnode(f), wrt)
    end
    if h == "of" && iscall(a[1]) && symb_funcname(a[1]) in ("diff", "function_diff")
        # partial derivatives of f at (x, y): of(diff(diff(f, 0), 1), x, y), indices from 0; the
        # derivative of a function of one argument: of(diff(f), x), evaluated of(function_diff(f), x)
        g, idx = a[1], Int[]
        while iscall(g) && symb_funcname(g) in ("diff", "function_diff")
            ga = arguments(g)
            prepend!(idx, length(ga) == 1 ? [0] : [Int(to_julia(i)) for i in ga[2:end]])
            g = ga[1]
        end
        args = a[2:end]
        if is_identifier(g) && all(i -> 0 <= i < length(args), idx)
            names = [_wrt_name(args[i + 1]) for i in idx]
            all(!isnothing, names) && return _PDeriv(string(g), _PNode[_pnode(x) for x in args], names)
            return _PDerivAt(string(g), _PNode[_pnode(x) for x in args], idx)
        end
    end
    h == "of" && return _PCall(string(a[1]), _PNode[_pnode(x) for x in a[2:end]], true)
    h == "integrate" && length(a) in (2, 4) && return _PIntegral(_pnode(a[1]), _pnode(a[2]), _PNode[_pnode(x) for x in a[3:end]])
    h == "subst" && length(a) == 2 && iscall(a[2]) && symb_funcname(a[2]) == "=" && return _PBar(_pnode(a[1]), _pnode(a[2]), nothing)
    h == "preval" && length(a) in (3, 4) && return _PBar(_pnode(a[1]), _pnode(a[2]), _pnode(a[3]))
    _is_held_call(_node(e)) && return _PHeld(Symbol(h), Tuple(a))       # lim, Σ, ℒ, … (held_cmd.jl)
    haskey(_CALL_TEX, h) && return _PCall(h, _PNode[_pnode(x) for x in a], false)
    return _PRaw(tidy_latex(_giac_tex(e)))
end

# The name of a variable a partial derivative is in, for ∂f/∂x: a symbol's; for a function of
# variables `q(t)`, the function's; for a derivative of one, that name with dots (q̇, for a
# derivative in time) or primes. `nothing` for any other expression.
function _wrt_name(x::GiacExpr)
    is_identifier(x) && return string(x)
    iscall(x) || return nothing
    h, a = symb_funcname(x), arguments(x)
    h == "of" && all(is_identifier, a) && return string(a[1])
    if h == "diff" && length(a) in (2, 3) && is_identifier(a[2]) && iscall(a[1]) && symb_funcname(a[1]) == "of"
        length(a) == 3 && !_is_plain_number(a[3]) && return nothing
        k = length(a) == 3 ? Int(to_julia(a[3])) : 1
        base = _wrt_name(a[1])
        base === nothing && return nothing
        string(a[2]) == _current_style().time && k <= 3 && return base * ("̇", "̈", "⃛")[k]
        return base * "′"^k
    end
    return nothing
end

_pnode(n::NumNode) = n.v isa Real ? _PNum(n.v) : n.v isa Complex ? _complex_pnode(n.v) : _PRaw(tidy_latex(_giac_tex(GiacExpr(n))))
function _pnode(n::SymNode)
    s = n.name
    s == "pi" && return _PAtom("\\pi", s, true)
    s == "infinity" && return _PAtom("\\infty", s, true)
    haskey(_ATOMS, s) && return _PAtom("{" * _ATOMS[s][4] * "}", s, false)
    return _PAtom(_tex_name(s), s, false)
end
_pnode(n::AddNode) = _PSum(_PNode[_pnode(_scale(m, c)) for (m, c) in n.terms])

# A complex number with i as a constant: b i, or a + b i
const _I = _PAtom("i", "i", true)
function _complex_pnode(v::Complex)
    a, b = real(v), imag(v)
    bi = isone(b) ? _I : _PProd(_PNode[_PNum(b), _I])
    return iszero(a) ? bi : _PSum(_PNode[_PNum(a), bi])
end
function _pnode(n::MulNode)
    fs = _PNode[]
    if n.coef isa Complex
        c = _complex_pnode(n.coef)
        append!(fs, c isa _PProd ? c.factors : _PNode[c])
    else
        _isone_exact(n.coef) || push!(fs, _PNum(n.coef))
    end
    rest, roots = _gather_roots(n.factors)
    for (b, e) in rest
        push!(fs, isequal(e, _ONE) ? _pnode(b) : _PPow(_pnode(b), _pnode(e)))
    end
    append!(fs, roots)
    return length(fs) == 1 ? fs[1] : _PProd(fs)
end

# Roots of factors known not to be negative, written under one radical above the line and one
# below: √2 √m as √(2m). When the radicand above is a number, or both are monomials, as one root
# of the quotient: √2/√a as √(2/a). Roots with nothing to join are left as they are.
function _gather_roots(factors::Vector{Pair{Node, Node}})
    index(e) = e isa NumNode && e.v isa Rational && denominator(e.v) > 1 ? denominator(e.v) : 0
    count_of(k) = count(((b, e),) -> index(e) == k && _known_nonnegative(b), factors)
    rest, roots = Pair{Node, Node}[], _PNode[]
    groups = Dict{Int, Tuple{Vector{Node}, Vector{Node}}}()
    for (b, e) in factors
        k = index(e)
        if k > 0 && _known_nonnegative(b) && count_of(k) > 1
            # the whole power stays outside: m^(3/2) is m √m
            q, r = divrem(numerator(e.v), k)
            iszero(q) || push!(rest, b => NumNode(q))
            above, below = get!(() -> (Node[], Node[]), groups, k)
            push!(r > 0 ? above : below, abs(r) == 1 ? b : _pow(b, NumNode(abs(r))))
        else
            push!(rest, b => e)
        end
    end
    monomial(n) = n isa NumNode || n isa SymNode || (n isa MulNode && all(((b, e),) -> b isa NumNode || b isa SymNode, n.factors))
    for k in sort!(collect(keys(groups)))
        above, below = groups[k]
        A, B = _mul(above...), _mul(below...)
        if !isempty(above) && !isempty(below) && (A isa NumNode || monomial(A) && monomial(B))
            push!(roots, _PPow(_pnode(_mul(A, _pow(B, NumNode(-1)))), _PNum(1 // k)))
        else
            isempty(above) || push!(roots, _PPow(_pnode(A), _PNum(1 // k)))
            isempty(below) || push!(roots, _PPow(_pnode(B), _PNum(-1 // k)))
        end
    end
    return rest, roots
end
_pnode(n::Union{CallNode, GenNode}) = _pnode(GiacExpr(n))

# ---------------------------------------------------------------------------------------------
# Products: sign, numeric coefficient, and the factors above and below the line
# ---------------------------------------------------------------------------------------------

_flat_factors(n::_PNode) = n isa _PProd ? reduce(vcat, map(_flat_factors, n.factors); init = _PNode[]) : _PNode[n]

function _split(n::_PNode)
    coef = 1 // 1
    float = false
    num, den = _PNode[], _PNode[]
    order = nothing
    factors = _flat_factors(n)
    # a product with powers of numbers is a factorization (2^3·3^2·5): its numbers stay factors
    factorization = any(f -> f isa _PPow && f.base isa _PNum && f.exp isa _PNum && f.exp.v isa Integer && f.exp.v > 0, factors)
    for f in factors
        if f isa _PNum && factorization && f.v != -1
            push!(num, f)
        elseif f isa _PNum
            f.v isa AbstractFloat && (float = true)
            coef *= f.v isa Integer ? f.v // 1 : f.v
        elseif f isa _POrder
            order = f
        elseif f isa _PPow && f.exp isa _PNum && f.exp.v < 0
            k = -f.exp.v
            push!(den, isone(k) ? f.base : _PPow(f.base, _PNum(k)))
        else
            push!(num, f)
        end
    end
    sign = coef < 0 ? -1 : 1
    coef = abs(coef)
    return (; sign, coef, float, num, den, order)
end

# ---------------------------------------------------------------------------------------------
# Variables and orders
# ---------------------------------------------------------------------------------------------

# a factor as `(symbol name, power)` if it is a symbol to an integer power
function _symbol_power(f::_PNode)
    f isa _PAtom && !f.constant && return (f.name, 1)
    if f isa _PPow && f.base isa _PAtom && !f.base.constant && f.exp isa _PNum && f.exp.v isa Integer
        return (f.base.name, Int(f.exp.v))
    end
    return nothing
end

function _term_degrees(t::_PNode)
    d = Dict{String, Int}()
    for f in _split(t).num
        sp = _symbol_power(f)
        sp === nothing || (d[sp[1]] = get(d, sp[1], 0) + sp[2])
    end
    return d
end

# The variables of an expression, in priority order: those asked for, the arguments of function
# variables (t in y(t)), and symbols that show as variables (different powers across a sum's terms,
# or a power of 2 or more), the last in the order they were declared. With functions of a variable
# in it, the variables are theirs: a square (ω²) is then a parameter's.
function _variables(root::_PNode, style::TypesetStyle)
    found = Set{String}()          # from powers
    arguments = Set{String}()      # of function variables
    function walk(n)
        if n isa _PSum
            degrees = map(_term_degrees, n.terms)
            for s in union(keys.(degrees)...)
                ds = [get(d, s, 0) for d in degrees]
                (count(>(0), ds) >= 2 && length(unique(filter(>(0), ds))) > 1) && push!(found, s)
            end
            foreach(walk, n.terms)
        elseif n isa _PProd
            foreach(walk, n.factors)
        elseif n isa _PPow
            sp = _symbol_power(n)
            sp !== nothing && sp[2] >= 2 && push!(found, sp[1])
            walk(n.base); walk(n.exp)
        elseif n isa _PCall || n isa _PDeriv || n isa _PDerivAt
            (n isa _PCall ? n.user : true) && foreach(a -> a isa _PAtom && !a.constant && push!(arguments, a.name), n.args)
            foreach(walk, n.args)
        elseif n isa _PDerivOf
            walk(n.arg)
        elseif n isa _PEq
            walk(n.lhs); walk(n.rhs)
        elseif n isa _POrder
            walk(n.arg)
        elseif n isa _PIntegral
            n.var isa _PAtom && push!(arguments, n.var.name)       # the variable of integration
            walk(n.f)
        elseif n isa _PBar
            walk(n.f)
        end
    end
    walk(root)
    # variables named in the style are the variables; others are found only without them
    isempty(style.by) || return copy(style.by)
    isempty(arguments) || (found = arguments)
    letter(name) = findfirst(==(first(name)), collect(style.variable_letters))
    function by_letter(n)
        n isa _PAtom && !n.constant && letter(n.name) !== nothing && push!(found, n.name)
        for f in (n isa _PSum ? n.terms : n isa _PProd ? n.factors : n isa _PPow ? [n.base, n.exp] :
                  n isa _PCall || n isa _PDeriv || n isa _PDerivAt ? n.args : n isa _PEq ? [n.lhs, n.rhs] :
                  n isa _POrder || n isa _PDerivOf ? [n.arg] : n isa Union{_PIntegral, _PBar} ? [n.f] : _PNode[])
            by_letter(f)
        end
    end
    by_letter(root)
    # variables by letter first, in the order of the letters, then the others as declared
    return sort!(collect(found); by = s -> (something(letter(s), length(style.variable_letters) + 1), _declared_order(s), s))
end

_declared_order(name::AbstractString) = lock(() -> get(_DECLARATION_ORDER, name, typemax(Int)), _DECLARED_LOCK)
_is_integer_name(name::AbstractString) = lock(() -> get(_DECLARED, name, nothing) === :integer, _DECLARED_LOCK)

# The place of a factor in a product: numbers, constants, parameters (as declared), variables (by
# priority), exponentials, other functions, sums.
function _factor_rank(f::_PNode, vars)
    b = f isa _PPow ? f.base : f
    b isa _PNum && return (0, 0, 0, "")                                # √2 with the numbers
    if b isa _PAtom
        b.constant && return (1, 0, 0, b.name)
        _is_differential(b.name) && return (7, 0, 0, b.name)          # dx last, as in an integrand
        _is_integer_name(b.name) && return (1, -1, 0, b.name)         # an integer like a number: nπ

        i = findfirst(==(b.name), vars)
        greek = first(b.name) in 'α':'ω' || first(b.name) in 'Α':'Ω'
        i === nothing ? (return (2, Int(greek), _declared_order(b.name), b.name)) : (return (3, 0, i, b.name))
    end
    b isa _PCall && b.user && return (3, 0, length(vars) + 1, b.f)
    b isa _PDeriv && return (3, 0, length(vars) + 1, b.f * "′"^length(b.wrt))
    b isa _PDerivAt && return (3, 0, length(vars) + 1, b.f * "′"^length(b.idx))
    f isa _PPow && b isa _PProd && f.exp isa _PNum && !(f.exp.v isa Integer) && return (4, -1, 0, "")   # √(2/a) before sin
    b isa _PIntegral && return (5, 0, 0, "∫")
    b isa _PBar && return (5, 0, 0, "|")
    (b isa _PCall && b.f == "exp" || f isa _PPow && b isa _PAtom && b.name == "e") && return (4, 0, 0, "")
    b isa _PCall && return (5, 0, 0, b.f)
    return (6, 0, 0, "")
end

# The highest order of a derivative among a term's factors
function _derivative_order(t::_PNode)
    k = 0
    for f in _split(t).num
        b = f isa _PPow ? f.base : f
        b isa Union{_PDeriv, _PDerivOf} && (k = max(k, length(b.wrt)))
        b isa _PDerivAt && (k = max(k, length(b.idx)))
    end
    return k
end

# The name of a differential (dx) among a term's factors, or nothing
function _differential_in(t::_PNode)
    for f in _split(t).num
        b = f isa _PPow ? f.base : f
        b isa _PAtom && _is_differential(b.name) && return b.name
    end
    return nothing
end

# A term's powers of the variables, in priority order.
_powers(t::_PNode, vars) = (d = _term_degrees(t); [get(d, v, 0) for v in vars])

function _order_terms(terms::Vector{_PNode}, style::TypesetStyle, vars)
    order = style.order
    series = any(t -> _split(t).order !== nothing, terms)
    order === :auto && (order = series ? :ascending : :descending)
    differentials = map(_differential_in, terms)
    if order !== :as_is && all(!isnothing, differentials)
        # a differential form, ∂f/∂x dx + ∂f/∂y dy, in the order of its differentials
        terms = terms[sortperm(map(_differential_order, differentials); alg = Base.Sort.DEFAULT_STABLE)]
    elseif order !== :as_is
        keys = [vcat(_derivative_order(t), _powers(t, vars)) for t in terms]      # each computed once
        if all(k -> all(iszero, k), keys)
            # nothing to order by: as given, with numbers last
            terms = vcat(filter(t -> !(t isa _PNum), terms), filter(t -> t isa _PNum, terms))
        else
            terms = terms[sortperm(keys; rev = order === :descending, alg = Base.Sort.DEFAULT_STABLE)]
        end
        # a series' remainder O(…) comes last
        terms = vcat(filter(t -> _split(t).order === nothing, terms), filter(t -> _split(t).order !== nothing, terms))
    end
    negative(t) = _split(t).sign < 0
    lead = style.lead
    # a second or higher derivative stays first, whatever its sign: -ħ²/2m ψ″ + Vψ
    led_by_derivative = order !== :as_is && length(terms) > 1 && _derivative_order(terms[1]) >= 2 &&
                        _derivative_order(terms[1]) > _derivative_order(terms[2])
    if negative(terms[1]) && lead !== :as_is && (lead === :positive || length(terms) == 2 && !led_by_derivative)
        i = findfirst(!negative, terms)
        i === nothing || (terms = vcat(terms[i:i], terms[1:(i - 1)], terms[(i + 1):end]))
    end
    return terms
end

# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------

struct _Ctx
    style::TypesetStyle
    vars::Vector{String}
    exponent::Bool
end
_in_exponent(c::_Ctx) = _Ctx(c.style, c.vars, true)

"LaTeX for one expression, written in the current style; Giac's own LaTeX where this has no rule."
function _present_latex(e::GiacExpr)
    style = _current_style()
    root = _pnode(e)
    return _tex(root, _Ctx(style, _variables(root, style), false))
end

function _number_tex(v::Real)
    v isa Integer && return string(v)
    v isa Rational && return isone(denominator(v)) ? string(numerator(v)) : "\\frac{$(numerator(v))}{$(denominator(v))}"
    s = string(v)
    m = match(r"^([0-9.]+)e(-?\d+)$", s)
    return m === nothing ? s : m[1] * " \\times 10^{" * m[2] * "}"
end

# in an exponent, a fraction is written inline: x^{1/2}
_tex(n::_PNum, c::_Ctx) = (t = c.exponent && n.v isa Rational && !isone(denominator(n.v)) ?
                           "$(abs(numerator(n.v)))/$(denominator(n.v))" : _number_tex(abs(n.v)); n.v < 0 ? "-" * t : t)
_tex(n::_PAtom, c::_Ctx) = n.tex
_tex(n::_PRaw, c::_Ctx) = n.tex
_tex(n::_PEq, c::_Ctx) = _tex(n.lhs, c) * " = " * _tex(n.rhs, c)
_tex(n::_POrder, c::_Ctx) = "O\\left(" * _tex(n.arg, c) * "\\right)"
_tex(n::_PProd, c::_Ctx) = ((s, t) = _product_tex(n, c); s < 0 ? "-" * t : t)
_tex(n::_PPow, c::_Ctx) = n.exp isa _PNum && n.exp.v < 0 ? _tex(_PProd(_PNode[n]), c) : _power_tex(n, c)

function _tex(n::_PSum, c::_Ctx)
    terms = _order_terms(reduce(vcat, (t isa _PSum ? t.terms : _PNode[t] for t in n.terms); init = _PNode[]), c.style, c.vars)
    out = ""
    for (i, t) in enumerate(terms)
        s, body = _product_tex(t, c)
        out *= i == 1 ? (s < 0 ? "-" * body : body) : (s < 0 ? " - " * body : " + " * body)
    end
    return out
end

# a product (or any term) as its sign and its written form without the sign
function _product_tex(n::_PNode, c::_Ctx)
    p = _split(n)
    if p.order !== nothing
        inner = _unsigned_product(p, c)
        return (p.sign, "O\\left(" * inner * "\\right)")
    end
    return (p.sign, _unsigned_product(p, c))
end

function _sorted(fs, c::_Ctx)
    c.style.products === :as_is && return fs
    return sort(fs; by = f -> _factor_rank(f, c.vars), alg = Base.Sort.DEFAULT_STABLE)
end

function _unsigned_product(p, c::_Ctx)
    num, den = _sorted(p.num, c), _sorted(p.den, c)
    # derivatives written as fractions, and differentials, stand after a fraction, not in it:
    # ħ²/2m ∂²ψ/∂x², p/m dp
    if !isempty(den) || (p.coef isa Rational && !isone(denominator(p.coef)))
        # (a derivative with only a number below stays in front of the rest: ½ dM/dq q̇²)
        derivative(f) = !isempty(den) && f isa Union{_PDeriv, _PDerivOf} && startswith(_tex(f, c), "\\frac")
        differential(f) = f isa _PAtom && _is_differential(f.name)
        d = vcat(findall(derivative, num), findall(differential, num))
        if !isempty(d) && length(d) < length(num) + (isone(p.coef) ? 0 : 1)
            rest = (; p..., num = num[setdiff(eachindex(num), d)])
            return _unsigned_product(rest, c) * " " * join((_tex(num[i], c) for i in d), " ")
        end
    end
    coef = p.coef
    q = coef isa Rational ? denominator(coef) : 1
    pn = coef isa Rational ? numerator(coef) : coef
    numtex = _factors_tex(pn, num, c)
    (isempty(den) && isone(q)) && return numtex
    dentex = _factors_tex(q, den, c)
    if c.exponent || c.style.fractions === :inline
        wrap(t, fs, k) = (length(fs) + (isone(k) ? 0 : 1) > 1) ? "\\left(" * t * "\\right)" : t
        # a product above the line needs no parentheses: e^{-iEt/ħ}; one below does: e^{-t/(RC)}
        above = any(f -> f isa _PSum, num) ? wrap(numtex, num, pn) : numtex
        return above * "/" * wrap(dentex, den, q)
    end
    # a fraction in the numerator (∂L/∂q) would stack fractions: the number goes in front
    if (c.style.fractions === :front || occursin("\\frac", numtex)) && isempty(den) && !isempty(num)
        return "\\frac{$(pn)}{$(q)} " * _factors_tex(1, num, c)
    end
    # a fraction's line already groups each side: no parentheses around a lone sum
    bare(t, fs, k) = isone(k) && length(fs) == 1 && fs[1] isa _PSum ? _tex(fs[1], c) : t
    return "\\frac{" * bare(numtex, num, pn) * "}{" * bare(dentex, den, q) * "}"
end

# a numeric coefficient and factors, side by side
function _factors_tex(k, fs, c::_Ctx)
    parts = String[]
    (isempty(fs) || !isone(k)) && push!(parts, _number_tex(k))
    for f in fs
        t = f isa _PSum || f isa _PEq ? "\\left(" * _tex(f, c) * "\\right)" : _tex(f, c)
        # a number next to a number, or before a fraction, is written with a dot
        !isempty(parts) && occursin(r"^[\d.]+$", parts[end]) && occursin(r"^(\d|\\frac)", t) && (t = "\\cdot " * t)
        push!(parts, t)
    end
    return join(parts, " ")
end

# a base that needs parentheses under a power
_needs_parens(b::_PNode) = b isa _PSum || b isa _PEq || (b isa _PProd) || (b isa _PPow) ||
                           (b isa _PNum && (b.v < 0 || !(b.v isa Integer))) || (b isa _PCall && !b.user && b.f != "abs")

function _power_tex(n::_PPow, c::_Ctx)
    b, e = n.base, n.exp
    if e isa _PNum && c.style.roots === :radical && e.v isa Rational
        e.v == 1 // 2 && return "\\sqrt{" * _tex(b, c) * "}"
        e.v == 1 // 3 && return "\\sqrt[3]{" * _tex(b, c) * "}"
    end
    etex = _tex(e, _in_exponent(c))
    b isa _PAtom && b.name == "e" && return "e^{" * etex * "}"
    if b isa _PCall && b.f in _TRIG && e isa _PNum && e.v isa Integer && e.v > 0 && c.style.trig_powers === :superscript
        return _CALL_TEX[b.f] * "^{" * etex * "}" * _call_argument(b, c)
    end
    btex = _tex(b, c)
    parens = _needs_parens(b) || (b isa Union{_PDeriv, _PDerivOf} && startswith(btex, "\\frac")) ||
             (b isa _PAtom && _juxtaposed(btex))
    return (parens ? "\\left(" * btex * "\\right)" : (occursin(r"[_^]", btex) ? "{" * btex * "}" : btex)) * "^{" * etex * "}"
end

# a function's argument: bare after a named function when it is a symbol or a power of one
function _call_argument(n::_PCall, c::_Ctx)
    a = only(n.args)
    simple = a isa _PAtom || a isa _PNum && a.v isa Integer && a.v >= 0 || _symbol_power(a) !== nothing ||
             (a isa _PCall && a.user && isempty(_arguments_tex(a.args, c)))      # sin q, for q(t)
    return simple ? " " * _tex(a, c) : "\\left(" * _tex(a, c) * "\\right)"
end

function _tex(n::_PCall, c::_Ctx)
    n.user && return _tex_name(n.f) * _arguments_tex(n.args, c)
    n.f == "abs" && return "\\left|" * _tex(only(n.args), c) * "\\right|"
    n.f == "conj" && return "\\overline{" * _tex(only(n.args), c) * "}"
    if n.f == "exp" && c.style.exp === :e
        return "e^{" * _tex(only(n.args), _in_exponent(c)) * "}"
    end
    length(n.args) == 1 && return _CALL_TEX[n.f] * _call_argument(n, c)
    return _CALL_TEX[n.f] * "\\left(" * join((_tex(a, c) for a in n.args), ", ") * "\\right)"
end

# ---------------------------------------------------------------------------------------------
# Functions of time and derivatives
# ---------------------------------------------------------------------------------------------

# a name written as symbols side by side (Δx): parenthesized under a power
function _juxtaposed(tex::AbstractString)
    depth = 0
    for ch in tex
        ch == '{' ? (depth += 1) : ch == '}' ? (depth -= 1) : (ch == ' ' && depth == 0 && return true)
    end
    return false
end

# a function's arguments, left out for a function of time alone (style `arguments = :auto`)
function _arguments_tex(args::Vector{_PNode}, c::_Ctx)
    c.style.arguments === :auto && length(args) == 1 && only(args) isa _PAtom && only(args).name == c.style.time && return ""
    return "\\left(" * join((_tex(a, c) for a in args), ", ") * "\\right)"
end

# an accent over the main symbol of a name, before its subscripts and superscripts
function _accent_tex(accent::AbstractString, tex::AbstractString)
    depth = 0
    for (i, ch) in pairs(tex)
        ch == '{' ? (depth += 1) : ch == '}' ? (depth -= 1) : nothing
        depth == 0 && ch in ('_', '^', '\'') && i > 1 && return accent * "{" * tex[1:prevind(tex, i)] * "}" * tex[i:end]
    end
    return accent * "{" * tex * "}"
end

const _DOTS = ("\\dot", "\\ddot", "\\dddot")

function _tex(n::_PDeriv, c::_Ctx)
    k, v = length(n.wrt), first(n.wrt)
    name = _tex_name(n.f)
    single = length(n.args) == 1 && only(n.args) isa _PAtom && only(n.args).name == v && all(==(v), n.wrt)
    if single && c.style.derivatives === :dot && v == c.style.time && k <= 3
        return _accent_tex(_DOTS[k], name) * _arguments_tex(n.args, c)
    elseif single && c.style.derivatives === :prime
        base = occursin(r"[_^]", name) ? "{" * name * "}" : name
        return (k <= 3 ? base * "'"^k : base * "^{($k)}") * _arguments_tex(n.args, c)
    end
    return _leibniz(length(n.args) > 1 ? "\\partial" : "\\mathrm{d}", name, n.wrt)
end

function _tex(n::_PDerivOf, c::_Ctx)
    k, v = length(n.wrt), first(n.wrt)
    if n.arg isa _PAtom && !n.arg.constant && all(==(v), n.wrt)
        # a symbol, as in a derivation's left side `diff(V, t)`: V̇, V′ or dV/dt
        # a written derivative of a quantity of several variables is a partial, in time too: ∂L/∂q̇, ∂ψ/∂t
        _written_partial(n.arg.name, v) && return _leibniz("\\partial", n.arg.tex, n.wrt)
        c.style.derivatives === :dot && v == c.style.time && k <= 3 && return _accent_tex(_DOTS[k], n.arg.tex)
        c.style.derivatives === :prime && return (occursin(r"[_^]", n.arg.tex) ? "{" * n.arg.tex * "}" : n.arg.tex) * "'"^k
        return _leibniz("\\mathrm{d}", n.arg.tex, n.wrt)
    end
    return _leibniz("\\mathrm{d}", "", n.wrt) * "\\left(" * _tex(n.arg, c) * "\\right)"
end

function _tex(n::_PIntegral, c::_Ctx)
    bounds = isempty(n.limits) ? "" : "_{" * _tex(n.limits[1], c) * "}^{" * _tex(n.limits[2], c) * "}"
    f = n.f isa _PSum ? "\\left(" * _tex(n.f, c) * "\\right)" : _tex(n.f, c)
    return "\\int" * bounds * " " * f * " \\, \\mathrm{d}" * _tex(n.var, c)
end

function _tex(n::_PBar, c::_Ctx)
    f = n.f isa _PSum ? "\\left(" * _tex(n.f, c) * "\\right)" : _tex(n.f, c)
    limits = "_{" * _tex(n.lower, c) * "}" * (n.upper === nothing ? "" : "^{" * _tex(n.upper, c) * "}")
    return "\\left. " * f * " \\right|" * limits
end

function _tex(n::_PDerivAt, c::_Ctx)
    name = _tex_name(n.f)
    args = "\\left(" * join((_tex(a, c) for a in n.args), ", ") * "\\right)"
    if length(n.args) == 1
        k = length(n.idx)
        base = occursin(r"[_^]", name) ? "{" * name * "}" : name
        return (k <= 3 ? base * "'"^k : base * "^{($k)}") * args
    end
    # in the names the function was declared with: ∂L/∂q̇ at (q + εη, q̇ + εη̇, t)
    names = _argument_names(n.f)
    names !== nothing && length(names) == length(n.args) &&
        return _leibniz("\\partial", name, [names[i + 1] for i in n.idx]) * args
    return join(("\\partial_{$(i + 1)}" for i in n.idx), " ") * " " * name * args
end

# d^k f / dx^i dy^j, the variables grouped in the order they first come
function _leibniz(d::AbstractString, name::AbstractString, wrt::Vector{String})
    k = length(wrt)
    num = (k == 1 ? d : d * "^{$k}") * (isempty(name) ? "" : " " * name)
    den = join((d * " " * _tex_name(v) * (count(==(v), wrt) > 1 ? "^{$(count(==(v), wrt))}" : "") for v in unique(wrt)), " ")
    return "\\frac{" * num * "}{" * den * "}"
end

# ---------------------------------------------------------------------------------------------
# Held commands: lim, Σ, Π, ℒ, 𝒵 and the generic `name(args)`, written by held_cmd.jl
# ---------------------------------------------------------------------------------------------

struct _PHeld <: _PNode
    cmd::Symbol
    args::Tuple
end
_tex(n::_PHeld, c::_Ctx) = sprint(_print_held_latex, n.cmd, n.args)

# ---------------------------------------------------------------------------------------------
# A derivative at a point, q′(0), and a condition on one, q′(0) = v₀ (initial conditions)
# ---------------------------------------------------------------------------------------------

function _derivative_point_latex(dp::DerivativePoint)
    name = _tex_name(dp.funcname)
    base = occursin(r"[_^]", name) ? "{" * name * "}" : name
    primes = dp.order <= 3 ? "'"^dp.order : "^{($(dp.order))}"
    return base * primes * "\\left(" * join((giac_latex(giac_eval(a)) for a in dp.point_args), ", ") * "\\right)"
end
giac_latex(dp::DerivativePoint) = _derivative_point_latex(dp)
_entry_latex(dp::DerivativePoint) = _derivative_point_latex(dp)
Base.show(io::IO, ::MIME"text/latex", dp::DerivativePoint) = print(io, "\$\$", giac_latex(dp), "\$\$")

function giac_latex(dc::DerivativeCondition)
    dc.point === nothing && return dc.condition_str
    return _derivative_point_latex(dc.point) * " = " * giac_latex(convert(GiacExpr, dc.value))
end
_entry_latex(dc::DerivativeCondition) = giac_latex(dc)
Base.show(io::IO, ::MIME"text/latex", dc::DerivativeCondition) = print(io, "\$\$", giac_latex(dc), "\$\$")
