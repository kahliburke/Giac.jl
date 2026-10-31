# How a symbol's name is written: Greek letters, accents, subscripts, superscripts and primes,
# from the way the name is typed in Julia.


const _GREEK_TEX = Dict(
    'α' => "\\alpha", 'β' => "\\beta", 'γ' => "\\gamma", 'δ' => "\\delta", 'ε' => "\\varepsilon",
    'ϵ' => "\\epsilon", 'ζ' => "\\zeta", 'η' => "\\eta", 'θ' => "\\theta", 'ϑ' => "\\vartheta",
    'ι' => "\\iota", 'κ' => "\\kappa", 'λ' => "\\lambda", 'μ' => "\\mu", 'ν' => "\\nu", 'ξ' => "\\xi",
    'π' => "\\pi", 'ρ' => "\\rho", 'ϱ' => "\\varrho", 'σ' => "\\sigma", 'ς' => "\\varsigma",
    'τ' => "\\tau", 'υ' => "\\upsilon", 'φ' => "\\varphi", 'ϕ' => "\\phi", 'χ' => "\\chi",
    'ψ' => "\\psi", 'ω' => "\\omega", 'Γ' => "\\Gamma", 'Δ' => "\\Delta", 'Θ' => "\\Theta",
    'Λ' => "\\Lambda", 'Ξ' => "\\Xi", 'Π' => "\\Pi", 'Σ' => "\\Sigma", 'Υ' => "\\Upsilon",
    'Φ' => "\\Phi", 'Ψ' => "\\Psi", 'Ω' => "\\Omega", '∂' => "\\partial", 'ℓ' => "\\ell",
    'ħ' => "\\hbar", '∇' => "\\nabla")

# combining marks over a letter (a name is decomposed first, so ẍ is x and a mark)
const _ACCENT_TEX = Dict('̂' => "\\hat", '̃' => "\\tilde", '̄' => "\\bar",
                         '̅' => "\\bar", '̇' => "\\dot", '̈' => "\\ddot",
                         '⃛' => "\\dddot", '⃗' => "\\vec", '̆' => "\\breve",
                         '̌' => "\\check")

_char_map(from, to) = (length(from) == length(to) || error("unequal maps"); Dict(zip(from, to)))
const _SUB_CHARS = _char_map("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥᵦᵧᵨᵩᵪ₊₋₌₍₎",
                             "0123456789aeoxhklmnpstijruvβγρφχ+-=()")
const _SUP_CHARS = _char_map("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵁⱽᵂᵅᵝᵞᵟᵋᶿᵠᵡ",
                             "0123456789+-=()niabcdefghjklmoprstuvwxyzABDEGHIJKLMNOPRUVWαβγδεθφχ")
# marks after a name: superscripts of their own
const _MARK_TEX = Dict('ᵀ' => "\\top", '★' => "\\star", '⋆' => "\\star", '†' => "\\dagger")
const _PRIMES = Dict('′' => 1, '″' => 2, '‴' => 3, '\'' => 1)

# Display names set with `@giac_var name => "tex"`, by Giac name
const _TEX_NAMES = Dict{String, String}()

"""
    _tex_name(name) -> String

LaTeX for a symbol name as typed in Julia:

- a letter, Greek letter or letter with accents is the main symbol (`x`, `θ`, `x̂`, `V̇`, `ẍ`);
  several are written side by side (`Δx`), and a run of several Latin letters is a word in
  upright type (`xdot`);
- trailing digits, subscript characters (`₀`, `ₚ`, `ᵢ`) and parts after `_` are subscripts:
  `x12`, `x₁₂`, `x_12` are x₁₂, and `x_1_2` is x₁,₂; a part that is a word is upright
  (`x_ref`);
- superscript characters, `ᵀ` (⊤), `★` and primes after the name are superscripts (`x⁺`, `Kᵀ`,
  `P★`, `x′`).

A display name set with `@giac_var name => "tex"` takes precedence, as does the one an array
declaration records for an entry with a negative index (`c_m1` is c₋₁).
"""
function _tex_name(name::AbstractString)
    custom = get(_TEX_NAMES, name, nothing)
    custom === nothing || return custom
    differential = match(_DIFFERENTIAL, Base.Unicode.normalize(name, :NFD))
    differential === nothing || return "\\mathrm{d}" * _tex_name(Base.Unicode.normalize(differential[1], :NFC))
    plain = _unspell(name)
    parts = split(Base.Unicode.normalize(plain, :NFD), '_')
    core = Tuple{Char, Vector{Char}}[]          # letters with their marks
    subs, sups, primes = String[], String[], 0
    run, runkind = IOBuffer(), :none            # a run of subscript digits, or superscript characters
    function flush!()
        s = String(take!(run))
        isempty(s) || push!(runkind === :sub ? subs : sups, s)
        runkind = :none
    end
    function add!(kind, s)
        runkind === kind || (flush!(); runkind = kind)
        print(run, s)
    end
    for c in parts[1]
        if haskey(_SUB_CHARS, c)
            add!(:sub, _SUB_CHARS[c])
        elseif isdigit(c) && !isempty(core)
            add!(:sub, c)
        elseif haskey(_SUP_CHARS, c)
            add!(:sup, _greek_or(_SUP_CHARS[c]))
        elseif haskey(_MARK_TEX, c)
            flush!(); push!(sups, _MARK_TEX[c])
        elseif haskey(_PRIMES, c)
            flush!(); primes += _PRIMES[c]
        elseif Base.Unicode.category_code(c) == Base.Unicode.UTF8PROC_CATEGORY_MN
            isempty(core) || push!(last(core)[2], c)
        elseif runkind === :sub && isascii(c) && isletter(c)
            add!(:sub, c)                       # x1a: the subscript goes on
        else
            flush!(); push!(core, (c, Char[]))
        end
    end
    flush!()
    isempty(core) && return "\\mathrm{" * replace(name, "_" => "\\_") * "}"
    append!(subs, _subscript_part(p) for p in parts[2:end] if !isempty(p))
    tex = _core_tex(core)
    if primes > 0
        isempty(sups) ? (tex *= "'"^primes) : pushfirst!(sups, "\\prime"^primes)
    end
    isempty(subs) || (tex *= "_{" * join(subs, ",") * "}")
    isempty(sups) || (tex *= "^{" * join(sups, " ") * "}")
    return tex
end

_greek_or(c::Char) = get(_GREEK_TEX, c, string(c))

# a differential: `d` before a single symbol with its marks, subscripts and primes (dx, dq̇, dθ₁,
# dx_ref, dH), not a word (dist)
const _DIFFERENTIAL = r"^d(\p{L}[\p{Mn}\p{No}\d′″‴]*(?:_\w+)*)$"

# the main symbol: letters side by side, a run of plain Latin letters as an upright word
function _core_tex(core)
    pieces = String[]
    i = 1
    while i <= length(core)
        c, marks = core[i]
        j = i
        while isempty(marks) && isascii(c) && isletter(c) && j < length(core) &&
              isempty(core[j + 1][2]) && isascii(core[j + 1][1]) && isletter(core[j + 1][1])
            j += 1
        end
        if j > i
            push!(pieces, "\\mathrm{" * String([first(core[k]) for k in i:j]) * "}")
        else
            tex = _greek_or(c)
            for m in marks
                haskey(_ACCENT_TEX, m) && (tex = _ACCENT_TEX[m] * "{" * tex * "}")
            end
            push!(pieces, tex)
        end
        i = j + 1
    end
    return join(pieces, " ")
end

# a subscript written after `_`: a number, a letter, or a word
function _subscript_part(p::AbstractString)
    occursin(r"^\d+$", p) && return String(p)
    occursin(r"^[A-Za-z]\d*$", p) && return String(p)                 # i, m1
    occursin(r"^[A-Za-z]+\d*$", p) && return "\\mathrm{" * p * "}"   # ref, max2
    return _tex_name(Base.Unicode.normalize(String(p), :NFC))
end

# Letters Giac's parser reserves are spelled out in names (see `GIAC_RESERVED_LETTERS`): the
# letters again, for display
_unspell(name::AbstractString) = occursin(r"Delta|Sigma|partial", name) ? replace(name, map(reverse, GIAC_RESERVED_LETTERS)...) : name

# The base and the order of a name with dots over its last letter (`q̇` is (q, 1), `q̈` (q, 2)),
# or nothing
function _dotted(name::AbstractString)
    s = Base.Unicode.normalize(name, :NFD)
    order = 0
    while !isempty(s) && last(s) in ('̇', '̈', '⃛')
        order += last(s) == '̇' ? 1 : last(s) == '̈' ? 2 : 3
        s = chop(s)
    end
    return order == 0 || isempty(s) ? nothing : (Base.Unicode.normalize(s, :NFC), order)
end

# A display name for `@giac_var name => "tex"`; `L"…"` strings keep their math without the `$`
function _set_tex_name!(name::AbstractString, tex)
    _TEX_NAMES[replace(name, GIAC_RESERVED_LETTERS...)] = strip(string(tex), '$')
    return nothing
end

# `q̇` declared after `q(t)`: the derivative of `q` in its variable, of the order of the dots
function _dotted_variable(name::AbstractString, base, order::Int)
    if base isa GiacExpr && iscall(base) && symb_funcname(base) == "of"
        args = arguments(base)
        length(args) == 2 && is_identifier(args[2]) && return Commands.diff(base, args[2], order)
    end
    return giac_identifier(name)
end
