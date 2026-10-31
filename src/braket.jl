# Dirac's bra-ket notation: ⟨m|n⟩, ⟨m|A|n⟩, ⟨m|, |n⟩ as held expressions, typeset as written.

"""
    braket(m, n)
    braket(m, A, n)
    bra(m)
    ket(n)

The inner product ⟨m|n⟩, the matrix element ⟨m|A|n⟩, the bra ⟨m| and the ket |n⟩, held: they are
written in Dirac's notation and not evaluated. The labels and the operator are any expressions or
names, shown as declared (`x̂` as x̂). On a derivation's left side they label the integral that
gives their value:

```julia
@giac_var x a::positive m::integer n::integer
φ(k) = sqrt(2 / a) * sin(k * π * x / a)
@derivation begin
    braket(m, n) ~ integral(φ(m) * φ(n), x, 0, a) ~ integrate(φ(m) * φ(n), x, 0, a)
end
```
"""
braket(m, n) = _dirac(:braket, m, n)
braket(m, A, n) = _dirac(:braket, m, A, n)
"The bra ⟨m|; see [`braket`](@ref)."
bra(m) = _dirac(:bra, m)
"The ket |n⟩; see [`braket`](@ref)."
ket(n) = _dirac(:ket, n)

"""
    kronecker_delta(m, n)

The Kronecker delta δₘₙ, held: 1 when `m` and `n` are equal and 0 otherwise, written δₘₙ. With
two numbers it is that value. An orthonormal set of states has `braket(m, n) ~ kronecker_delta(m, n)`.
"""
function kronecker_delta(m, n)
    a, b = convert(GiacExpr, m), convert(GiacExpr, n)
    (_is_plain_number(a) && _is_plain_number(b)) && return convert(GiacExpr, Int(string(a) == string(b)))
    string(a) == string(b) && return convert(GiacExpr, 1)
    return _dirac(:kronecker_delta, a, b)
end

_dirac(kind::Symbol, args...) =
    GiacExpr(CallNode("of", Node[SymNode(string(kind)); [_node(convert(GiacExpr, a)) for a in args]]))
