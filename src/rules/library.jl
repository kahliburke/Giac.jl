# Rule sets for identities Giac does not apply on its own, each with the conditions under which
# it holds. The conditions ask Giac, so declared assumptions decide where they apply:
# with `@giac_var p::positive q::positive`, `log(p*q)` expands and `log(x*y)` does not.

const _LIBRARY = Dict{Symbol, Vector{Rule}}()
const _LIBRARY_LOCK = ReentrantLock()

# a rule set, built on first use (building one evaluates its patterns in Giac)
_library(build, name::Symbol) = lock(() -> get!(build, _LIBRARY, name), _LIBRARY_LOCK)

"""
    log_expand()
    log_combine()

Logarithms of products split into sums of logarithms, and back: `log(a*b) = log(a) + log(b)` and
`log(a/b) = log(a) - log(b)`, for positive `a` and `b`. (Giac itself writes `log(a^n)` as
`n*log(a)` for positive `a`.)
"""
log_expand() = _library(:log_expand) do
    @rules begin
        log(~a * ~b) => log(~a) + log(~b) where is_positive(~a) && is_positive(~b)
    end
end

"See [`log_expand`](@ref)."
log_combine() = _library(:log_combine) do
    @rules begin
        log(~a) + log(~b) => log(~a * ~b) where is_positive(~a) && is_positive(~b)
        log(~a) - log(~b) => log(~a / ~b) where is_positive(~a) && is_positive(~b)
    end
end

"""
    power_expand()
    power_combine()

A power of a product as a product of powers, `(a*b)^n = a^n * b^n`, and back, where it holds:
for an integer `n`, or nonnegative `a` and `b`. Square roots are powers `1/2`, so
`√a·√b = √(ab)` for nonnegative `a` and `b`.
"""
power_expand() = _library(:power_expand) do
    @rules begin
        (~a * ~b)^~n => ~a^~n * ~b^~n where is_integer(~n) || is_nonnegative(~a) && is_nonnegative(~b)
    end
end

"See [`power_expand`](@ref)."
power_combine() = _library(:power_combine) do
    @rules begin
        ~a^~n * ~b^~n => (~a * ~b)^~n where is_integer(~n) || is_nonnegative(~a) && is_nonnegative(~b)
    end
end

"""
    exp_expand()
    exp_combine()

`exp(a + b) = exp(a) * exp(b)`, and back.
"""
exp_expand() = _library(:exp_expand) do
    @rules begin
        exp(~a + ~b) => exp(~a) * exp(~b)
    end
end

"See [`exp_expand`](@ref)."
exp_combine() = _library(:exp_combine) do
    @rules begin
        exp(~a) * exp(~b) => exp(~a + ~b)
    end
end

"""
    trig_simplify()

Shorter forms of trigonometric and hyperbolic expressions: `sin²t + cos²t = 1`,
`1 - sin²t = cos²t`, `sin t / cos t = tan t`, `2 sin t cos t = sin 2t`,
`cos²t - sin²t = cos 2t`, `cosh²t - sinh²t = 1`. Sums match up to a common factor, so
`3sin²t + 3cos²t` becomes `3` and `sin t cos t` becomes `sin(2t)/2`.
"""
trig_simplify() = _library(:trig_simplify) do
    @rules begin
        sin(~t)^2 + cos(~t)^2 => 1
        1 - sin(~t)^2 => cos(~t)^2
        1 - cos(~t)^2 => sin(~t)^2
        sin(~t) / cos(~t) => tan(~t)
        2 * sin(~t) * cos(~t) => sin(2 * ~t)
        cos(~t)^2 - sin(~t)^2 => cos(2 * ~t)
        cosh(~t)^2 - sinh(~t)^2 => 1
    end
end
