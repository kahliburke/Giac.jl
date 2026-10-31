# Typesetting the forms physics writes: the imaginary unit, wave equations

@testset "Physics typesetting" begin
    L = Giac.giac_latex
    @giac_var x t k ω ħ::positive m::positive ψ(x, t) V(x)

    @testset "the imaginary unit" begin
        @test L(exp(im * (k * x - ω * t))) == "e^{i \\left(k x - \\omega t\\right)}"
        @test L(-im * x) == "-i x"
        @test L(im * ħ * diff(exp(im * (k * x - ω * t)), t)) == "\\hbar \\omega e^{i \\left(k x - \\omega t\\right)}"
    end

    @testset "the Schrödinger equation" begin
        SE = im * ħ * diff(ψ, t) ~ -ħ^2 / (2m) * diff(ψ, x, 2) + V * ψ
        # the derivative after its coefficient's fraction, and first in its sum though negative
        @test L(SE) == "i \\hbar \\frac{\\partial \\psi}{\\partial t} = -\\frac{\\hbar^{2}}{2 m} \\frac{\\partial^{2} \\psi}{\\partial x^{2}} + V\\left(x\\right) \\psi\\left(x, t\\right)"
    end

    @testset "separation of variables" begin
        @giac_var φ(x) T(t)
        SE = im * ħ * diff(ψ, t) ~ -ħ^2 / (2m) * diff(ψ, x, 2) + V * ψ
        # substituting for ψ(x, t) reaches its derivatives
        sep = Giac.Commands.expand(substitute(SE, ψ => φ * T) / (φ * T))
        @test L(sep) == "\\frac{i \\hbar \\dot{T}}{T} = -\\frac{\\hbar^{2}}{2 m \\varphi\\left(x\\right)} \\frac{\\mathrm{d}^{2} \\varphi}{\\mathrm{d} x^{2}} + V\\left(x\\right)"
    end
end
