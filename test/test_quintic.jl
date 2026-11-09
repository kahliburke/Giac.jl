# The roots of solvable quintics in radicals

@testset "Solvable quintics" begin
    @giac_var x
    C = Giac.Commands
    value(e) = Complex{Float64}(to_julia(C.evalf(e, 30)))
    radical(r) = !occursin("rootof", string(r)) && !Giac._has_decimal(r)
    # Galois groups D₅, F₂₀, D₅, C₅ (the periods of the 11th roots of unity), and a multiple
    for p in (x^5 - 5x + 12, x^5 + 15x + 12, x^5 + 20x + 32, x^5 + x^4 - 4x^3 - 3x^2 + 3x + 1, 2x^5 - 10x + 24)
        r = C.cSolve(p ~ 0, x)
        @test length(r) == 5 && radical(r)
        @test maximum(abs(value(C.subst(p, x, r[i]))) for i in 1:5) < 1e-9
        @test length(unique(round.([value(r[i]) for i in 1:5]; digits = 8))) == 5
    end
    @test Giac._has_decimal(C.cSolve(x^5 - x + 1 ~ 0, x))                   # S₅: no radicals
    @test Giac._quintic_radicals(Rational{BigInt}[1, 0, 0, 0, -1, 1]) === nothing
end
