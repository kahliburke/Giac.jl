# The roots of solvable quintics in radicals, and cube roots denested

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

@testset "Cube roots denested" begin
    @giac_var x
    C = Giac.Commands
    c(n) = convert(GiacExpr, n)^(1 // 3)
    value(e) = to_julia(C.evalf(e, 30))
    depth1(e) = !occursin(r"\([^()]*\^\(1/3\)[^()]*\)\^\(1/3\)", string(e))
    for (e, form) in (((c(2) - 1)^(1 // 3), "(1-2^(1/3)+2^(2/3))/9^(1/3)"),        # Ramanujan's
                      ((7c(20) - 19)^(1 // 3), "(5+20^(1/3)-20^(2/3))/45^(1/3)"))   # ∛(5/3) - ∛(2/3)
        r = C.simplify(e)
        @test string(r) == form
        @test value(r) ≈ value(e) rtol = 1e-25
    end
    @test C.simplify((3c(3) - 4)^(1 // 3)) == (3c(3) - 4)^(1 // 3)               # does not denest
    r = C.simplify(2 * (c(2) - 1)^(1 // 3) * x)                                  # a factor of a product
    @test !occursin("-1)^(1/3)", string(r)) && C.simplify(r - 2x * (1 - c(2) + c(4)) / c(9)) == 0
    @test Giac._cubic_cube(Rational{BigInt}[1, -1, 1], 2) == [-9, 9, 0]         # (1 - ∛2 + ∛4)³ = 9(∛2 - 1)
end
