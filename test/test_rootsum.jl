# Sums over the roots of a polynomial, for rational functions whose denominators have no roots
# in radicals

@testset "Root sums" begin
    @giac_var x
    C = Giac.Commands
    value(e, v) = to_julia(C.evalf(C.subst(e, x, v)))
    q = x^5 - x + 1

    @testset "integrals" begin
        F = C.integrate(1 / q, x)
        @test string(F) == "rootsum(z^5-z+1,z,ln(x-z)/(5*z^4-1))"
        @test Giac._free_names(F) == Set(["x"])
        for f in (1 / q, (x^2 + 1) / q, 1 / (q * (x - 1)^2), x^3 / q^2, (x^6 + 2) / (q * (x^2 + 1)), 1 / (x^7 + x + 1))
            F = C.integrate(f, x)
            @test Giac._has_rootsum(F)
            @test C.simplify(C.diff(F, x) - f) == 0
            @test value(C.diff(F, x), 7 // 10) ≈ value(f, 7 // 10) rtol = 1e-12
        end
        @test !Giac._has_rootsum(C.integrate(1 / (x^4 + 1), x))           # roots in radicals: GIAC's
    end

    @testset "sums, derivatives and values" begin
        @giac_var z
        @test C.simplify(Giac.rootsum(z^3 - 2, z, z^2)) == 0                 # the power sum s₂
        @test C.simplify(Giac.rootsum(z^3 - 2z - 5, z, z^3)) == 15
        @test C.simplify(Giac.rootsum(substitute(q, x => z), z, 1 / (x - z))) == C.simplify(C.diff(q, x) / q)
        @test to_julia(C.evalf(Giac.rootsum(z^5 - z + 1, z, z^5))) ≈ -5
        @test Giac.giac_latex(C.integrate(1 / q, x)) == "\\sum_{z^{5} - z + 1 = 0} \\frac{\\ln\\left(x - z\\right)}{5 z^{4} - 1}"
        # a call with no rule on the tree beside one GIAC cannot differentiate
        @test C.diff(Giac.SpecialFunctions.besselj(0, x) * abs(x), x) ==
              -abs(x) * Giac.SpecialFunctions.besselj(1, x) + Giac.SpecialFunctions.besselj(0, x) * abs(x) / x
    end
end
