# Tests for additional math operations on GiacExpr (PR #9 by @jverzani)
# Covers: degree/pi trig variants, sec/csc/cot variants, sincos paired forms,
# deg2rad/rad2deg, exp2/exp10/log1p/log(b,x), adjoint, zero/one, and the
# widened ^ that accepts any Number on either side.

using Giac.Commands: simplify

@testset "Math Operations (PR #9)" begin

    # ========================================================================
    # Degree-based forward trig: sind, cosd, secd, cscd, cotd
    # Implementation: sind(x) = sin(deg2rad(x)) etc.
    # ========================================================================

    @testset "Degree forward trig" begin
        @test string(sind(giac_eval("30")))  == "1/2"
        @test string(sind(giac_eval("90")))  == "1"
        @test string(cosd(giac_eval("60")))  == "1/2"
        @test string(cosd(giac_eval("0")))   == "1"
        @test string(secd(giac_eval("60")))  == "2"
        @test string(cscd(giac_eval("30")))  == "2"
        @test string(cotd(giac_eval("45")))  == "1"
    end

    # ========================================================================
    # Pi-multiple trig: sinpi, cospi
    # Implementation: sinpi(x) = sin(pi*x), cospi(x) = cos(pi*x)
    # ========================================================================

    @testset "Pi-multiple trig" begin
        @test string(sinpi(giac_eval("1/6")))  == "1/2"
        @test string(sinpi(giac_eval("1/2")))  == "1"
        @test string(cospi(giac_eval("1/3")))  == "1/2"
        @test string(cospi(giac_eval("0")))    == "1"
    end

    # ========================================================================
    # Degree-based inverse trig: asind, acosd, atand
    # Implementation MUST be rad2deg(asin(x)) (NOT asin(deg2rad(x))).
    # Note: GIAC keeps the literal 180/pi factor; simplify reduces to integer.
    # ========================================================================

    @testset "Degree inverse trig" begin
        @test string(simplify(asind(giac_eval("1"))))   == "90"
        @test string(simplify(asind(giac_eval("1/2")))) == "30"
        @test string(simplify(acosd(giac_eval("0"))))   == "90"
        @test string(simplify(acosd(giac_eval("1"))))   == "0"
        @test string(simplify(atand(giac_eval("1"))))   == "45"
        @test string(simplify(atand(giac_eval("0"))))   == "0"
    end

    # ========================================================================
    # Paired trig: sincos, sincosd, sincospi
    # Each MUST return a Tuple{GiacExpr, GiacExpr} (NOT a single GiacExpr).
    # ========================================================================

    @testset "Paired trig (tuples)" begin
        @giac_var x

        s, c = sincos(x)
        @test s isa GiacExpr
        @test c isa GiacExpr
        @test string(s) == "sin(x)"
        @test string(c) == "cos(x)"

        sd, cd = sincosd(giac_eval("30"))
        @test sd isa GiacExpr
        @test cd isa GiacExpr
        @test string(sd) == "1/2"
        # cosd(30) = sqrt(3)/2; GIAC's exact form is "sqrt(3)/2"
        @test string(cd) == "sqrt(3)/2"

        sp, cp = sincospi(giac_eval("1/2"))
        @test sp isa GiacExpr
        @test cp isa GiacExpr
        @test string(sp) == "1"
        @test string(cp) == "0"
    end

    # ========================================================================
    # Angle conversion: deg2rad, rad2deg
    # ========================================================================

    @testset "deg2rad / rad2deg" begin
        @test string(simplify(deg2rad(giac_eval("180"))))   == "pi"
        @test string(simplify(deg2rad(giac_eval("90"))))    == "pi/2"
        @test string(simplify(rad2deg(giac_eval("pi"))))    == "180"
        @test string(simplify(rad2deg(giac_eval("pi/2")))) == "90"
    end

    # ========================================================================
    # Exponential / logarithm extensions
    # ========================================================================

    @testset "exp2 / exp10 / log1p" begin
        @test string(exp2(giac_eval("3")))   == "8"
        @test string(exp10(giac_eval("2")))  == "100"
        @test string(log1p(giac_eval("0")))  == "0"
    end

    @testset "log(b, x) — arbitrary base" begin
        # log(2, 8) == 3; GIAC may leave this as ln(8)/ln(2). simplify reduces.
        @test string(simplify(log(giac_eval("2"), giac_eval("8")))) == "3"
        @test string(simplify(log(giac_eval("10"), giac_eval("100")))) == "2"

        # Number-base form also works (promotes the base to GiacExpr).
        @giac_var x
        r = log(2, x)
        @test r isa GiacExpr
        # Same value as log(giac_eval("2"), x).
        @test string(simplify(r - log(giac_eval("2"), x))) == "0"
    end

    # ========================================================================
    # adjoint (so postfix ' works on GiacExpr) — equivalent to conj
    # ========================================================================

    @testset "adjoint" begin
        @giac_var x
        @test string(adjoint(x)) == string(conj(x))
        # Postfix ' uses adjoint
        @test string(x') == string(conj(x))
        # Conjugate of a real-typed variable is itself in GIAC's view
        @test string(simplify(adjoint(giac_eval("3+4*i")))) == "3-4*i"
    end

    @testset "transpose: scalars unchanged, matrices transposed" begin
        @giac_var a b
        @test string(transpose(a + b)) == "a+b"
        M = GiacExpr[a b; 0 1]
        @test string.(transpose(M)) == ["a" "0"; "b" "1"]
        @test string.(transpose(M) * M) == string.(permutedims(M) * M)
        @test string(transpose(giac_eval("[[1,2],[3,4]]"))) == "matrix[[1,3],[2,4]]"
        @test string(simplify(adjoint(giac_eval("[[1,i],[0,2]]")))) == "matrix[[1,0],[-i,2]]"
    end

    # ========================================================================
    # zero / one — needed for Julia's identity-element machinery (matrices etc.)
    # ========================================================================

    @testset "zero / one" begin
        @giac_var x
        @test string(zero(x))            == "0"
        @test string(zero(GiacExpr))     == "0"
        @test string(one(x))             == "1"
        @test string(one(GiacExpr))      == "1"
        @test zero(x) isa GiacExpr
        @test one(x) isa GiacExpr
    end

    # ========================================================================
    # Power widening: GiacExpr^Number and Number^GiacExpr both work via promote.
    # ========================================================================

    @testset "Power widening" begin
        @giac_var x

        # GiacExpr ^ Integer (was the only form before PR #9; must still work).
        @test string(x^2) == "x^2"
        @test string(x^0) == "1"

        # GiacExpr ^ Rational
        @test string(x^(1//2)) == "sqrt(x)"

        # GiacExpr ^ Float
        @test x^2.0 isa GiacExpr

        # Number ^ GiacExpr
        @test 2^x isa GiacExpr
        @test string(2^x) == "2^x"

        # Float ^ GiacExpr
        @test 2.5^x isa GiacExpr

        # GiacExpr ^ GiacExpr (already supported pre-PR-9; no regression)
        @test string(x^x) == "x^x"
    end
end
