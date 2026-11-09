# Elliptic integrals: values against Mathematica's, derivatives against differences, rules and
# integrals

import SpecialFunctions as SF

@testset "Elliptic integrals" begin
    G = Giac
    C = Giac.Commands
    @testset "values" begin
        # Mathematica's, with its branches for m > 1 and the principal value for n > 1
        for (v, w) in ((G.ellipk(2.0), 1.31102877714605990521 - 1.31102877714605990525im),
                       (G.ellipe(2.0), 0.599070117367796103 + 0.599070117367796103im),
                       (G.ellipk(-3.0), 1.07825782374982161771),
                       (G.ellipf(1.2, 0.7), 1.4294484330227628),
                       (G.ellipf(2.5, 0.7), 3.4768751906448907),
                       (G.ellipe(2.5, 0.7), 1.8713294303838581),
                       (G.ellippi(0.3, 1.1, 0.6), 1.3922279110617555),
                       (G.ellippi(0.3, 0.6), 2.377858827830847),
                       (G.ellipf(0.4, 3.0), 0.439965510363979),
                       (G.ellippi(1.5, 0.6), -0.6413534067491944),
                       (G.ellipf(0.3 + 0.2im, 0.5 - 0.1im), 0.30001467936156084 + 0.20401127366118074im))
            @test abs(v - w) < 1e-14
        end
        @test G.ellipk(0.5) isa Float64 && G.ellipk(2.0) isa ComplexF64
        @test abs(G.ellipk(big"0.5") - big"1.854074677301371918433850347195260046217598823521766905585928045056021776838120") < big(10.0)^-70
        @test G.ellipk(0.3) ≈ SF.ellipk(0.3)
    end

    @giac_var m φ n a x
    π_ = giac_eval("pi")
    @testset "rules, derivatives and series" begin
        @test G.ellipk(convert(GiacExpr, 0)) == π_ / 2
        @test G.ellipe(convert(GiacExpr, 1)) == 1
        @test G.ellipf(π_ / 2, m) == G.ellipk(m) && G.ellipe(π_, m) == 2G.ellipe(m)
        @test G.ellipf(φ, 0) == φ && G.ellippi(0, φ, m) == G.ellipf(φ, m)
        @test string(G.ellipf(a / 2, 1 / sin(a / 2)^2)) == "ellipk(sin(a/2)^2)*abs(sin(a/2))"   # m sin²φ = 1
        @test SF.ellipk(m) == G.ellipk(m)
        # each partial derivative against a central difference
        at = Dict(m => 0.37, φ => 0.81, n => 0.23)
        num(e, d) = to_julia(C.evalf(substitute(e, d)))
        for e in (G.ellipk(m), G.ellipe(m), G.ellipf(φ, m), G.ellipe(φ, m), G.ellippi(n, m), G.ellippi(n, φ, m))
            for v in (m, φ, n)
                string(v) in Giac._free_names(e) || continue
                h = 1e-6
                fd = (num(e, merge(at, Dict(v => at[v] + h))) - num(e, merge(at, Dict(v => at[v] - h)))) / 2h
                @test abs(num(C.diff(e, v), at) - fd) < 1e-7
            end
        end
        @test C.simplify(C.diff(G.ellipe(φ, m), φ)) == sqrt(1 - m * sin(φ)^2)
        @test string(C.series(G.ellipk(m), m, 0, 3)) == "pi/2+pi/8*m+9*pi/128*m^2+25*pi/512*m^3+m^4*order_size(m)"
        @test Giac.giac_latex(G.ellippi(n, φ, m) + G.ellipk(m)^2) == "\\Pi\\left(n; \\varphi \\middle| m\\right) + K\\left(m\\right)^{2}"
    end

    @testset "integrals" begin
        for (f, F) in ((1 / sqrt(1 - m * sin(x)^2), G.ellipf(x, m)),
                       (sqrt(1 - m * sin(x)^2), G.ellipe(x, m)),
                       (1 / sqrt(2 - cos(x)^2), G.ellipf(x, -1)),
                       (sin(x)^2 / sqrt(1 - m * sin(x)^2), G.ellipf(x, m) / m - G.ellipe(x, m) / m),
                       (1 / ((1 - n * sin(x)^2) * sqrt(1 - m * sin(x)^2)), G.ellippi(n, x, m)),
                       (1 / sqrt(1 - x^4), G.ellipf(asin(x), -1)),
                       (1 / sqrt((1 - x^2) * (1 - m * x^2)), G.ellipf(asin(x), m)),
                       (sqrt(1 - m * x^2) / sqrt(1 - x^2), G.ellipe(asin(x), m)),
                       (sqrt(1 + cos(x)^2), sqrt(convert(GiacExpr, 2)) * G.ellipe(x, 1 // 2)))
            @test C.integrate(f, x) == F
        end
        f = 1 / sqrt(1 + 3x^2 + 2x^4)                                    # an imaginary amplitude
        @test Giac._is_antiderivative(C.integrate(f, x), f, x)
        @test Giac._elliptic_integral(1 / sqrt(1 + x^3), x) === nothing
        # definite: complete integrals, and the pendulum's period from m sin²φ = 1
        @test C.integrate(1 / sqrt(1 - m * sin(x)^2), x, 0, π_ / 2) == G.ellipk(m)
        @test C.integrate(sqrt(1 - x^2 / 2) / sqrt(1 - x^2), x, 0, 1) == G.ellipe(convert(GiacExpr, 1 // 2))
        @test C.integrate(1 / sqrt(cos(x) - cos(π_ / 3)), x, 0, π_ / 3) == sqrt(convert(GiacExpr, 2)) * G.ellipk(convert(GiacExpr, 1 // 4))
        T = C.integrate(1 / sqrt(cos(x) - cos(a)), x, 0, a)
        @test string(T) == "sqrt(2)*ellipk(sin(a/2)^2)"
        @test abs(to_julia(C.evalf(substitute(T, a => 1.1))) - sqrt(2) * G.ellipk(sin(0.55)^2)) < 1e-12
    end
end
