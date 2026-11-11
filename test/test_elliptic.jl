# Elliptic integrals and Jacobi's functions: values against Mathematica's, derivatives against
# differences, rules, integrals, and the autonomous equations solved with them

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
        # m sin²φ = 1: F = K(sin²φ) sin φ for |φ| ≤ π/2, odd in φ; left as it is where |φ| ≤ π/2 is not known
        for φ0 in (giac_eval("1/2"), giac_eval("-1/2"))
            @test G.ellipf(φ0, 1 / sin(φ0)^2) == G.ellipk(sin(φ0)^2) * sin(φ0)
        end
        @test string(G.ellipf(a / 2, 1 / sin(a / 2)^2)) == "ellipf(a/2,1/sin(a/2)^2)"
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
        for a0 in (1.1, -1.1)                   # the integral from 0 to a < 0 is negative
            @test abs(to_julia(C.evalf(substitute(T, a => a0))) - sign(a0) * sqrt(2) * G.ellipk(sin(a0 / 2)^2)) < 1e-12
        end
    end

    @testset "Jacobi's functions" begin
        # Mathematica's values, m > 1 and a complex u among them
        for (v, w) in ((G.jacobiam(0.7, 0.4), 0.6793975853463283), (G.jacobisn(0.7, 0.4), 0.6283244887511652),
                       (G.jacobicn(3.1, 0.8), -0.4008800843489769), (G.jacobidn(5.3, 0.6), 0.6880395546674922),
                       (G.jacobisn(0.9, 2.5), 0.6083703815327519), (G.jacobicn(0.9, 2.5), 0.7936532485120276),
                       (G.jacobidn(0.9, 2.5), 0.27333806391396553), (G.jacobisn(1.2, -2.0), 0.998775699224254),
                       (G.jacobiam(10.0, 0.9), 5.9752822590846275),
                       (G.jacobisn(0.3 + 0.4im, 0.5), 0.32892700418250387 + 0.38683569436766424im))
            @test abs(v - w) < 1e-14
        end
        @giac_var u
        at = Dict(u => 0.71, m => 0.37)
        num(e, d) = to_julia(C.evalf(substitute(e, d)))
        for e in (G.jacobiam(u, m), G.jacobisn(u, m), G.jacobicn(u, m), G.jacobidn(u, m)), v in (u, m)
            h = 1e-6
            fd = (num(e, merge(at, Dict(v => at[v] + h))) - num(e, merge(at, Dict(v => at[v] - h)))) / 2h
            @test abs(num(C.diff(e, v), at) - fd) < 1e-7
        end
        @test C.diff(G.jacobisn(u, m), u) == G.jacobicn(u, m) * G.jacobidn(u, m)
        @test G.jacobisn(u, 0) == sin(u) && G.jacobicn(-u, m) == G.jacobicn(u, m) && G.jacobidn(u, 1) == 1 / cosh(u)
        @test G.jacobiam(G.ellipf(φ, m), m) == φ && G.jacobisn(G.ellipk(m), m) == 1
        @test Giac.giac_latex(G.jacobisn(u, m)) == "\\operatorname{sn}\\left(u \\middle| m\\right)"
    end

    @testset "autonomous equations" begin
        @giac_var t y(t) k::positive
        y′ = Giac.invoke_cmd(:diff, y, t)
        D2(Y) = C.diff(Y, t, 2)
        C0, C1 = giac_eval("c_0"), giac_eval("c_1")
        pendulum, duffing = Y -> D2(Y) + sin(Y), Y -> D2(Y) + Y + Y^3
        # general solutions, checked where the constants give real parameters below 1
        for (L, values) in ((pendulum, Dict{GiacExpr, Any}(C0 => 1.9, C1 => 0.3)),
                            (Y -> D2(Y) + k * sin(Y), Dict{GiacExpr, Any}(C0 => 1.9, C1 => 0.3, k => 1.7)),
                            (duffing, Dict{GiacExpr, Any}(C0 => 0.7, C1 => 0.3)))
            S = C.desolve(L(y) ~ 0, y)
            @test Giac._residual(L(y) ~ 0, S, t, y; values = values) === true
        end
        @test string(C.desolve(pendulum(y) ~ 0, y)) == "2*jacobiam(sqrt(c_0+1)*(t+c_1)/sqrt(2),2/(c_0+1))"
        # from rest, swinging, turning over, and Duffing's from rest
        @test string(C.desolve([pendulum(y) ~ 0, y(0) ~ a, y′(0) ~ 0], y)) ==
              "2*asin(sin(a/2)*jacobicn(t,sin(a/2)^2)/jacobidn(t,sin(a/2)^2))"
        @test string(C.desolve([pendulum(y) ~ 0, y(0) ~ 0, y′(0) ~ 1], y)) == "2*asin(jacobisn(t,1/4)/2)"
        @test string(C.desolve([pendulum(y) ~ 0, y(0) ~ 0, y′(0) ~ 3], y)) == "2*jacobiam(3*t/2,4/9)"
        @test string(C.desolve([duffing(y) ~ 0, y(0) ~ a, y′(0) ~ 0], y)) == "a*jacobicn(sqrt(a^2+1)*t,a^2/(2*a^2+2))"
        s = C.desolve([pendulum(y) ~ 0, y(0) ~ 1, y′(0) ~ -1 // 2], y)
        at(e, v) = to_julia(C.evalf(substitute(e, t => v)))
        @test abs(at(s, 0) - 1) < 1e-12 && abs(at(C.diff(s, t), 0) + 0.5) < 1e-12
        @test Giac._residual(pendulum(y) ~ 0, s, t, y) === true
    end
end
