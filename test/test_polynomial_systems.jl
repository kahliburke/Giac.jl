# polynomial systems solved exactly by msolve (GiacAlgebraicSolvingExt)

using AlgebraicSolving

@testset "polynomial systems" begin
    @giac_var x y z t
    num(e) = Float64(Giac.to_julia(Giac.Commands.evalf(e)))
    sys = [x^2 + y^2 - 1, y - x^2]

    @testset "real solutions" begin
        sols = real_solutions(sys)
        @test length(sols) == 2
        for s in sols
            @test all(s.lower .<= s.upper)
            @test s.lower[1] <= s[x] <= s.upper[1]
            @test all(s.upper .- s.lower .< big(2)^-60)
            @test Float64(s[y]) ≈ (sqrt(5) - 1) / 2
            @test abs(num(substitute(x^2 + y^2, s...)) - 1) < 1e-15
        end
        @test sort(Float64.(getindex.(sols, x))) ≈ [-1, 1] .* sqrt((sqrt(5) - 1) / 2)
        @test isempty(real_solutions([x^2 + 1]))
        @test length(real_solutions([x ~ 2, y^2 ~ x])) == 2
        @test all(s -> s.upper[1] - s.lower[1] < big(2)^-200, real_solutions([x^2 - 2]; precision = 200))
        @test string(only(real_solutions([x - 1, y - 2], [y, x]))) == "(y ≈ 2.0, x ≈ 1.0)"
    end

    @testset "rational solutions" begin
        sols = rational_solutions([x^2 - 1, y - x])
        @test Set([string.(last.(s)) for s in sols]) == Set([["1", "1"], ["-1", "-1"]])
        @test isempty(rational_solutions([x^2 - 2, y]))
    end

    @testset "rational parametrization" begin
        # every real root of the eliminant gives a real solution, and the parameter is the form
        function consistent(p, sys, vars)
            sols = real_solutions(sys, vars)
            roots = real_solutions([p.eliminant], [p.t])
            length(roots) == length(sols) || return false
            return all(roots) do r
                τ = r[p.t]
                pt = [num(substitute(last(v), p.t => τ)) for v in p.values]
                any(s -> maximum(abs.(pt .- [Float64(s[v]) for v in vars])) < 1e-9, sols) &&
                    abs(num(substitute(p.form, (vars .=> pt)...)) - Float64(τ)) < 1e-9
            end
        end
        p = rational_parametrization(sys)
        @test p.degree == 4
        @test consistent(p, sys, [x, y])
        cyc = [x^2 + y^2 + z^2 - 3, x * y * z - 1, x + y - z]
        @test consistent(rational_parametrization(cyc), cyc, [x, y, z])
        # the last variable separates the solutions: it is the parameter
        tri = [x + 2y + 2z - 1, x^2 + 2y^2 + 2z^2 - x, 2x * y + 2y * z - y]
        q = rational_parametrization(tri; t = t)
        @test string(q.form) == "z" && string(last(q.values[3])) == "t"
        @test consistent(q, tri, [x, y, z])
        @test occursin("4 solutions", sprint(show, MIME"text/plain"(), q))
        @test_throws ArgumentError rational_parametrization([x - 1], [x]; t = x)
    end

    @testset "Gröbner bases, elimination, normal forms" begin
        @test string.(groebner_basis(sys)) == ["y^2+y-1", "x^2-y"]
        @test string.(eliminate([x - (1 - t^2), y - 2t], t)) == ["4*x+y^2-4"]
        @test string(normal_form(x^4 + 2x^2 * y^2 + y^4, [x^2 + y^2 - 1])) == "1"
        @test string(normal_form(x^3 * y + x * y^3 - x * y + 7, [x^2 + y^2 - 1])) == "7"
        @test dimension([x^2 + y^2 - 1]) == 1
        @test dimension(sys) == 0
        @test dimension([x^2 + 1, x]) == -1
    end

    @testset "errors" begin
        @test_throws ArgumentError real_solutions([x^2 + y^2 - 1])          # infinitely many
        @test_throws ArgumentError real_solutions([x / y - 1, y - 2])        # not a polynomial
        @test_throws ArgumentError real_solutions([x^2 - 2.5])               # not rational
        @test_throws ArgumentError real_solutions([x^2 - 1, y - z], [x, y])  # z not a variable
        @test_throws ArgumentError real_solutions([x - 1], [x, x])
    end
end
