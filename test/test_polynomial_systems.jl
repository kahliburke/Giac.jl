# polynomial systems solved exactly by msolve (GiacAlgebraicSolvingExt)

import AlgebraicSolving

@testset "polynomial systems" begin
    @giac_var x y z t
    num(e) = Float64(Giac.to_julia(Giac.Commands.evalf(e)))
    sys = [x^2 + y^2 - 1, y - x^2]

    @testset "real solutions" begin
        sols = AlgebraicSolving.real_solutions(sys)
        @test length(sols) == 2
        for s in sols
            @test all(s.lower .<= s.upper)
            @test s.lower[1] <= s[x] <= s.upper[1]
            @test all(s.upper .- s.lower .< big(2)^-60)
            @test Float64(s[y]) ≈ (sqrt(5) - 1) / 2
            @test abs(num(Giac.substitute(x^2 + y^2, s...)) - 1) < 1e-15
        end
        @test sort(Float64.(getindex.(sols, x))) ≈ [-1, 1] .* sqrt((sqrt(5) - 1) / 2)
        @test isempty(AlgebraicSolving.real_solutions([x^2 + 1]))
        @test length(AlgebraicSolving.real_solutions([x ~ 2, y^2 ~ x])) == 2
        @test all(s -> s.upper[1] - s.lower[1] < big(2)^-200, AlgebraicSolving.real_solutions([x^2 - 2]; precision = 200))
        @test string(only(AlgebraicSolving.real_solutions([x - 1, y - 2], [y, x]))) == "(y ≈ 2.0, x ≈ 1.0)"
    end

    @testset "rational solutions" begin
        sols = AlgebraicSolving.rational_solutions([x^2 - 1, y - x])
        @test Set([string.(last.(s)) for s in sols]) == Set([["1", "1"], ["-1", "-1"]])
        @test isempty(AlgebraicSolving.rational_solutions([x^2 - 2, y]))
    end

    @testset "rational parametrization" begin
        # every real root of the eliminant gives a real solution, and the parameter is the form
        function consistent(p, sys, vars)
            sols = AlgebraicSolving.real_solutions(sys, vars)
            roots = AlgebraicSolving.real_solutions([p.eliminant], [p.t])
            length(roots) == length(sols) || return false
            return all(roots) do r
                τ = r[p.t]
                pt = [num(Giac.substitute(last(v), p.t => τ)) for v in p.values]
                any(s -> maximum(abs.(pt .- [Float64(s[v]) for v in vars])) < 1e-9, sols) &&
                    abs(num(Giac.substitute(p.form, (vars .=> pt)...)) - Float64(τ)) < 1e-9
            end
        end
        p = AlgebraicSolving.rational_parametrization(sys)
        @test p.degree == 4
        @test consistent(p, sys, [x, y])
        cyc = [x^2 + y^2 + z^2 - 3, x * y * z - 1, x + y - z]
        @test consistent(AlgebraicSolving.rational_parametrization(cyc), cyc, [x, y, z])
        # the last variable separates the solutions: it is the parameter
        tri = [x + 2y + 2z - 1, x^2 + 2y^2 + 2z^2 - x, 2x * y + 2y * z - y]
        q = AlgebraicSolving.rational_parametrization(tri; t = t)
        @test string(q.form) == "z" && string(last(q.values[3])) == "t"
        @test consistent(q, tri, [x, y, z])
        @test occursin("4 solutions", sprint(show, MIME"text/plain"(), q))
        @test_throws ArgumentError AlgebraicSolving.rational_parametrization([x - 1], [x]; t = x)
    end

    @testset "Gröbner bases, elimination, normal forms" begin
        @test string.(AlgebraicSolving.groebner_basis(sys)) == ["y^2+y-1", "x^2-y"]
        @test string.(AlgebraicSolving.eliminate([x - (1 - t^2), y - 2t], t)) == ["4*x+y^2-4"]
        @test string(AlgebraicSolving.normal_form(x^4 + 2x^2 * y^2 + y^4, [x^2 + y^2 - 1])) == "1"
        @test string(AlgebraicSolving.normal_form(x^3 * y + x * y^3 - x * y + 7, [x^2 + y^2 - 1])) == "7"
        @test AlgebraicSolving.dimension([x^2 + y^2 - 1]) == 1
        @test AlgebraicSolving.dimension(sys) == 0
        @test AlgebraicSolving.dimension([x^2 + 1, x]) == -1
    end

    # real_root_classification needs AlgebraicSolving 0.13 (an environment with Oscar has 0.10)
    isdefined(AlgebraicSolving, :real_root_classification) && @testset "real root classification" begin
        @giac_var a b
        # the cusp: three equilibria inside 4a³ = 27b², one outside
        rc = AlgebraicSolving.real_root_classification([x^3 - a * x + b], [x]; params = [a, b])
        @test string.(rc.params) == ["a", "b"]
        @test string.(rc.excluded) == ["4*a^3-27*b^2"]
        @test sort([c.count for c in rc.cases]) == [1, 1, 3]
        for c in rc.cases
            # the count at the witness point agrees with solving there
            at = Giac.substitute(x^3 - a * x + b, c.witness...)
            @test length(AlgebraicSolving.real_solutions([at], [x])) == c.count
        end
        @test occursin("3 real solutions", sprint(show, MIME"text/plain"(), rc))
        # only the positive solutions
        rp = AlgebraicSolving.real_root_classification([x^2 - a], [x]; params = [a], positive = [x])
        @test Set((c.count, Giac.to_julia(last(only(c.witness))) > 0) for c in rp.cases) == Set([(0, false), (1, true)])
    end

    @testset "equilibria and their stability" begin
        eqs = Giac.Control.equilibria([y, x - x^3 - y], [x, y])
        @test length(eqs) == 3
        byx = Dict(round(Int, Float64(e.point[x])) => e.unstable for e in eqs)
        @test byx == Dict(-1 => 0, 0 => 1, 1 => 0)
        @test count(Giac.Control.stable, eqs) == 2
        @test occursin("unstable, 1 eigenvalue", sprint(show, eqs[findfirst(e -> e.unstable == 1, eqs)]))
        # a center: the linearization does not decide
        @test only(Giac.Control.equilibria([y, -x], [x, y])).unstable === nothing
        # Lorenz above the Hopf bifurcation: a saddle at the origin, two eigenvalues to the right
        # at the other two
        lz = Giac.Control.equilibria([10 * (y - x), x * (28 - z) - y, x * y - 8 // 3 * z], [x, y, z])
        @test sort([e.unstable for e in lz]) == [1, 2, 2]
    end

    @testset "errors" begin
        @test_throws ArgumentError AlgebraicSolving.real_solutions([x^2 + y^2 - 1])          # infinitely many
        @test_throws ArgumentError AlgebraicSolving.real_solutions([x / y - 1, y - 2])        # not a polynomial
        @test_throws ArgumentError AlgebraicSolving.real_solutions([x^2 - 2.5])               # not rational
        @test_throws ArgumentError AlgebraicSolving.real_solutions([x^2 - 1, y - z], [x, y])  # z not a variable
        @test_throws ArgumentError AlgebraicSolving.real_solutions([x - 1], [x, x])
    end
end
