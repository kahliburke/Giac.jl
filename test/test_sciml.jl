# SciML problems from symbolic models (GiacSciMLBaseExt, with GiacSparseArraysExt)

using SciMLBase, SparseArrays, OrdinaryDiffEqRosenbrock, SimpleNonlinearSolve

@testset "SciML problems from Giac expressions" begin
    @giac_var x[1:2] g ℓ c t A ω
    F = [x2, -g / ℓ * sin(x1) - c * x2]
    u0, ps = [x1 => π / 2, x2 => 0], [g => 9.81, ℓ => 1, c => 0.2]
    u, p = [0.1, 0.2], [9.81, 1.0, 0.2]
    rhs(u, p) = [u[2], -p[1] / p[2] * sin(u[1]) - p[3] * u[2]]

    @testset "the compiled functions" begin
        f = ODEFunction(F, x; p = [g, ℓ, c])
        du = zeros(2)
        f(du, u, p, 0.0)
        @test du ≈ rhs(u, p)
        J = fill(NaN, 2, 2)
        f.jac(J, u, p, 0.0)
        @test J ≈ [0 1; -p[1] / p[2] * cos(u[1]) -p[3]]
        Jp = zeros(2, 3)
        f.paramjac(Jp, u, p, 0.0)
        @test Jp ≈ [0 0 0; -sin(u[1]) / p[2] p[1] / p[2]^2 * sin(u[1]) -u[2]]
        @test f.tgrad === nothing                                   # no time variable
        f(du, u, p, 0.0)
        @test (@allocated f(du, u, p, 0.0)) == 0
        @test (@allocated f.jac(J, u, p, 0.0)) == 0

        fo = ODEFunction(F, x; p = [g, ℓ, c], inplace = false)
        @test !SciMLBase.isinplace(fo)
        @test fo(u, p, 0.0) ≈ rhs(u, p)
        @test fo.jac(u, p, 0.0) ≈ J

        # forcing in time: the exact time gradient
        ff = ODEFunction([x2, -sin(x1) + A * cos(ω * t)], x; p = [A, ω], t)
        tg = zeros(2)
        ff.tgrad(tg, u, [0.5, 1.2], 1.0)
        @test tg ≈ [0, -0.5 * 1.2 * sin(1.2)]

        # a sparse Jacobian has the symbolic one's structure
        fs = ODEFunction(F, x; p = [g, ℓ, c], sparse = true)
        @test fs.jac_prototype isa SparseMatrixCSC
        @test findnz(fs.jac_prototype)[1:2] == ([2, 1, 2], [1, 2, 2])
        Js = copy(fs.jac_prototype)
        fs.jac(Js, u, p, 0.0)
        @test Matrix(Js) ≈ J

    end

    @testset "variables named like the functions' arguments" begin
        @giac_var out u p
        fn = ODEFunction([p, -u * out, u], [u, p, out])
        d = zeros(3)
        fn(d, [1.0, 2.0, 3.0], nothing, 0.0)
        @test d == [2, -3, 1]
    end

    @testset "problems, solutions and symbolic indexing" begin
        prob = ODEProblem(F, u0, (0, 10), ps)
        @test prob.u0 == [π / 2, 0] && prob.p == [9.81, 1, 0.2] && prob.tspan == (0.0, 10.0)
        # exact values become floating-point numbers
        @test ODEProblem(F, [x1 => 1, x2 => 0], (0, 1), [g => 981//100, ℓ => 1, c => 1//5]).p == [9.81, 1, 0.2]
        sol = solve(prob, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.stats.njacs > 0
        @test sol[x1] == sol[1, :]
        @test sol[[x1, x2]] == sol.u
        E = x1^2 + x2^2
        @test sol[E] ≈ [v[1]^2 + v[2]^2 for v in sol.u]
        @test prob.ps[g] == 9.81

        # the same solution out of place and with a sparse Jacobian
        for kw in ((; inplace = false), (; sparse = true))
            s = solve(ODEProblem(F, u0, (0, 10), ps; kw...), Rodas5P(); abstol = 1e-10, reltol = 1e-10)
            @test s(10.0) ≈ sol(10.0) atol = 1e-7
        end

        # values by name
        q = remake(prob; p = [g => 3.7], u0 = [x1 => 0.3])
        @test q.p == [3.7, 1, 0.2] && q.u0 == [0.3, 0]

        # states in the order of the assignment
        r = ODEProblem(F, [x2 => 0, x1 => 1], (0, 1), ps)
        @test r.f.sys.states == [x2, x1] && r.u0 == [0, 1]
    end

    @testset "functions from @giac_function" begin
        @giac_function P(q[1:2], k[1:2]) = [q2, -k1 * sin(q1) - k2 * q2]
        prob = ODEProblem(P, [π / 2, 0.0], (0, 10), [9.81, 0.2])
        ref = solve(ODEProblem(F, u0, (0, 10), [g => 9.81, ℓ => 1, c => 0.2]), Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        sol = solve(prob, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        @test sol(10.0) ≈ ref(10.0) atol = 1e-7
        @test string.(prob.f.sys.states) == ["q1", "q2"]
        @test_throws ArgumentError ODEProblem(P, [π / 2, 0.0], (0, 10))
    end

    @testset "nonlinear problems" begin
        prob = NonlinearProblem(F, [x1 => 3.0, x2 => 0.1], ps)
        sol = solve(prob, SimpleNewtonRaphson())
        @test sol[x1] ≈ π && abs(sol[x2]) < 1e-12
        # an equation stands for the difference of its sides
        e = NonlinearProblem([x1^2 ~ 2, x2 ~ x1], [x1 => 1.0, x2 => 0.0])
        @test solve(e, SimpleNewtonRaphson()).u ≈ [√2, √2]
    end

    @testset "errors" begin
        @giac_var k
        @test_throws ArgumentError ODEFunction([x2, -k * x1], x)              # k: not a parameter
        @test_throws DimensionMismatch ODEFunction([x2], x)
        @test_throws ArgumentError ODEProblem(F, x => [1, 0], (0, 1), [g => 1])
    end
end
