# SciML problems from symbolic models (GiacSciMLBaseExt, with GiacSparseArraysExt and GiacStaticArraysExt)

using SciMLBase, SparseArrays, StaticArrays, OrdinaryDiffEqRosenbrock, SimpleNonlinearSolve
using SciMLBase.SymbolicIndexingInterface: variable_symbols

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
        local out, u, p                 # not the outer `u, p`
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

        # values written in the parameters, computed in Julia from their exact expressions
        w = ODEProblem(F, [x1 => π / 4 + c, x2 => sqrt(g / ℓ)], (0, 1), [g => 9.81, ℓ => 2, c => 0.1])
        @test w.u0 == [π / 4 + 0.1, sqrt(9.81 / 2)]
        @test_throws "uses x2: a value may be written only in the parameters" ODEProblem(F, [x1 => c + x2, x2 => 0], (0, 1), ps)

        # states in the order of the assignment
        r = ODEProblem(F, [x2 => 0, x1 => 1], (0, 1), ps)
        @test string.(variable_symbols(r)) == ["x2", "x1"] && r.u0 == [0, 1]
    end

    @testset "functions from @giac_function" begin
        @giac_function P(q[1:2], k[1:2]) = [q2, -k1 * sin(q1) - k2 * q2]
        prob = ODEProblem(P, [π / 2, 0.0], (0, 10), [9.81, 0.2])
        ref = solve(ODEProblem(F, u0, (0, 10), [g => 9.81, ℓ => 1, c => 0.2]), Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        sol = solve(prob, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        @test sol(10.0) ≈ ref(10.0) atol = 1e-7
        @test string.(variable_symbols(prob)) == ["q1", "q2"]
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

    @testset "equations in unknown functions of time" begin
        local t, q, w, c, k
        @giac_var t q(t) w(t) c k
        D = Differential(t)
        eqs = [D(D(q)) + c * D(q) + k * sin(q) ~ 0]
        prob = ODEProblem(eqs, [q => π / 2, D(q) => 0], (0, 10), [c => 0.2, k => 9.81])
        @test string.(variable_symbols(prob)) == ["q", "q′"]
        sol = solve(prob, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        ref = solve(ODEProblem(F, u0, (0, 10), ps), Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        @test sol(10.0) ≈ ref(10.0) atol = 1e-7
        @test sol[q] == sol[1, :] && sol[D(q)] == sol[2, :]
        # the highest derivative is observed, from the solved equation
        @test sol(3.0; idxs = D(D(q))) ≈ -0.2 * sol(3.0; idxs = D(q)) - 9.81 * sin(sol(3.0; idxs = q))
        @test remake(prob; u0 = [D(q) => 1.0]).u0 == [π / 2, 1.0]

        # coupled, of different orders: the states in order of appearance
        two = ODEProblem([D(D(q)) ~ -q + w, D(w) ~ -w], [w => 1, q => 1, D(q) => 0], (0, 1))
        @test string.(variable_symbols(two)) == ["q", "q′", "w"]
        @test two.u0 == [1, 0, 1]
        du = zeros(3)
        two.f(du, [1.0, 2.0, 3.0], two.p, 0.0)
        @test du == [2, 2, -3]

        @test_throws ArgumentError ODEProblem([D(D(q))^2 ~ q], [q => 1, D(q) => 0], (0, 1))   # not linear in q″
        @test_throws ArgumentError ODEProblem(eqs, [q => 1], (0, 1), [c => 0.2, k => 1])    # no value for q′
    end

    @testset "algebraic equations: a DAE with a mass matrix" begin
        local t, y, z
        @giac_var t y(t) z(t)
        D = Differential(t)
        # ẏ = -y + z with z = y², so y = 1 / (1 + eᵗ) from y(0) = 1/2
        prob = ODEProblem([D(y) ~ -y + z, z ~ y^2], [y => 0.5, z => 0.25], (0, 2))
        @test Matrix(prob.f.mass_matrix) == [1 0; 0 0]
        sol = solve(prob, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
        @test sol(2.0; idxs = y) ≈ 1 / (1 + exp(2)) atol = 1e-7
        @test sol(2.0; idxs = z) ≈ sol(2.0; idxs = y)^2 atol = 1e-9
    end

    @testset "inputs" begin
        local u, k1, k2
        @giac_var u k1 k2
        G = [x2, -x1 + u]
        # a feedback law, substituted: part of the exact Jacobian
        fb = ODEFunction(G, x; p = [k1, k2], inputs = [u => -k1 * x1 - k2 * x2])
        Jfb = zeros(2, 2)
        fb.jac(Jfb, [0.0, 0.0], [2.0, 3.0], 0.0)
        @test Jfb == [0 1; -3 -3]
        # a Julia function of time
        ft = ODEFunction(G, x; inputs = [u => τ -> 2τ])
        d = zeros(2)
        ft(d, [1.0, 0.5], nothing, 3.0)
        @test d == [0.5, 5.0]
        @test ft.jac !== nothing
        # a Julia function of the state: the solver differentiates the right-hand side
        fs = ODEFunction(G, x; inputs = [u => (x, p, τ) -> -x[1]^3])
        fs(d, [2.0, 0.5], nothing, 0.0)
        @test d == [0.5, -10.0]
        @test fs.jac === nothing
        # solved, with the input observed
        prob = ODEProblem(G, [x1 => 1, x2 => 0], (0, 1); inputs = [u => τ -> sin(τ)])
        sol = solve(prob, Rodas5P())
        @test SciMLBase.successful_retcode(sol)
        @test sol(0.5; idxs = u) ≈ sin(0.5)
    end

    @testset "events" begin
        local t, q, r, k
        @giac_var r k
        tb, tol = sqrt(2 / 9.81), (; abstol = 1e-10, reltol = 1e-10)
        # a bouncing ball: the event when the inequality becomes true, the velocity from before it
        ball = ODEProblem([x2, -g], [x1 => 1, x2 => 0], (0, 3), [g => 9.81, r => 0.8];
                          events = [x1 < 0 => [x2 => -r * x2]])
        sol = solve(ball, Rodas5P(); tol...)
        @test sol(tb + 2 * 0.8 * tb; idxs = x1) ≈ 0 atol = 1e-8
        @test sol(tb + 0.8 * tb; idxs = x1) ≈ 0.64
        cond = only(ball.kwargs[:callback].continuous_callbacks).condition
        ub, integ = [0.5, 1.0], (; p = ball.p)
        cond(ub, 0.0, integ)
        @test (@allocated cond(ub, 0.0, integ)) == 0

        # a parameter switched at an equation's crossing; the problem keeps its values
        sw = ODEProblem([-k * x1], [x1 => 1], (0, 2), [k => 1]; events = [(x1 ~ 1 // 2) => [k => 5]])
        s = solve(sw, Rodas5P(); tol...)
        @test s(2.0)[1] ≈ 0.5 * exp(-5 * (2 - log(2)))
        @test sw.p == [1]

        # a Julia function of the integrator, alongside a callback of the problem's own
        seen = Ref(0)
        own = SciMLBase.DiscreteCallback((u, t, i) -> true, i -> (seen[] += 1; SciMLBase.u_modified!(i, false)))
        st = solve(ODEProblem([x2, -g], [x1 => 1, x2 => 0], (0, 3), [g => 9.81];
                              events = [x1 < 0 => SciMLBase.terminate!], callback = own), Rodas5P())
        @test st.t[end] ≈ tb && seen[] > 0

        # in an equation model, by the unknown functions and their derivatives
        @giac_var t q(t)
        D = Differential(t)
        eqs = [D(D(q)) ~ -9.81]
        bq = solve(ODEProblem(eqs, [q => 1, D(q) => 0], (0, 3), [r => 0.8];
                              events = [q < 0 => [D(q) => -r * D(q)]]), Rodas5P(); tol...)
        @test bq(tb + 2 * 0.8 * tb; idxs = q) ≈ 0 atol = 1e-8
        top = solve(ODEProblem(eqs, [q => 0, D(q) => 3], (0, 3); events = [(D(q) ~ 0) => SciMLBase.terminate!]), Rodas5P())
        @test top.t[end] ≈ 3 / 9.81

        @test_throws ArgumentError ODEProblem([x2, -g], [x1 => 1, x2 => 0], (0, 3), [g => 9.81];
                                              events = [x1 < 0 => [g * x1 => 0]])
    end

    @testset "stochastic models" begin
        local t, q, θ, μ, σ
        @giac_var θ μ σ
        # diagonal noise: one per state
        ou = SDEProblem([θ * (μ - x1)], [σ * x1], [x1 => 1], (0, 10), [θ => 2, μ => 0, σ => 0.3])
        d, u1 = zeros(1), [2.0]
        ou.f(d, u1, ou.p, 0.0)
        @test d == [-4.0]
        ou.g(d, u1, ou.p, 0.0)
        @test d ≈ [0.6]
        @test ou.f.jac !== nothing && string.(variable_symbols(ou)) == ["x1"]
        @test (@allocated ou.g(d, u1, ou.p, 0.0)) == 0

        # a matrix: a column for each noise process, and the problem's noise prototype
        m = SDEProblem([x2, -x1], [0 0; σ σ * x1], [x1 => 1, x2 => 0], (0, 1), [σ => 0.5])
        G = fill(NaN, 2, 2)
        m.g(G, [2.0, 1.0], m.p, 0.0)
        @test G == [0 0; 0.5 1.0] && size(m.noise_rate_prototype) == (2, 2)
        @test SDEProblem([x2, -x1], [0 0; σ σ * x1], [x1 => 1, x2 => 0], (0, 1), [σ => 0.5]; inplace = false).g([2.0, 1.0], [0.5], 0.0) == G

        # equations, with the noise in the unknown function's derivative
        @giac_var t q(t)
        D = Differential(t)
        e = SDEProblem([D(D(q)) ~ -q], [0, σ * D(q)], [q => 1, D(q) => 2], (0, 1), [σ => 0.5])
        d2 = zeros(2)
        e.g(d2, [1.0, 2.0], e.p, 0.0)
        @test d2 == [0, 1.0]
        @test_throws DimensionMismatch SDEProblem([x2, -x1], [σ], [x1 => 1, x2 => 0], (0, 1), [σ => 0.5])
    end

    @testset "static arrays" begin
        local r, k
        @giac_var r k
        tol = (; abstol = 1e-10, reltol = 1e-10)
        sp = ODEProblem(F, u0, (0, 10), ps; static = true)
        @test sp.u0 isa SVector{2, Float64} && sp.p isa SVector{3, Float64} && !SciMLBase.isinplace(sp)
        us, pst = SVector(0.1, 0.2), SVector(9.81, 1.0, 0.2)
        @test @inferred(sp.f(us, pst, 0.0)) isa SVector{2, Float64}
        @test sp.f(us, pst, 0.0) ≈ rhs(u, p)
        @test sp.f.jac(us, pst, 0.0) isa SMatrix{2, 2, Float64}
        @test sp.f.paramjac(us, pst, 0.0) isa SMatrix{2, 3, Float64}
        @test (@allocated sp.f(us, pst, 0.0)) == 0
        ref = solve(ODEProblem(F, u0, (0, 10), ps), Rodas5P(); tol...)
        @test solve(sp, Rodas5P(); tol...)(10.0) ≈ ref(10.0) atol = 1e-7

        # events replace an immutable state or parameter vector
        tb = sqrt(2 / 9.81)
        ball = solve(ODEProblem([x2, -g], [x1 => 1, x2 => 0], (0, 3), [g => 9.81, r => 0.8]; static = true,
                                events = [x1 < 0 => [x2 => -r * x2]]), Rodas5P(); tol...)
        @test ball(tb + 2 * 0.8 * tb)[1] ≈ 0 atol = 1e-8
        sw = ODEProblem([-k * x1], [x1 => 1], (0, 2), [k => 1]; static = true, events = [(x1 ~ 1 // 2) => [k => 5]])
        @test solve(sw, Rodas5P(); tol...)(2.0)[1] ≈ 0.5 * exp(-5 * (2 - log(2)))
        @test sw.p == [1]

        @test NonlinearProblem(F, [x1 => 3.0, x2 => 0.1], ps; static = true).u0 isa SVector{2}
    end

    @testset "Jacobian-vector products and exact solutions" begin
        local k, cc, ff
        @giac_var k cc ff
        f = ODEFunction(F, x; p = [g, ℓ, c], jvp = true, vjp = true)
        uu, pp, vv = [0.3, -0.2], [9.81, 1.5, 0.2], [0.7, -1.1]
        Jm, out = zeros(2, 2), zeros(2)
        f.jac(Jm, uu, pp, 0.0)
        f.jvp(out, vv, uu, pp, 0.0)
        @test out ≈ Jm * vv
        f.vjp(out, vv, uu, pp, 0.0)
        @test out ≈ transpose(Jm) * vv
        @test (@allocated f.jvp(out, vv, uu, pp, 0.0)) == 0
        @test ODEFunction(F, x; p = [g, ℓ, c], jvp = true, inplace = false).jvp(vv, uu, pp, 0.0) ≈ Jm * vv

        # a model affine in its states: the solution reports its errors against the exact one
        lin = [x2, -k * x1 - cc * x2 + ff]
        pr = ODEProblem(lin, [x1 => 1, x2 => 0], (0, 10), [k => 4, cc => 0.4, ff => 1]; analytic = true)
        @test solve(pr, Rodas5P(); abstol = 1e-10, reltol = 1e-10).errors[:final] < 1e-8
        crit = ODEProblem(lin, [x1 => 1, x2 => 0], (0, 10), [k => 1, cc => 2, ff => 0]; analytic = true)
        @test crit.f.analytic(crit.u0, crit.p, 3.0)[1] ≈ 4exp(-3)            # a repeated eigenvalue
        @test_throws "affine in its states" ODEProblem(F, u0, (0, 1), ps; analytic = true)
    end

    @testset "errors" begin
        @giac_var k
        @test_throws ArgumentError ODEFunction([x2, -k * x1], x)              # k: not a parameter
        @test_throws DimensionMismatch ODEFunction([x2], x)
        @test_throws ArgumentError ODEProblem(F, x => [1, 0], (0, 1), [g => 1])
        @test_throws "solves a SciML problem, and x is a Float64" solve(0.5, Rodas5P())
        @test_throws "solves a SciML problem" solve([1.0, 2.0], Rodas5P())
    end
end
