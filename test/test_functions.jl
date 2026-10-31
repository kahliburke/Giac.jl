# Symbolic functions (@giac_function)

@testset "@giac_function" begin
    S = string
    @giac_var y p

    @testset "calls" begin
        @giac_function f(x) = x^2 * sin(x)
        @test f isa Giac.GiacFunction
        @test S(f(y)) == "y^2*sin(y)"
        @test S(f(2)) == "4*sin(2)"                 # exact numbers stay exact
        @test S(f(π)) == "0"
        @test f(0.5) isa Float64                     # floats run the compiled version
        @test f(0.5) ≈ 0.25 * sin(0.5)
        @test f.([0.1, 0.2]) ≈ [0.01 * sin(0.1), 0.04 * sin(0.2)]
        @test_throws ArgumentError f(1, 2)
    end

    @testset "arguments are local to the definition" begin
        @giac_function g(fresh_arg) = fresh_arg + 1
        @test !@isdefined(fresh_arg)
        @giac_function h(wvec[1:2]) = wvec1 * wvec2
        @test !@isdefined(wvec1)
        @test S(h([y, 3])) == "y*3"
    end

    @testset "vector arguments and bodies" begin
        @giac_function F(q[1:2], u) = [q2, -sin(q1) + u]
        @test F([0.1, 0.0], 0.5) ≈ [0.0, -sin(0.1) + 0.5]
        @test S.(F([y, 0], p)) == ["0", "-sin(y)+p"]
        J = jacobian(F)
        @test J isa Giac.GiacFunction
        @test S.(J.body) == ["0" "1" "0"; "-cos(q1)" "0" "1"]
        @test J([0.0, 0.0], 0.0) ≈ [0.0 1.0 0.0; -1.0 0.0 1.0]
    end

    @testset "derivatives" begin
        @giac_function f(x) = x^3
        @test S(f'.body) == "3*x^2"
        @test S(f''.body) == "6*x"
        @test f'(2.0) ≈ 12.0
        @test S(f'(y)) == "3*y^2"
        @giac_matrix P[2, 2] symmetric
        @giac_function V(z[1:2]) = z' * P * z
        @test S.(∇(V).body) == ["2*p11*z1+2*p12*z2", "2*p12*z1+2*p22*z2"]
        @test S.(hessian(V).body) == ["2*p11" "2*p12"; "2*p12" "2*p22"]
        @test_throws ArgumentError V'
    end

    @testset "a body with parameters stays symbolic" begin
        @giac_var m g
        @giac_function E(h) = m * g * h
        @test E(2.0) isa GiacExpr
        @test S(E(2)) == "m*g*2"
    end

    @testset "display" begin
        @giac_function f(x) = x^2 + 1
        @test S(f) == "f(x) = x^2+1"
        @test sprint(show, MIME"text/latex"(), f'') == "\$\$f''\\left(x\\right) = 2\$\$"
        @giac_function F(q[1:2]) = [q2, q1]
        @test S(F) == "F(q) = [q2, q1]"
        @test occursin("\\nabla", sprint(show, MIME"text/latex"(), ∇(@giac_function G(x, y) = x * y)))
    end
end
