# Symbolic linear algebra on Julia arrays of GiacExpr, and identifier creation

@testset "Array algebra" begin
    @giac_var a11 a12 a21 a22 x1 x2
    A = [a11 a12; a21 a22]
    x = [x1, x2]

    @testset "UniformScaling" begin
        @test string.(A + I) == ["a11+1" "a12"; "a21" "a22+1"]
        @test string.(I + A) == string.(A + I)
        @test string.(A - 2I) == ["a11-2" "a12"; "a21" "a22-2"]
        @test string.(3I - A) == ["-a11+3" "-a12"; "-a21" "-a22+3"]
        @test_throws DimensionMismatch [a11 a12] + I
    end

    @testset "det and inv hand off to Giac" begin
        @test string(det(A)) == "a11*a22-a12*a21"
        Ainv = inv(A)
        @test Ainv isa Matrix{GiacExpr}
        @test string.(Giac.Commands.simplify.(A * Ainv)) == ["1" "0"; "0" "1"]
        sol = A \ x
        @test string.(Giac.Commands.simplify.(A * sol - x)) == ["0", "0"]
    end

    @testset "GiacMatrix ⇄ Matrix" begin
        M = GiacMatrix(A)
        @test M isa GiacMatrix && size(M) == (2, 2)
        @test string.(Matrix(M)) == string.(A)
    end

    @testset "generic Julia operations" begin
        @test string(Giac.Commands.expand(x' * A * x)) ==
              string(Giac.Commands.expand(a11 * x1^2 + (a12 + a21) * x1 * x2 + a22 * x2^2))
        @test A' isa LinearAlgebra.Adjoint
        @test string.(A[:, 1]) == ["a11", "a21"]
    end
end

@testset "symbolic array declarations" begin
    @testset "giac_array" begin
        @test string.(giac_array(:v, 3)) == ["v1", "v2", "v3"]
        @test giac_array(:v, 3) isa Vector{GiacExpr}
        @test string.(giac_array(:m, 2, 2)) == ["m11" "m12"; "m21" "m22"]
        @test string.(giac_array(:s, 2, 2; symmetric = true)) == ["s11" "s12"; "s12" "s22"]
        @test string.(giac_array(:w, 0:1)) == ["w0", "w1"]
        @test string(giac_array(:big, 10)[10]) == "big_10"
        @test_throws ArgumentError giac_array(:s, 2, 3; symmetric = true)
        @test_throws ArgumentError giac_array(:s)
    end

    @testset "@giac_var x[1:n]" begin
        @giac_var y[1:2] c
        @test string.(y) == ["y1", "y2"]
        @test string(y2) == "y2" && string(c) == "c"
        n = 3
        @giac_var z[1:n]                         # non-literal dims: array only
        @test string.(z) == ["z1", "z2", "z3"]
    end

    @testset "@giac_matrix" begin
        @giac_matrix M[2, 2] S[2, 2] symmetric
        @test M isa Matrix{GiacExpr} && string(m21) == "m21"
        @test string.(S) == ["s11" "s12"; "s12" "s22"]
        @test !@isdefined(s21)                   # the shared entry is bound once, as s12
        @giac_var u[1:2]
        @test string(Giac.Commands.expand(u' * S * u)) == "s11*u1^2+s22*u2^2+2*s12*u1*u2"
    end
end

@testset "matrix equations" begin
    @giac_matrix A[2, 2] P[2, 2] symmetric
    eqs = P * A + A' * P ~ -I
    @test size(eqs) == (2, 2)
    N = Giac.Commands.normal
    @test string(N(eqs[1, 2])) == string(N(eqs[2, 1]))      # symmetric unknown: repeated equation
    @test_throws DimensionMismatch A ~ [a11 a12]

    P★ = solve(eqs, P)
    @test P★ isa Matrix{GiacExpr}
    @test string.(Giac.Commands.simplify.(P★ * A + A' * P★)) == ["-1" "0"; "0" "-1"]
    @test string(P★[1, 2]) == string(P★[2, 1])

    # a concrete stable A gives a numeric P
    @giac_var b
    Ab = GiacExpr[giac_eval("-b") giac_eval("-1"); giac_eval("1") giac_eval("0")]
    Pb = solve(P * Ab + Ab' * P ~ -I, P)
    @test string.(Giac.Commands.normal.(Pb)) == ["1/b" "1/2"; "1/2" "(b^2+2)/(2*b)"]

    # no solution
    @test_throws ArgumentError solve(P ~ P + I, P)             # reduces to 0 = 1
end

@testset "express" begin
    @giac_matrix A[2, 2]
    @giac_var τ Δ
    inv_names = (τ => tr(A), Δ => det(A))
    @test string(express(a11^2 + 2a11 * a22 + a22^2, inv_names...)) == "τ^2"
    @test string(express(a11 * a22 - a12 * a21 + 1, inv_names...)) == "Delta+1"
    @test string(express(1 / (a11 + a22), inv_names...)) == "1/τ"
    @test string(express(giac_eval("7"), inv_names...)) == "7"
    @test string(express(3, inv_names...)) == "3"
    @test string(express(a11, )) == "a11"                        # no definitions: unchanged
    @test string.(express.([a11 + a22, a12], inv_names...)) == ["τ", "a12"]
end

@testset "giac_identifier" begin
    @giac_var Δ Σ
    @test string(Δ) == "Delta" && string(Σ) == "Sigma"            # reserved letters, spelled out
    @test occursin("\\Delta", sprint(show, MIME("text/latex"), Δ))
    giac_eval("assigned_elsewhere := [1, 2, 3]")
    @giac_var assigned_elsewhere
    @test string(assigned_elsewhere) == "assigned_elsewhere"     # a variable, not the stored value
    @giac_several_vars assigned_elsewhere 2
    @test string(assigned_elsewhere1) == "assigned_elsewhere1"
    @test string(giac_identifier("u(t)")) == "u(t)"
    giac_eval("purge(assigned_elsewhere)")
end
