# Kronecker products and vectorization of linear matrix equations

@testset "Kronecker products" begin
    @matsym A[n, n] B[m, m] C[n, n] D[m, m] E[n, n] F[m, m] P[n, n]::symmetric X[n, m] S[n, n]::skew
    T(M) = sprint(show, M)

    @testset "the algebra" begin
        @test size(kron(A, B)) == (Symbol("m*n"), Symbol("m*n"))
        @test T(kron(A, B) * kron(C, D)) == "kron(A*C, B*D)"               # the mixed product
        @test T(kron(A, B)') == "kron(A', B')"
        @test T(inv(kron(A, B))) == "kron(inv(A), inv(B))"
        @test T(inv(kron(A, B)) * kron(A, B)) == "I"
        @test T(kron(A + C, 2B)) == "2*kron(A, B) + 2*kron(C, B)"          # bilinear
        @test T(kron(S, A)') == "-kron(S, A')"
        @test T(tr(kron(A, B))) == "tr(A)*tr(B)"
        @test T(det(kron(A, B))) == "det(A)^m*det(B)^n"
        @test T(differential(kron(A, B))) == "kron(A, dB) + kron(dA, B)"
    end

    @testset "sizes that are products" begin
        @matsym Y[m*n, m*n]
        @test size(Y) == size(kron(A, B)) && size(Y * kron(B, A)) == size(Y)
        @test size(vec(X)) == (Symbol("m*n"), 1)
        @test_throws DimensionMismatch Y * A
    end

    @testset "vectorizing linear matrix equations" begin
        K, R = vectorize(A' * P + P * A, P)
        @test T(K) == "kron(A', I) + kron(I, A')" && iszero(R)
        @test T(first(vectorize(E * X + X * F, X))) == "kron(F', I) + kron(I, E)"
        K2, R2 = vectorize(A * X * B + C * C * X, X)
        @test T(K2) == "kron(B', A) + kron(I, C^2)" && iszero(R2)
        @test T(last(vectorize(A * P + C, P))) == "C"
        @test_throws ArgumentError vectorize(P * A * P, P)
        @test_throws ArgumentError vectorize(A * E' * A, E)                  # a transpose of a general E
        @test Giac.giac_latex(K) == "A^{\\top} \\otimes I + I \\otimes A^{\\top}"
    end

    @testset "explicit: the Lyapunov equation as a linear system" begin
        K, _ = vectorize(A' * P + P * A, P)
        A₂ = GiacExpr[0 1; -2 -3]
        K₂ = substitute(K, A => A₂)
        @test size(K₂) == (4, 4)
        P₂ = reshape(Giac.Commands.simplify.(K₂ \ (-vec(GiacExpr[1 0; 0 1]))), 2, 2)
        @test string.(P₂) == ["5/4" "1/4"; "1/4" "1/4"]
        @test P₂ * A₂ + A₂' * P₂ == -GiacExpr[1 0; 0 1]
        @test string(substitute(tr(kron(A, B)), A => A₂, B => GiacExpr[1 2; 3 4])) == "-15"
    end
end
