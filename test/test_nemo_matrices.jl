# Matrix algebra over rational functions, done in Nemo when it is loaded

using Nemo

@testset "matrix algebra in Nemo" begin
    @giac_var a b c d k s
    C = Giac.Commands
    @test string(det(GiacExpr[a b; c d])) == "a*d-b*c"
    @test string(det(GiacExpr[1 2; 3 4])) == "-2"
    @test string.(inv(GiacExpr[2a 1; 0 b / 3])) == ["1/(2*a)" "-3/(2*a*b)"; "0" "3/b"]
    @test string.(GiacExpr[a b; c d] \ GiacExpr[1, k]) == ["(-b*k+d)/(a*d-b*c)", "(a*k-c)/(a*d-b*c)"]
    @test rank(GiacExpr[a b; 2a 2b]) == 1
    # the Lyapunov equation, solved as a linear system
    @giac_matrix P[2, 2] symmetric
    A = GiacExpr[-k -1; 1 0]
    @test string.(C.solve(P * A + A' * P ~ -I, P)) == ["1/k" "1/2"; "1/2" "(k^2+2)/(2*k)"]
    # eigenvalues of a matrix of rationals: exact, real ones in increasing order
    @test string.(eigvals(GiacExpr[0 1; -2 -3])) == ["-2", "-1"]
    @test string.(eigvals(GiacExpr[0 1; -1 0])) == ["-i", "i"]
    @test string.(eigvals(GiacExpr[0 1 0; 0 0 1; -1 3 0])) ==
          ["rootof([-1,0],[1,0,-3,-1])", "rootof([-1,-3,0],[1,6,9,1])", "rootof([1,0],[1,0,-3,1])"]
    @test length(eigvals(GiacExpr[0 1; -k -1])) == 2                        # symbolic: GIAC
    # entries that are not rational functions go to GIAC
    @test string(det(GiacExpr[sin(a) 1; 0 1])) == "sin(a)"
    @test iszero(C.simplify(det(GiacExpr[sqrt(a) 1; 1 sqrt(a)]) - (a - 1)))
    # the same values as GIAC's, on a larger matrix
    M = [a + i * b - j * c + (i == j ? k * s : 0) for i in 1:4, j in 1:4]
    @test iszero(C.expand(det(M) - det(Giac.GiacMatrix(M))))
    Mi = inv(GiacExpr[a 1 0; 0 b 1; k 0 c])
    @test all(iszero, C.simplify.(Mi * GiacExpr[a 1 0; 0 b 1; k 0 c] - I))
    @test_throws ErrorException inv(GiacExpr[a b; 2a 2b])                  # singular: GIAC says so
end
