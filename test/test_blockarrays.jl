# Block matrices of expressions (GiacBlockArraysExt), and the scalar behaviour they rely on

using BlockArrays

@testset "symbolic scalars in broadcasting and 5-argument mul!" begin
    @giac_var t
    @test string.(t .* [1, 2]) == ["t", "t*2"]
    C = Matrix{GiacExpr}(undef, 2, 2)
    A = GiacExpr[1 2; 3 4]
    mul!(C, A, A, one(GiacExpr), zero(GiacExpr))
    @test string.(C) == string.(A * A)
    mul!(C, A, A, t, one(GiacExpr))                    # C = t A² + C
    @test string.(Giac.Commands.expand.(C - (t .+ 1) .* (A * A))) == ["0" "0"; "0" "0"]
end

@testset "BlockArrays" begin
    @giac_matrix A[2, 2]
    @giac_var b[1:2] c[1:2] d
    M = BlockArray([A I; I A'], [2, 2], [2, 2])
    @test string.(M[Block(1, 2)]) == ["1" "0"; "0" "1"]

    # products of conforming block matrices work and keep the partition
    MM = M * M
    @test MM isa BlockMatrix && blocksize(MM) == (2, 2)
    @test all(iszero, Giac.Commands.expand.(Matrix(MM) - Matrix(M) * Matrix(M)))

    # partition lines at the block boundaries
    tex = Giac.giac_latex(M)
    @test occursin("\\begin{array}{cc|cc}", tex) && count("\\hline", tex) == 1
    K = BlockArray([A b; c' d], [2, 1], [2, 1])
    @test occursin("{cc|c}", Giac.giac_latex(K))
    @test occursin("\\hline", Giac.giac_latex(BlockArray([b; d], [2, 1])))
    @test startswith(sprint(show, MIME("text/latex"), M), "\$\$\\left[")
end
