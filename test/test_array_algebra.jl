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

    @testset "block matrices" begin
        @giac_var b1 b2 d
        Z = zeros(GiacExpr, 2, 2)
        @test string.([A I; I A']) == ["a11" "a12" "1" "0"; "a21" "a22" "0" "1"; "1" "0" "a11" "a21"; "0" "1" "a12" "a22"]
        @test string.([A Z; Z A])[1, 3] == "0"
        @test size([A [b1, b2]; [b1 b2] d]) == (3, 3)
        @test string(Giac.Commands.factor(det([A Z; Z A]))) == string(Giac.Commands.factor(det(A)^2))
        # Julia's `true` is the integer 1; Giac's logical constant is giac_eval("true")
        @test string(convert(GiacExpr, true)) == "1" && string(x1 + true) == "x1+1"
    end

    @testset "det and inv hand off to Giac" begin
        @test string(det(A)) == "a11*a22-a12*a21"
        Ainv = inv(A)
        @test Ainv isa Matrix{GiacExpr}
        @test string.(Giac.Commands.simplify.(A * Ainv)) == ["1" "0"; "0" "1"]
        sol = A \ x
        @test string.(Giac.Commands.simplify.(A * sol - x)) == ["0", "0"]
    end

    @testset "symbolic scalars with arrays and I" begin
        @giac_var s t
        @test string.(s * I - A) == ["-a11+s" "-a12"; "-a21" "-a22+s"]
        @test string.(A - I * s) == ["a11-s" "a12"; "a21" "a22-s"]
        @test string(det(s * I - GiacExpr[0 1; -2 -3])) == "s^2+3*s+2"
        @test string.(t * x) == ["t*x1", "t*x2"]
        @test string.([1 2] * t) == ["t" "2*t"]
        @test string.(x / t) == ["x1/t", "x2/t"]
        @test size(A ~ s * I) == (2, 2)
    end

    @testset "spectra, rank, null space, exp" begin
        @giac_var t k
        M = GiacExpr[0 1; -2 -3]
        @test string.(eigvals(M)) == ["-1", "-2"]
        E = eigen(M)
        @test all(iszero, Giac.Commands.simplify.(M * E.vectors - E.vectors * Diagonal(E.values)))
        @test eigvecs(M) == E.vectors
        @test startswith(sprint(show, MIME("text/latex"), E), "\$\$\\Lambda = " * Giac.giac_latex(GiacExpr[-1 0; 0 -2]))
        @test length(eigvals(GiacExpr[0 1; -k -3])) == 2
        @test string.(eigvals(GiacExpr[1 1; 0 1])) == ["1", "1"]
        @test_throws ArgumentError eigen(GiacExpr[1 1; 0 1])     # not diagonalizable
        @test rank(M) == 2 && rank(GiacExpr[1 2; 2 4]) == 1
        N = nullspace(GiacExpr[1 2; 2 4])
        @test size(N) == (2, 1) && all(iszero, GiacExpr[1 2; 2 4] * N)
        @test size(nullspace(M)) == (2, 0)
        Φ = exp(M * t)
        @test string(Φ[1, 1]) == "2*exp(-t)-exp(-2*t)"
        @test string.(substitute(Φ, t => 0)) == ["1" "0"; "0" "1"]
        @test all(iszero, Giac.Commands.simplify.(map(e -> Giac.Commands.diff(e, t), Φ) - M * Φ))   # Φ' = MΦ
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
        @test string.(Giac.giac_array(:v, 3)) == ["v1", "v2", "v3"]
        @test Giac.giac_array(:v, 3) isa Vector{GiacExpr}
        @test string.(Giac.giac_array(:m, 2, 2)) == ["m11" "m12"; "m21" "m22"]
        @test string.(Giac.giac_array(:s, 2, 2; symmetric = true)) == ["s11" "s12"; "s12" "s22"]
        @test string.(Giac.giac_array(:w, 0:1)) == ["w0", "w1"]
        @test string(Giac.giac_array(:big, 10)[10]) == "big_10"
        @test_throws ArgumentError Giac.giac_array(:s, 2, 3; symmetric = true)
        @test_throws ArgumentError Giac.giac_array(:s)
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

@testset "vector calculus" begin
    @giac_var w[1:2] k
    f = w1^2 * w2 + k * w2
    @test string.(∇(f, w)) == ["2*w1*w2", "w1^2+k"]
    @test string.(hessian(f, w)) == ["2*w2" "2*w1"; "2*w1" "0"]
    @test string.(jacobian([w1 * w2, w1 + w2], w)) == ["w2" "w1"; "1" "1"]
    @test size(jacobian([w1, w2, k * w1], w)) == (3, 2)
end

@testset "array typesetting" begin
    @giac_var r[1:2] k
    @test Giac.giac_latex([r1 r2; k 0]) == "\\begin{bmatrix}r_{1} & r_{2} \\\\ k & 0\\end{bmatrix}"
    @test Giac.giac_latex(r) == "\\begin{bmatrix}r_{1} \\\\ r_{2}\\end{bmatrix}"
    @test Giac.giac_latex(r') == "\\begin{bmatrix}r_{1} & r_{2}\\end{bmatrix}"
    @test Giac.giac_latex(r1) == "r_{1}"
    # a column of plain numbers is right-aligned; a column with anything symbolic stays centred
    @test Giac.giac_latex(GiacExpr[2, -1]) ==
          "\\left[\\hspace{-5pt}\\begin{array}{r}2 \\\\ -1\\end{array}\\hspace{-5pt}\\right]"
    @test occursin("\\begin{array}{r}", Giac.giac_latex(GiacExpr[1 // 2, -3]))
    @test occursin("\\begin{array}{cr}", Giac.giac_latex([r1 1; -2 -3]))
    @test startswith(Giac.giac_latex([r1, -1]), "\\begin{bmatrix}")
    @test Giac._is_list(solve(k^2 ~ 4, k)) && !Giac._is_list(giac_eval("[1,2]"))
    strut = "\\rule[-1.4em]{0pt}{3.3em}"
    @test Giac.giac_latex([1 / r1 0; 0 1]) ==
          "\\left[\\hspace{-5pt}\\begin{array}{cr}$(strut)\\dfrac{1}{r_{1}} & 0 \\\\ $(strut)0 & 1\\end{array}\\hspace{-5pt}\\right]"
    @test startswith(sprint(show, MIME("text/latex"), [r1 r2]), "\$\$\\begin{bmatrix}")
    # solve's lists are sets; points are column vectors
    @test Giac.giac_latex(solve([r1 * (1 - r1), r2] ~ 0, r)) ==
          "\\left\\{" * Giac.giac_latex(GiacExpr[0, 0]) * ",\\; " * Giac.giac_latex(GiacExpr[1, 0]) * "\\right\\}"
    @test Giac.giac_latex(solve(k^2 ~ 4, k)) == "\\left\\{-2,\\; 2\\right\\}"
    # a lone expression is tidied too
    @giac_var τ
    @test sprint(show, MIME("text/latex"), τ^2) == "\$\$\\tau^{2}\$\$"
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
    @test string(Giac.giac_identifier("u(t)")) == "u(t)"
    giac_eval("purge(assigned_elsewhere)")
end

@testset "@derivation" begin
    @giac_var q[1:2] s
    @giac_matrix S[2, 2] symmetric
    d = @derivation Giac.Commands.expand begin
        W = q' * S * q / 2
        ∇(W, q)
        diff(W, s) ~ s * W
    end
    @test d isa Giac.Derivation && length(d.rows) == 3
    @test string(W) == string(q' * S * q / 2)              # bound, unsimplified
    @test d.rows[1][1] == "W"
    @test d.rows[2][1] == "\\nabla _{q}W"
    @test occursin("\\mathrm{d}", d.rows[3][1])
    @test d.rows[2][2] == Giac.giac_latex(Giac.Commands.expand.(S * q))   # ∇(qᵀSq/2) = Sq
    tex = sprint(show, MIME("text/latex"), d)
    @test startswith(tex, "\$\$\\begin{aligned}") && count("&=", tex) == 3

    # rows record the value at the time of each step
    d2 = @derivation begin
        z = s
        z = 2s
    end
    @test d2.rows[1][2] == "s" && d2.rows[2][2] == "2 s"

    # chains show the middle terms as written; strings are text rows
    d3 = @derivation begin
        "by definition"
        diff(W, s) ~ ∇(W, q) ⋅ q ~ Giac.Commands.expand(q' * S * q)
        raw"since $S$ is symmetric"
    end
    @test length(d3.rows) == 3
    @test d3.rows[1] == ("", "\\text{by definition}")
    @test count(" = ", d3.rows[2][2]) == 1 && occursin("\\nabla", d3.rows[2][2])
    @test d3.rows[3][2] == "\\text{since \$S\$ is symmetric}"
    tex3 = sprint(show, MIME("text/latex"), d3)
    # notes are flush-left rows of an outer array; the steps between them stay aligned
    @test startswith(tex3, "\$\$\\begin{array}{l}\\text{by definition}") && count("&=", tex3) == 1

    # written left sides: transpose, dot, reserved letters
    @test Giac.written_latex(:(x' * P * x)) == "x^{\\mathsf{T}} P x"
    @test occursin("\\Delta", Giac.written_latex(:(Δ + 1)))
    # module-qualified and broadcast calls are written as plain calls
    @test Giac.written_latex(:(Giac.Commands.factor(x))) == Giac.written_latex(:(factor(x)))
    @test Giac.written_latex(:(C.expand.(x))) == Giac.written_latex(:(expand(x)))
end

@testset "tidy_latex" begin
    T = Giac.tidy_latex
    @test T("\\mathrm{P★}") == "P^{\\star}"
    @test T("\\Delta \\mathrm{τ}^{2}") == "\\Delta \\tau^{2}"
    @test T("\\mathrm{θ1}") == "\\theta_{1}"
    @test T("\\frac1{b}") == "\\frac{1}{b}"
    @test T("\\mathrm{det}\\left(P\\right)") == "\\det\\left(P\\right)"
    @test T("\\mathrm{trace}(A)") == "\\operatorname{tr}(A)"
    @test T("x_{1}+\\mathrm{Pstar}") == "x_{1}+\\mathrm{Pstar}"         # ASCII names untouched
    @test T("A₀") == "A_{0}" && T("\\mathrm{A₁₂}") == "A_{12}"
    @test T("x̄") == "\\bar{x}" && T("θ̇") == "\\dot{\\theta}" && T("\\mathrm{q̇1}") == "\\dot{q}_{1}"
    @giac_var A₀ x̄
    @test Giac.giac_latex(A₀ + x̄) == "A_{0}+\\bar{x}"
    @giac_var τ
    @test Giac.giac_latex(τ^2) == "\\tau^{2}"
end
