# Pattern rewrite rules (Giac.Rules)

using Giac.Rules

@testset "Rules" begin
    @giac_var x y z w a b
    factor_out = @rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
    pythagoras = @rule sin(~t)^2 + cos(~t)^2 => 1
    R(e, rules...) = string(rewrite(e, rules...))

    @testset "sums and products match in any order" begin
        @test R(x * y + x * z, factor_out) == "x*(y+z)"
        @test R(z * x + y * x, factor_out) == "x*(z+y)"
        @test R(x * y + z * w, factor_out) == "x*y+z*w"                 # no common factor
        @test R(2x * y + 2x * z, factor_out) == "2*x*(y+z)"             # the last slot takes the rest
    end

    @testset "unused terms at the top are kept" begin
        @test R(x * y + x * z + w, factor_out) == "x*(y+z)+w"
        @test R(sin(a)^2 + cos(a)^2 + b, pythagoras) == "1+b"
        @test R(exp(x) * exp(y) * z, @rule exp(~p) * exp(~q) => exp(~p + ~q)) == "exp(x+y)*z"
    end

    @testset "identity and negation" begin
        @test R(x + x * y, factor_out) == "x*(1+y)"                     # x is x*1
        @test R(x - x * y, factor_out) == "x*(1-y)"                     # -(x*y) is (-1)*x*y
        @test string((@rule ~p - ~q => ~q - ~p)(x - y)) == "y-x"        # the pattern is normalised too
    end

    @testset "repeated slots must agree" begin
        @test R(sin(a)^2 + cos(b)^2, pythagoras) == "sin(a)^2+cos(b)^2"
    end

    @testset "applies throughout the expression" begin
        @test R(exp(sin(a)^2 + cos(a)^2), pythagoras) == "exp(1)"
        @test R(giac_eval("f(sin(a)^2+cos(a)^2, b)"), pythagoras) == "f(1,b)"
    end

    @testset "Julia variables in patterns" begin
        @giac_matrix A[2, 2]
        @giac_var τ
        @test R(a11 + a22 + 1, @rule a11 + a22 => τ) == "τ+1"
    end

    @testset "predicates and declined matches" begin
        squares = @rule (~v::is_identifier)^2 => 100 * ~v
        @test R(x^2 + (x + 1)^2, squares) == "100*x+(x+1)^2"
        cubes = @rule ~v^~n => (string(~n) == "3" ? ~v : nothing)     # `~v^~n` reads as `(~v)^(~n)`
        @test R(x^2 + y^3, cubes) == "x^2+y"
    end

    @testset "rule sets, arrays and numbers" begin
        @giac_matrix A[2, 2]
        @giac_var τ
        rs = @rules begin
            sin(~t)^2 + cos(~t)^2 => 1
            a11 + a22 => τ
        end
        @test rs isa Vector{Giac.Rules.Rule} && length(rs) == 2
        @test R(sin(x)^2 + cos(x)^2 + a11 + a22, rs) == "τ+1"
        @test string.(rewrite([a11 + a22 x * y + x * z; 1 0], rs, factor_out)) == ["τ" "x*(y+z)"; "1" "0"]
        @test R(3, rs) == "3"
        @test pythagoras(x) === nothing
        @test sprint(show, pythagoras) == "Rule(sin(~t) ^ 2 + cos(~t) ^ 2 => 1)"
    end

    @testset "sums match up to sign" begin
        @giac_matrix A[2, 2]
        @giac_var Δ
        det_rule = @rule a11 * a22 - a12 * a21 => Δ
        @test R(a12 * a21 - a11 * a22, det_rule) == "-Delta"
        @test R(a12 * a21 - a11 * a22 + 5, det_rule) == "-Delta+5"
        @test R(-sin(a)^2 - cos(a)^2 + b, pythagoras) == "-1+b"
        @test R(x * y + z * w, factor_out) == "x*y+z*w"                 # still no match either way
    end

    @testset "typesetting" begin
        tex = sprint(show, MIME("text/latex"), pythagoras)
        @test startswith(tex, "\$\$") && occursin("\\to", tex) && !occursin("~", tex)
        @test occursin("\\sin", tex)
        rs = @rules begin
            sin(~t)^2 + cos(~t)^2 => 1
            ~u * ~v + ~u * ~w => ~u * (~v + ~w)
        end
        @test count("\\to", sprint(show, MIME("text/latex"), rs)) == 2
        cubes = @rule ~v^~n => (string(~n) == "3" ? ~v : nothing)       # Julia logic: shown as code
        @test occursin("\\texttt{", sprint(show, MIME("text/latex"), cubes))
    end

    @testset "errors" begin
        @test_throws ErrorException rewrite(x, @rule(~v => ~v + 1); maxpasses = 5)
        @test_throws ArgumentError Giac.Rules._rule_expr(:(x + y))
        @test_throws ArgumentError Giac.Rules._rule_expr(:(~x + ~~x => 0))
    end

    @testset "segments" begin
        distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
        @test R(a * (x + y + z), distribute) == "a*x+a*y+a*z"
        @test distribute(a * x) === nothing                     # +(~~ts) needs a sum
        @test rewrite(a * (x + y) * (z + w), distribute) == a * x * z + a * x * w + a * y * z + a * y * w
        # the terms a segment takes are a vector; a segment in a sum may take none
        nterms = @rule sin(~t) + ~~rest => length(~~rest)
        @test nterms(sin(x) + y + z) == 2 && nterms(sin(x)) == 0
        # a run of a call's arguments
        @giac_var f
        nargs = @rule f(~a, ~~rest) => ~a + length(~~rest)
        @test R(f(x, y, z), nargs) == "x+2"
        @test R(f(x), nargs) == "x"
        # +(~~xs) on the right side is the sum again
        swap = @rule cos(~t) + ~~rest => sin(~t) + +(~~rest)
        @test rewrite(cos(x) + y + z, swap) == sin(x) + y + z
    end

    @testset "where" begin
        trunc3 = @rule ~v^~n => 0 where ~n > 3
        @test rewrite(1 + x + x^2 + x^3 + x^4 + x^5, trunc3) == 1 + x + x^2 + x^3
        square = @rule ~a * ~v^2 + ~b * ~v + ~~rest => ~a * (~v + ~b / (2 * ~a))^2 - ~b^2 / (4 * ~a) + +(~~rest) where is_free(~a, ~v) && is_free(~b, ~v)
        @test R(2x^2 + 3x + 7, square) == "2*(x+3/4)^2-9/8+7"
        @test square(x * x^2 + 3x) === nothing                # a coefficient that depends on x
        @test is_free(a * y, x) && !is_free(a * x, [x, y])
        tex = sprint(show, MIME("text/latex"), trunc3)
        @test occursin("\\text{if } n > 3", tex)
        @test occursin("free of", sprint(show, MIME("text/latex"), square))
        @test_throws ArgumentError (@rule ~v => 0 where 3)(x)  # a condition must be true or false
    end

    @testset "strategies" begin
        distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
        e = a * (x + y * (z + w))
        @test rewrite(e, distribute; order = :topdown) == a * x + a * y * z + a * y * w
        lower = @rule ~v^~n => ~v^(~n - 1) where ~n > 1
        @test rewrite(x^3, lower; once = true) == x^2         # one pass
        @test rewrite(x^3, lower) == x
        @test_throws ArgumentError rewrite(e, distribute; order = :sideways)
        # without an expression, a function to compose
        @test (sin(x)^2 + cos(x)^2 + a * (x + y)) |> rewrite(pythagoras) |> rewrite(distribute) == 1 + a * x + a * y
        # a trace: one rule in one place per step
        t = rewrite(a * (x + y) + sin(z)^2 + cos(z)^2, distribute, pythagoras; trace = true)
        @test last(t) == 1 + a * x + a * y
        @test length(t.steps.rows) == 2
        tex = sprint(show, MIME("text/latex"), t)
        @test count("\\text{by }", tex) == 2
        @test occursin("no rule applies", sprint(show, MIME("text/latex"), rewrite(x, pythagoras; trace = true)))
    end
end
