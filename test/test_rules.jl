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
    end
end
