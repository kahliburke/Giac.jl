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
        @test R(x^2 + (x + 1)^2, squares) in ("100*x+(x+1)^2", "(x+1)^2+100*x")     # (x+1)^2 untouched
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
        @test R(2x^2 + 3x + 7, square) == "2*(x+3/4)^2+47/8"        # -9/8 + 7 combine
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
        @test length(t.by) == 2 && length(t) == 3
        @test first(t) == a * (x + y) + sin(z)^2 + cos(z)^2 && last(t) == t[end]
        @test t.by[2] === pythagoras && collect(t) == t.exprs
        tex = sprint(show, MIME("text/latex"), t)
        @test count("\\text{by }", tex) == 2
        @test occursin("no rule applies", sprint(show, MIME("text/latex"), rewrite(x, pythagoras; trace = true)))
    end

    @testset "sums up to a common factor" begin
        @test R(3sin(x)^2 + 3cos(x)^2, pythagoras) == "3"
        @test rewrite(a * sin(x)^2 + a * cos(x)^2 + y, pythagoras) == a + y
        @test rewrite(sin(x)^2 / 2 + cos(x)^2 / 2, pythagoras) == 1 // 2
        # a larger coefficient is used in part
        @test rewrite(3sin(x)^2 + 2cos(x)^2, pythagoras) == 2 + sin(x)^2
        @test rewrite(2sin(x)^2 + 3cos(x)^2 + y, pythagoras) == 2 + cos(x)^2 + y
        @test pythagoras(3sin(x)^2 - 2cos(x)^2) === nothing     # opposite signs
        # a product up to its numeric coefficient
        double = @rule 2 * sin(~t) * cos(~t) => sin(2 * ~t)
        @test rewrite(sin(x) * cos(x), double) == sin(2x) / 2
        @test rewrite(3sin(x) * cos(x) * y, double) == 3 // 2 * sin(2x) * y
    end

    @testset "defaults and powers" begin
        merge = @rule ~v^(~n = 1) * ~v^(~m = 1) => ~v^(~n + ~m)
        @test R(x * x^2, merge) == "x^3"
        exponent = @rule ~v^~n => ~n
        @test string(exponent(1 / x)) == "-1"                   # inv(x) is x^-1
        @test string(exponent(1 / x^2)) == "-2"
        coefficient = @rule (~c = 1) * sin(~t) => ~c
        @test R(sin(y), coefficient) == "1" && R(3sin(y), coefficient) == "3"
        @giac_var f
        second = @rule f(~p, ~q = 0) => ~p + ~q
        @test R(f(x), second) == "x" && R(f(x, y), second) == "x+y"
    end

    @testset "segments with tests" begin
        @giac_var q[1:2]
        parts = @rule ~~l::of_degree(1, q) + ~~n::of_degree(2, q) + ~~rest => 100length(~~l) + 10length(~~n) + length(~~rest)
        @test string(parts(q1 + 2q2 + q1 * q2 + q1^2 + 7 + sin(q1))) == "222"
        linear = @rule +(~~l::of_degree(1, q)) + ~~rest => +(~~l)
        @test linear(q1 + 2q2 + q1 * q2 + 7) == q1 + 2q2
        consts = @rule ~~c::is_free(q) + ~~rest => length(~~c)
        @test string(consts(q1 + a + 3)) == "2"
        @test depends_on(q1 * a, q) && !depends_on(a, q)
    end

    @testset "tests that use assumptions" begin
        @giac_var p::positive w::positive r::real
        @test is_positive(p + 1) && is_positive(p * w) && !is_positive(x)
        @test is_negative(-p) && is_nonnegative(p^2 + 0) && is_real(r * p)
        @test is_positive(3) && !is_positive(-2)
        @giac_var p w r
        @test !is_positive(p)
    end

    @testset "identity library" begin
        @giac_var p::positive w::positive
        @test rewrite(log(p * w), log_expand()) == log(p) + log(w)
        @test rewrite(2log(p) + 2log(w), log_combine()) == 2log(p * w)
        @test rewrite(sqrt(p) * sqrt(w), power_combine()) == sqrt(p * w)
        @giac_var p w
        @test string(rewrite(log(x * y), log_expand())) == string(log(x * y))   # sign unknown
        @test rewrite((x * y)^3, power_expand()) == x^3 * y^3                    # integer power
        @test rewrite(exp(x) * exp(y), exp_combine()) == exp(x + y)
        @test rewrite(3 - 3sin(x)^2, trig_simplify()) == 3cos(x)^2
        @test rewrite(sin(x) / cos(x) + 1, trig_simplify()) == tan(x) + 1
        @test trig_simplify() === trig_simplify()                                 # built once
        @test occursin("> 0", sprint(show, MIME("text/latex"), log_expand()))
    end

    @testset "patterns as queries" begin
        quad = @pattern ~a * x^2 + (~b = 0) * x + ~c
        m = match(quad, 3x^2 + 2x + 1)
        @test string.((m[:a], m[:b], m[:c])) == ("3", "2", "1")
        @test string(match(quad, 3x^2 + 1)[:b]) == "0"
        @test match(quad, x^3) === nothing
        @test occursin(@pattern(sin(~t)), x + sin(y)^2) && !occursin(@pattern(sin(~t)), x + y)
        found = findall(@pattern(sin(~t)), sin(x) + cos(sin(y)))
        @test Set(string(f[:t]) for f in found) == Set(["x", "y"])
        classify(e) = @match e begin
            ~a * x^2 + (~b = 0) * x + ~c => (:quadratic, string(~a)) where is_free(~a, x) && is_free(~b, x) && is_free(~c, x)
            ~a * x + ~b => (:linear, string(~a)) where is_free(~a, x)
            _ => :other
        end
        @test classify(3x^2 + 1) == (:quadratic, "3")
        @test classify(2x + y) == (:linear, "2")
        @test classify(x^3) == :other
        strict(e) = @match e begin
            sin(~t) => ~t
        end
        @test_throws ArgumentError strict(x)
    end

    @testset "where and which place" begin
        distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
        @test rewrite(a * (x + y) + sin(z * (x + y)), distribute; within = @pattern(sin(~t))) ==
              a * (x + y) + sin(z * x + z * y)
        e = a * (x + y) + z * (x + w)
        @test rewrite(e, distribute; at = 1) == a * x + a * y + z * (x + w)
        @test rewrite(e, distribute; at = 2) == a * (x + y) + z * x + z * w
        @test rewrite(e, distribute; at = 3) == e
    end

    @testset "traces and cycles" begin
        distribute = @rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)
        e = a * (x + y) + z * (x + w) + sin(y)^2 + cos(y)^2
        rows(t) = Giac.Rules._trace_rows(t)
        @test length(rows(rewrite(e, distribute, pythagoras; trace = true))) == 2      # distribute ×2
        @test length(rows(rewrite(e, distribute, pythagoras; trace = :steps))) == 3
        @test occursin("\\times 2", sprint(show, MIME("text/latex"), rewrite(e, distribute, pythagoras; trace = true)))
        swap = @rules begin
            sin(~t) => cos(~t)
            cos(~t) => sin(~t)
        end
        err = try rewrite(sin(x), swap); nothing catch err; err end
        @test err isa ErrorException && occursin("undo each other", err.msg) && occursin("cos(~t) => sin(~t)", err.msg)
        @test_throws ErrorException rewrite(sin(x), swap; trace = true)
    end

    @testset "simplest" begin
        e = sin(x)^4 - cos(x)^4
        @test rewrite(e, trig_simplify()) == e                  # no rule applies as it stands
        @test simplest(e, trig_simplify(), Giac.Commands.factor, Giac.Commands.expand) == -cos(2x)
        t = simplest(e, trig_simplify(), Giac.Commands.factor, Giac.Commands.expand; trace = true)
        @test last(t) == -cos(2x)
        @test length(t.by) == 4 && first(t) == e
        @test expression_size(x + y) == 3
        @test simplest(x + y, trig_simplify()) == x + y
        @test_throws ArgumentError simplest(x, 42)
    end

    @testset "rules from identities" begin
        double = rule(sin(2x) ~ 2sin(x) * cos(x), x)
        @test double(sin(2y)) == 2sin(y) * cos(y)
        @test double(sin(2 * (y + 1))) == 2sin(y + 1) * cos(y + 1)
        @test double(sin(y)) === nothing
        @test occursin("\\to", sprint(show, MIME("text/latex"), double))
        back = inverse(double)
        @test back(2sin(y) * cos(y)) == sin(2y)
        @test back(sin(y) * cos(y)) == sin(2y) / 2               # up to the coefficient
        combine = inverse.(log_expand())
        @giac_var p::positive w::positive
        @test rewrite(log(p) + log(w), combine) == log(p * w)
        @giac_var p w
        @test_throws ArgumentError inverse(@rule ~u * +(~~ts) => sum(~u * t for t in ~~ts))
        @test_throws ArgumentError inverse(@rule sin(~u) * cos(~v) => sin(~u))       # drops ~v
        @test_throws ArgumentError rule(x + 1, x)
    end

    @testset "equivalent and verify" begin
        @test equivalent(sin(x)^2 + cos(x)^2, 1)
        @test equivalent((x + 1)^2, x^2 + 2x + 1)
        @test !equivalent(log(x * y), log(x) + log(y))           # not for negative x and y
        @test !equivalent(sqrt(x^2), x)
        @giac_var p::positive w::positive
        @test equivalent(log(p * w), log(p) + log(w))            # sampled where assumed
        @giac_var p w
        @test verify(rule(sin(2x) ~ 2sin(x) * cos(x), x)) === true
        @test verify(rule(sin(2x) ~ sin(x) * cos(x), x)) === false
        @test all(verify(trig_simplify()))
        @test all(verify(log_expand())) && all(verify(power_combine())) && all(verify(exp_combine()))
    end

    @testset "substituting a relation" begin
        @giac_var r s
        @test substitute(x^2 + y^2 + z, x^2 + y^2 => r^2) == r^2 + z
        @test substitute(2x^2 + 2y^2 + z, x^2 + y^2 => r^2) == 2r^2 + z
        @test substitute(x^2 + 2y^2, x^2 + y^2 => r^2) == r^2 + y^2
        @test string(substitute(x^4 + 2x^2 * y^2 + y^4, x^2 + y^2 => r^2)) == "r^4"   # by Gröbner reduction
        @test substitute(3sin(z)^2 + 3cos(z)^2 + x, sin(z)^2 + cos(z)^2 => 1) == 3 + x
        @test substitute(x * y + s, x * y => r, s => 2) == r + 2
    end

    @testset "simplest with the standard transforms" begin
        @test simplest(sin(z)^4 - cos(z)^4) == -cos(2z)
        @test string(simplest((x^2 - 1) / (x^2 + 2x + 1))) == "(x-1)/(x+1)"
        @test simplest(exp(2log(x)) + exp(x) * exp(y)) == x^2 + exp(x + y)
        @test simplest(2sin(z) * cos(z) + 3sin(z)^2 + 3cos(z)^2) == sin(2z) + 3
        @test simplest(sin(z)^3; goal = :trig_linear) == (3sin(z) - sin(3z)) / 4
        @test string(simplest((x + 1)^2 * (x - 1); goal = :expanded)) == "x^3+x^2-x-1"
        @test string(simplest(x^2 - 1; goal = :factored)) == "(x-1)*(x+1)"
        @test_throws ArgumentError simplest(x; goal = :prettiest)
        @test goal_cost(:factored, (x - 1) * (x + 1))[1] == 1
        # with verify, only steps checked to be equal are taken
        @test simplest(sin(z)^4 - cos(z)^4; verify = true) == -cos(2z)
        @test any(x -> x isa Transform, standard_transforms())
        @test string.(simplest([x^2 - 1, (x + 1)^2 - x^2]; goal = :factored)) == ["(x-1)*(x+1)", "2*x+1"]
    end

    @testset "traces in derivations" begin
        d = @derivation begin
            V = sin(z)^4 - cos(z)^4
            V ~ simplest(V; trace = true)
        end
        @test length(d.rows) == 3                                # V = e, V = e (start), {} = -cos(2z)
        @test occursin("\\text{by }", last(d.rows[end]))
    end

    @testset "identity catalog" begin
        all_ids = Giac.Rules._catalog()
        @test length(all_ids) > 40
        # every identity holds, in both directions, under its conditions
        @test all(r -> verify(r) === true, first.(all_ids))
        @test all(r -> verify(r) === true, identities(; direction = :expand))
        @test all(r -> verify(r) === true, identities(:trig_linear, :hyperbolic_linear))
        @test identities() isa IdentityCatalog && identities() === identities()
        @test all(t -> !endswith(string(t), "_linear"), identities().topics)
        @test_throws ArgumentError identities(:astrology)
        @test occursin("\\textbf{trig}", sprint(show, MIME("text/latex"), identities(:trig)))
        # generated families
        @test rewrite(sin(x) * cos(y) + cos(x) * sin(y), identities(:trig)) == sin(x + y)
        @test rewrite(cosh(x) * cosh(y) + sinh(x) * sinh(y), identities(:hyperbolic)) == cosh(x + y)
        @test rewrite(4cos(x)^3 - 3cos(x), identities(:trig)) == cos(3x)
        @test rewrite(3sinh(x) + 4sinh(x)^3, identities(:hyperbolic)) == sinh(3x)
        @test rewrite(sin(x)^2, identities(:trig_linear)) == (1 - cos(2x)) / 2
        @test rewrite(cosh(asinh(y)), identities(:inverse_trig)) == sqrt(1 + y^2)
        # conditions from tests on the variables, shown with the rule
        @giac_var u::(-1 // 2, 1 // 2) m::integer
        @test rewrite(sinh(atanh(u)), identities(:inverse_trig)) == u / sqrt(1 - u^2)
        @test string(rewrite(sinh(atanh(y)), identities(:inverse_trig))) == string(sinh(atanh(y)))   # |y| < 1 unknown
        @test rewrite(Giac.invoke_cmd(:floor, 2m + 1), identities(:integer)) == 2m + 1
        @giac_var u m
        @test occursin("\\le", sprint(show, MIME("text/latex"), identities(:inverse_trig)))
    end

    @testset "integer and interval tests" begin
        @giac_var k::integer v::(0, 1)
        @test is_integer_valued(2k + 1) && is_integer_valued(k^2) && is_integer_valued(3)
        @test !is_integer_valued(x) && !is_integer_valued(k / 2)
        @test in_interval(v, 0, 1) && !in_interval(v, 0, 1 // 2) && in_interval(1 // 3, 0, 1)
        @test !in_interval(x, 0, 1)
        @giac_var k v
    end

    @testset "properties of functions" begin
        @giac_var t n::integer
        @giac_var φ(t)::odd ψ(t)::(even, periodic(2π)) L(t)::linear h(t)::involution
        @giac_var g(t) f(t)::inverse_of(g)
        F(name) = Giac.giac_identifier(name)
        φ_, ψ_, L_, h_, f_, g_ = F("φ"), F("ψ"), F("L"), F("h"), F("f"), F("g")
        rules = function_identities()
        @test length(rules) == 9
        @test rewrite(φ_(-(t^2 + 1)), rules) == -φ_(t^2 + 1)          # Giac writes the argument -t^2-1
        @test string(rewrite(φ_(t - 1), rules)) == string(φ_(t - 1))   # not mostly negative: unchanged
        @test rewrite(ψ_(-t), rules) == ψ_(t)
        p = convert(GiacExpr, π)
        @test string(rewrite(ψ_(t + 4p) + ψ_(t - 2p) + ψ_(t + 2n * p), rules)) == "3*ψ(t)"
        @test string(rewrite(ψ_(t + p), rules)) == string(ψ_(t + p))   # π is not a period
        @test rewrite(L_(3t + t^2), rules) == 3L_(t) + L_(t^2)
        @test rewrite(h_(h_(t + 1)), rules) == t + 1
        @test rewrite(f_(g_(t)) + g_(f_(a)), rules) == t + a
        @test simplest(φ_(-t) + φ_(t) + sin(t)^2 + cos(t)^2) == 1
        tex = sprint(show, MIME("text/latex"), rules)
        @test occursin("2 \\pi", tex) && occursin("\\varphi", tex)
        @test_throws ArgumentError Giac._declare_function!(φ_(t), :bouncy)
        # a declaration without properties drops them
        @giac_var φ(t) ψ(t) L(t) h(t) f(t)
        @test isempty(function_identities())
        @giac_var n
    end

    @testset "properties of matrices" begin
        @giac_matrix Q[2, 2]::orthogonal P[2, 2]::posdef N[2, 2]::negdef A[2, 2]::invertible
        @test equivalent(Q' * Q, I) && equivalent(Q * Q', I)
        @test apply_relations(LinearAlgebra.det(Q)^2) == 1
        @test apply_relations(q11^2 + q21^2 + y) == 1 + y
        @test simplest((q11^2 + q21^2)^2) == 1
        @test !equivalent(q11, q22)
        @test is_positive(p11) && is_positive(LinearAlgebra.det(P)) && is_positive(3LinearAlgebra.det(P))
        @test !is_positive(p12)
        @test is_negative(n11) && is_positive(LinearAlgebra.det(N))
        @test is_nonzero(LinearAlgebra.det(A)) && !is_nonzero(a11)
        @test any(t -> t isa Transform && t.name == "declared relations", standard_transforms())
        # a new declaration drops what was recorded
        @giac_matrix Q[2, 2] P[2, 2] N[2, 2] A[2, 2]
        @test !any(n -> haskey(Giac._MATRIX_FACTS, n), ("Q", "P", "N", "A"))
        @test !is_positive(p11)
        @test isempty(Giac._relations()) == !any(t -> t isa Transform && t.name == "declared relations", standard_transforms())
        @test_throws ArgumentError @giac_matrix B[2, 3]::orthogonal
    end
end
