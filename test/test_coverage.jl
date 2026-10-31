# Additional tests to improve code coverage for uncovered but valid code paths

# ============================================================================
# 1. Iteration edge cases (src/iteration.jl)
# ============================================================================
@testset "Iteration Coverage" begin
    @testset "Scalar iteration" begin
        # Iterating over a non-vector GiacExpr yields itself once
        x = giac_eval("42")
        # Use iterate directly since collect may use getindex which throws for scalars
        result = iterate(x)
        @test result !== nothing
        @test string(result[1]) == "42"
        # Second call should return nothing (single element)
        result2 = iterate(x, result[2])
        @test result2 === nothing
    end

    @testset "Scalar length and size" begin
        x = giac_eval("42")
        @test length(x) == 1
        @test size(x) == (1,)
        @test firstindex(x) == 1
        @test lastindex(x) == 1
        @test collect(eachindex(x)) == [1]
    end

    @testset "Vector iteration" begin
        v = giac_eval("[1,2,3]")
        collected = collect(v)
        @test length(collected) == 3
        @test string(collected[1]) == "1"
        @test string(collected[2]) == "2"
        @test string(collected[3]) == "3"
    end

    @testset "Scalar in operator" begin
        # in() on a non-vector compares string representations
        x = giac_eval("42")
        @test giac_eval("42") in x
        @test !(giac_eval("43") in x)
    end

    @testset "Vector keys/values/pairs" begin
        v = giac_eval("[10,20,30]")
        ks = collect(keys(v))
        @test ks == [1, 2, 3]
        vs = collect(values(v))
        @test length(vs) == 3
        @test string(vs[2]) == "20"
        ps = collect(pairs(v))
        @test ps[1][1] == 1
        @test string(ps[1][2]) == "10"
    end

    @testset "Vector slicing" begin
        v = giac_eval("[1,2,3,4,5]")
        sliced = v[2:4]
        @test length(sliced) == 3
        @test string(sliced[1]) == "2"

        all_elems = v[:]
        @test length(all_elems) == 5
    end

    @testset "Indexing errors" begin
        x = giac_eval("42")
        @test_throws ErrorException x[1]

        v = giac_eval("[1,2,3]")
        @test_throws BoundsError v[0]
        @test_throws BoundsError v[4]

        # Range indexing on non-vector
        @test_throws ErrorException x[1:2]
    end
end

# ============================================================================
# 2. Introspection edge cases (src/introspection.jl)
# ============================================================================
@testset "Introspection Edge Cases" begin
    @testset "numer on integer" begin
        n = giac_eval("5")
        result = Giac.numer(n)
        @test string(result) == "5"
    end

    @testset "denom on integer" begin
        n = giac_eval("5")
        result = Giac.denom(n)
        @test string(result) == "1"
    end

    @testset "numer/denom on fraction" begin
        f = giac_eval("3/7")
        @test string(Giac.numer(f)) == "3"
        @test string(Giac.denom(f)) == "7"
    end

    @testset "real_part on real number" begin
        r = giac_eval("3")
        @test string(Giac.real_part(r)) == "3"
    end

    @testset "imag_part on real number" begin
        r = giac_eval("3")
        @test string(Giac.imag_part(r)) == "0"
    end

    @testset "real_part/imag_part on complex" begin
        c = giac_eval("3+4*i")
        @test string(Giac.real_part(c)) == "3"
        @test string(Giac.imag_part(c)) == "4"
    end

    @testset "Type predicates" begin
        @test Giac.is_integer(giac_eval("42"))
        @test !Giac.is_integer(giac_eval("3.14"))
        @test Giac.is_numeric(giac_eval("42"))
        @test Giac.is_numeric(giac_eval("3.14"))
        @test !Giac.is_numeric(giac_eval("x"))
        @test Giac.is_vector(giac_eval("[1,2,3]"))
        @test !Giac.is_vector(giac_eval("42"))
        @test Giac.is_symbolic(giac_eval("sin(x)"))
        @test !Giac.is_symbolic(giac_eval("42"))
        @test Giac.is_identifier(giac_eval("x"))
        @test !Giac.is_identifier(giac_eval("42"))
        @test Giac.is_fraction(giac_eval("3/4"))
        @test !Giac.is_fraction(giac_eval("42"))
        @test Giac.is_boolean(giac_eval("true"))
        @test Giac.is_boolean(giac_eval("false"))
        @test !Giac.is_boolean(giac_eval("1"))
    end

    @testset "symb_funcname and symb_argument" begin
        s = giac_eval("sin(x)")
        @test Giac.symb_funcname(s) == "sin"
        arg = Giac.symb_argument(s)
        @test string(arg) == "x"
    end

    @testset "Error on non-fraction numer/denom" begin
        x = giac_eval("x")
        @test_throws GiacError Giac.numer(x)
        @test_throws GiacError Giac.denom(x)
    end

    @testset "Error on non-symbolic symb_funcname/symb_argument" begin
        n = giac_eval("42")
        @test_throws GiacError Giac.symb_funcname(n)
        @test_throws GiacError Giac.symb_argument(n)
    end
end

# ============================================================================
# 3. Held command LaTeX edge cases (src/held_cmd.jl)
# ============================================================================
@testset "Held command LaTeX Coverage" begin
    @giac_var x t s n k

    @testset "Directional limits" begin
        # Left limit (dir = -1)
        lim_left = hold_cmd(:limit, giac_eval("1/x"), x, giac_eval("0"), giac_eval("-1"))
        latex_left = sprint(show, MIME("text/latex"), lim_left)
        @test occursin("\\lim", latex_left)
        @test occursin("^-", latex_left)

        # Right limit (dir = 1)
        lim_right = hold_cmd(:limit, giac_eval("1/x"), x, giac_eval("0"), giac_eval("1"))
        latex_right = sprint(show, MIME("text/latex"), lim_right)
        @test occursin("\\lim", latex_right)
        @test occursin("^+", latex_right)
    end

    @testset "Limit with fewer args" begin
        # 2 args
        lim2 = hold_cmd(:limit, giac_eval("1/x"), x)
        latex2 = sprint(show, MIME("text/latex"), lim2)
        @test occursin("\\lim", latex2)

        # 1 arg
        lim1 = hold_cmd(:limit, giac_eval("1/x"))
        latex1 = sprint(show, MIME("text/latex"), lim1)
        @test occursin("\\lim", latex1)
    end

    @testset "Sum and product" begin
        h_sum = hold_cmd(:sum, giac_eval("k^2"), k, giac_eval("1"), giac_eval("10"))
        latex_sum = sprint(show, MIME("text/latex"), h_sum)
        @test occursin("\\sum", latex_sum)

        h_prod = hold_cmd(:product, giac_eval("k"), k, giac_eval("1"), giac_eval("5"))
        latex_prod = sprint(show, MIME("text/latex"), h_prod)
        @test occursin("\\prod", latex_prod)
    end

    @testset "Sum/product with fewer args" begin
        # 2 args
        h2 = hold_cmd(:sum, giac_eval("k^2"), k)
        latex2 = sprint(show, MIME("text/latex"), h2)
        @test occursin("\\sum", latex2)

        # 1 arg
        h1 = hold_cmd(:sum, giac_eval("k^2"))
        latex1 = sprint(show, MIME("text/latex"), h1)
        @test occursin("\\sum", latex1)
    end

    @testset "equations with held commands" begin
        held = hold_cmd(:integrate, giac_eval("x^2"), x)
        tex(e) = sprint(show, MIME("text/latex"), e)
        @test occursin("=", tex(giac_eval("x^2") ~ 0))
        @test occursin("\\int", tex(held ~ giac_eval("x^3/3"))) && occursin("=", tex(held ~ giac_eval("x^3/3")))
        both = hold_cmd(:diff, giac_eval("x^3/3"), x) ~ hold_cmd(:integrate, giac_eval("x"), x)
        @test occursin("\\frac{\\mathrm{d}}{\\mathrm{d} x}", tex(both)) && occursin("\\int", tex(both))
        f = hold_cmd(:factor, giac_eval("x^2-1"))
        @test occursin("factor", string(f ~ giac_eval("(x-1)*(x+1)"))) && occursin("=", sprint(show, MIME("text/plain"), f ~ 0))
        for eq in (f ~ giac_eval("(x-1)*(x+1)"), giac_eval("x^2") ~ f, f ~ hold_cmd(:diff, x, x), f ~ 42, 42 ~ f)
            @test eq isa GiacExpr && Giac._is_equation(eq)
        end
    end

    @testset "LaTeX: ilaplace alias" begin
        h = hold_cmd(:ilaplace, giac_eval("1/s"), s, t)
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\mathcal{L}^{-1}", latex)
    end

    @testset "LaTeX: ztransform alias" begin
        # held without the check that the command exists, to test its written form
        h = Giac._hold(:ztransform, (giac_eval("n"), n, giac_eval("z")))
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\mathcal{Z}", latex)
    end

    @testset "LaTeX: invztransform alias" begin
        h = Giac._hold(:invztransform, (giac_eval("1/z"), giac_eval("z"), n))
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\mathcal{Z}^{-1}", latex)
    end

    @testset "LaTeX: transform with fewer args" begin
        # 1 arg
        h1 = hold_cmd(:laplace, giac_eval("sin(t)"))
        latex1 = sprint(show, MIME("text/latex"), h1)
        @test occursin("\\mathcal{L}", latex1)

        # 0 args (edge case)
        h0 = hold_cmd(:laplace)
        latex0 = sprint(show, MIME("text/latex"), h0)
        @test occursin("\\mathcal{L}", latex0)

        # inv with 1 arg
        h1i = hold_cmd(:invlaplace, giac_eval("1/s"))
        latex1i = sprint(show, MIME("text/latex"), h1i)
        @test occursin("\\mathcal{L}^{-1}", latex1i)

        # inv with 0 args
        h0i = hold_cmd(:invlaplace)
        latex0i = sprint(show, MIME("text/latex"), h0i)
        @test occursin("\\mathcal{L}^{-1}", latex0i)
    end

    @testset "LaTeX: integrate with 1 arg" begin
        h = hold_cmd(:integrate, giac_eval("x^2"))
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\int", latex)
    end

    @testset "LaTeX: diff with 1 arg" begin
        h = hold_cmd(:diff, giac_eval("x^2"))
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\frac{d}{d?}", latex)
    end

    @testset "held command text/plain show" begin
        h = hold_cmd(:factor, giac_eval("x^2-1"))
        txt = sprint(show, MIME("text/plain"), h)
        @test occursin("GiacExpr:", txt)
        @test occursin("factor", txt)
    end

    @testset "_arg_to_latex edge cases" begin
        # Symbol arg
        h = hold_cmd(:diff, giac_eval("x^2"), :x)
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\mathrm{d} x", latex)

        # a string argument is read as GIAC input
        latex_str = sprint(show, MIME("text/latex"), hold_cmd(:factor, "x^2-1"))
        @test occursin("\\mathrm{factor}", latex_str) && occursin("x^{2}", latex_str)

        # a vector argument is a list
        latex_vec = sprint(show, MIME("text/latex"), hold_cmd(:sum, [1, 2, 3]))
        @test occursin("1", latex_vec) && occursin("3", latex_vec)
    end

    @testset "Generic fallback with no args" begin
        h = Giac._hold(:foo, ())
        latex = sprint(show, MIME("text/latex"), h)
        @test occursin("\\mathrm{foo}", latex)
        @test occursin("()", latex)
    end
end

# ============================================================================
# 4. GiacExpr method syntax (src/types.jl)
# ============================================================================
@testset "GiacExpr Method Syntax" begin
    @testset "Method-style command access" begin
        expr = giac_eval("x^2 - 1")
        result = expr.factor()
        @test string(result) == "(x-1)*(x+1)"
    end

    @testset "Method-style with additional args" begin
        @giac_var x
        expr = giac_eval("x^2")
        result = expr.integrate(x)
        @test occursin("x^3", string(result))
    end

    @testset "propertynames" begin
        expr = giac_eval("42")
        @test propertynames(expr) == (:ptr,)
    end
end
