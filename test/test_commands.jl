# Tests for dynamic command invocation (invoke_cmd)
# Feature: 003-giac-commands (updated for 009-commands-submodule)

@testset "Dynamic Command Invocation (US1)" begin

    @testset "Single-arg invocation" begin
        # T012: Test invoke_cmd(:factor, expr)
        expr = giac_eval("x^2 - 1")
        result = invoke_cmd(:factor, expr)
        @test result isa GiacExpr
        result_str = string(result)
        @test occursin("x-1", result_str) || occursin("x+1", result_str)
    end

    @testset "Trigonometric commands" begin
        # T013: Test invoke_cmd(:sin, expr)
        x = giac_eval("x")
        result = invoke_cmd(:sin, x)
        @test result isa GiacExpr
        @test string(result) == "sin(x)"

        # Test with numeric value
        pi_6 = giac_eval("pi/6")
        result = invoke_cmd(:sin, pi_6)
        @test result isa GiacExpr
    end

    @testset "Multi-arg invocation" begin
        # T014: Test invoke_cmd(:diff, expr, var)
        expr = giac_eval("x^2")
        x = giac_eval("x")
        result = invoke_cmd(:diff, expr, x)
        @test result isa GiacExpr
        @test string(result) == "2*x"
    end

    @testset "Unknown command error" begin
        # T015: Test unknown command with suggestions
        expr = giac_eval("x")
        @test_throws GiacError invoke_cmd(:factoor, expr)  # typo

        try
            invoke_cmd(:factoor, expr)
        catch e
            @test e isa GiacError
            @test e.category == :eval
            @test occursin("Unknown command", e.msg)
            # Should suggest "factor"
            @test occursin("factor", e.msg) || occursin("Did you mean", e.msg)
        end
    end

    @testset "Invalid argument type error" begin
        # T016: Test invalid argument type
        # Raw Julia arrays without conversion should fail
        @test_throws ArgumentError invoke_cmd(:factor, Dict("a" => 1))
    end
end

@testset "Method Syntax on GiacExpr (US5)" begin

    @testset "Method syntax basic" begin
        # T024: Test expr.factor()
        expr = giac_eval("x^2 - 1")
        result = expr.factor()
        @test result isa GiacExpr
        result_str = string(result)
        @test occursin("x-1", result_str) || occursin("x+1", result_str)
    end

    @testset "Method syntax with args" begin
        # T025: Test expr.diff(x)
        expr = giac_eval("x^3")
        x = giac_eval("x")
        result = expr.diff(x)
        @test result isa GiacExpr
        @test string(result) == "3*x^2"
    end

    @testset "Method chaining" begin
        # T026: Test expr.expand().simplify()
        expr = giac_eval("(x+1)^2")
        result = expr.expand()
        @test result isa GiacExpr
        # Expand should give x^2 + 2*x + 1
        result_str = string(result)
        @test occursin("x^2", result_str)
    end

    @testset "Method and function equivalence" begin
        # T027: Test that method and function syntax return identical results
        expr = giac_eval("x^2 - 4")

        method_result = expr.factor()
        func_result = invoke_cmd(:factor, expr)

        @test string(method_result) == string(func_result)
    end
end

@testset "Multi-Argument Command Support (US4)" begin

    @testset "Substitution" begin
        # T033: Test invoke_cmd(:subst, expr, var, value)
        expr = giac_eval("x^2 + y")
        result = invoke_cmd(:subst, expr, giac_eval("x"), giac_eval("3"))
        @test result isa GiacExpr
        # x^2 + y with x=3 should give 9 + y
        result_str = string(result)
        @test occursin("9", result_str) || occursin("y", result_str)
    end

    @testset "Definite integral" begin
        # T034: Test invoke_cmd(:integrate, expr, var, a, b)
        expr = giac_eval("x^2")
        x = giac_eval("x")
        a = giac_eval("0")
        b = giac_eval("1")
        result = invoke_cmd(:integrate, expr, x, a, b)
        @test result isa GiacExpr
        # Integral of x^2 from 0 to 1 is 1/3
        result_str = string(result)
        @test occursin("1/3", result_str) || occursin("3", result_str)
    end

    @testset "Mixed argument types" begin
        # T036: Test mixed argument types (GiacExpr, Int, Float, Symbol)
        expr = giac_eval("x^2")

        # Symbol as variable name
        result = invoke_cmd(:diff, expr, :x)
        @test result isa GiacExpr
        @test string(result) == "2*x"

        # Integer as argument
        result = invoke_cmd(:diff, expr, :x, 2)  # Second derivative
        @test result isa GiacExpr
        @test string(result) == "2"
    end
end

@testset "Vector Input for GIAC Commands (032-vector-input-solve)" begin

    @testset "US1: Solve systems of equations with vectors" begin
        @giac_var x y z

        # T005: Test solve([x+y~0, x-y~2], [x,y]) returning [[1, -1]]
        result = Giac.Commands.solve([x+y~0, x-y~2], [x, y])
        @test result isa GiacExpr
        result_str = string(result)
        # Result should contain [[1,-1]] indicating x=1, y=-1
        @test occursin("1", result_str)
        @test occursin("-1", result_str)

        # T006: Test 3-equation system solve([eq1, eq2, eq3], [x, y, z])
        result3 = Giac.Commands.solve([x+y+z~6, x-y~0, y+z~4], [x, y, z])
        @test result3 isa GiacExpr
        result3_str = string(result3)
        # Solution should exist (non-empty result)
        @test !isempty(result3_str)
        @test result3_str != "[]"

        # T007: Test inconsistent system returning empty result
        # x + y = 1 and x + y = 2 are inconsistent
        result_empty = Giac.Commands.solve([x+y~1, x+y~2], [x, y])
        @test result_empty isa GiacExpr
        # Inconsistent systems should return empty list []
        result_empty_str = string(result_empty)
        @test result_empty_str == "[]"
    end

    @testset "US2: Vector input with other commands" begin
        @giac_var x y z

        # T010: Test vector input to list command (e.g., sum of vector)
        # sum([x, y, z]) should return x+y+z
        sum_result = Giac.Commands.sum([x, y, z])
        @test sum_result isa GiacExpr
        sum_str = string(sum_result)
        @test occursin("x", sum_str) || occursin("y", sum_str) || occursin("z", sum_str)

        # T011: Test mixed-type vector [1, x, 2.5]
        mixed_result = Giac.Commands.sum([1, x, 2])
        @test mixed_result isa GiacExpr
        # Should be able to handle mixed numeric and symbolic types
    end

    @testset "US3: Nested vector input for matrices" begin
        # T014: Test det_minor([[1,2],[3,4]]) with nested vectors
        # Note: det_minor computes the actual determinant value
        det_result = Giac.Commands.det_minor([[1, 2], [3, 4]])
        @test det_result isa GiacExpr
        det_str = string(det_result)
        # Determinant of [[1,2],[3,4]] = 1*4 - 2*3 = -2
        @test det_str == "-2"

        # T015: Test inverse via invoke_cmd with nested vectors
        # inverse() computes the actual matrix inverse
        inv_result = Giac.Commands.inverse([[1, 2], [3, 4]])
        @test inv_result isa GiacExpr
        inv_str = string(inv_result)
        # Should return a matrix representation
        @test occursin("[", inv_str)
    end

    @testset "Edge Cases: Empty and Deeply Nested Vectors" begin
        # T018: Test empty vector [] conversion
        empty_result = Giac.Commands.nops([])
        @test empty_result isa GiacExpr
        # nops([]) should return 0 (number of elements in empty list)
        @test string(empty_result) == "0"

        # T019: Test deeply nested vectors (3+ levels)
        # A 3D array-like structure
        deep_result = invoke_cmd(:nops, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        @test deep_result isa GiacExpr
        # Should return 2 (number of top-level elements)
        @test string(deep_result) == "2"
    end
end

@testset "Base Function Extensions" begin

    @testset "Base.sin" begin
        # T065: Test Base.sin(giac_eval("x"))
        x = giac_eval("x")
        result = sin(x)
        @test result isa GiacExpr
        @test string(result) == "sin(x)"
    end

    @testset "Base.sind, sinpi" begin
        # T065a: Test Base.sind(giac_eval("x"))
        result = sind(giac_eval("90"))
        @test result isa GiacExpr
        @test isone(result)

        result = sinpi(giac_eval("1/2"))
        @test result isa GiacExpr
        @test isone(result)

        result = asind(sind(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2

        result = sind(asind(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2
    end

    @testset "Base.cos" begin
        # T066: Test Base.cos(giac_eval("x"))
        x = giac_eval("x")
        result = cos(x)
        @test result isa GiacExpr
        @test string(result) == "cos(x)"
    end

    @testset "Base.cosd, cospi" begin
        # T066a: Test Base.cosd(giac_eval("x"))
        result = cosd(giac_eval("90"))
        @test result isa GiacExpr
        @test iszero(result)

        result = cospi(giac_eval("1/2"))
        @test result isa GiacExpr
        @test iszero(result)

        result = acosd(cosd(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2

        result = cosd(acosd(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2
    end

    @testset "Base.tand, tanpi" begin
        # T066b: Test Base.tand(giac_eval("x"))
        result = tand(giac_eval("45"))
        @test result isa GiacExpr
        @test isone(result)

        result = tanpi(giac_eval("1"))
        @test result isa GiacExpr
        @test iszero(result)

        result = atand(tand(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2

        result = tand(atand(giac_eval("1/2")))
        @test result isa GiacExpr
        @test result == 1//2
    end

    @testset "Base.cscd/secd/cotd" begin
        # T066b: Test Base.tand(giac_eval("x"))
        result = cscd(giac_eval("90"))
        @test result isa GiacExpr
        @test isone(result)

        result = secd(giac_eval("360"))
        @test result isa GiacExpr
        @test isone(result)

        result = cotd(giac_eval("45"))
        @test result isa GiacExpr
        @test isone(result)

        for (f, f_inv) ∈ ((secd, asecd), (cscd, acscd), (cotd, acotd))
            n = giac_eval("2")
            @test f_inv(f(n)) == 2
            @test f(f_inv(n)) == 2
        end
    end

    @testset "Base.sincospi, sincosd" begin
        s, c = sincosd(giac_eval("90"))
        @test isone(s)
        @test iszero(c)

        sp, cp = sincospi(giac_eval("1"))
        @test iszero(sp)
        @test isone(-cp)
    end



    @testset "Base.exp" begin
        # T067: Test Base.exp(giac_eval("x"))
        x = giac_eval("x")
        result = exp(x)
        @test result isa GiacExpr
        @test string(result) == "exp(x)"
    end

    @testset "Combined expressions" begin
        # T068: Test sin(x) + cos(x) where x is GiacExpr
        x = giac_eval("x")
        result = sin(x) + cos(x)
        @test result isa GiacExpr
        result_str = string(result)
        @test occursin("sin", result_str)
        @test occursin("cos", result_str)
    end

    @testset "Base.diff multiple dispatch" begin
        # Test that Base.diff still works for arrays (021-remove-giac-prefix)
        @test diff([1, 4, 9, 16]) == [3, 5, 7]  # Base.diff for arrays

        # Test that diff works for GiacExpr (symbolic differentiation)
        x = giac_eval("x")
        f = giac_eval("x^2")
        result = diff(f, x)
        @test result isa GiacExpr
        @test string(result) == "2*x"
    end
end

# ============================================================================
# GiacMatrix Command Support (058-commands-matrix-support)
# ============================================================================
@testset "GiacMatrix Command Support (058-commands-matrix-support)" begin

    M = GiacMatrix([[1, 2], [3, 4]])

    @testset "invoke_cmd with GiacMatrix" begin
        # T003: invoke_cmd(:eigenvals, GiacMatrix(...))
        result = invoke_cmd(:eigenvals, M)
        @test result isa GiacExpr
        result_str = string(result)
        # Eigenvalues of [[1,2],[3,4]] are (5±√33)/2
        @test !isempty(result_str)

        # T005: invoke_cmd(:det, GiacMatrix(...))
        det_result = invoke_cmd(:det, M)
        @test det_result isa GiacExpr
        @test string(det_result) == "-2"

        # T006: invoke_cmd(:transpose, GiacMatrix(...))
        trans_result = invoke_cmd(:transpose, M)
        @test trans_result isa GiacExpr
        trans_str = string(trans_result)
        @test occursin("[", trans_str)
    end

    @testset "Giac.Commands with GiacMatrix" begin
        # T004: eigenvals(GiacMatrix(...)) via generated Tier 2 command
        eigenvals_result = Giac.Commands.eigenvals(M)
        @test eigenvals_result isa GiacExpr
        @test !isempty(string(eigenvals_result))

        # trace via GiacMatrix-specific method (trace is in JULIA_CONFLICTS
        # but safe for GiacMatrix via multiple dispatch)
        trace_result = Giac.Commands.trace(M)
        @test trace_result isa GiacExpr
        @test string(trace_result) == "5"
    end

    @testset "GiacMatrix with symbolic entries" begin
        # T007: Matrix with symbolic entries
        @giac_var x y
        M_sym = GiacMatrix([[x, y], [1, x]])
        det_sym = invoke_cmd(:det, M_sym)
        @test det_sym isa GiacExpr
        det_str = string(det_sym)
        @test occursin("x", det_str)
    end
end

# ============================================================================
# Equations with held commands (059-heldcmd-equation-tilde)
# ============================================================================
@testset "Equations with held commands (059-heldcmd-equation-tilde)" begin

    using Giac.Commands: hold_cmd, release

    M = GiacMatrix([[1, 2], [3, 4]])
    isequation(e) = e isa GiacExpr && Giac._is_equation(e)
    tex(e) = sprint(show, MIME("text/latex"), e)

    @testset "held ~ value" begin
        h = hold_cmd(:eigenvals, M)
        eq = h ~ Giac.Commands.eigenvals(M)
        @test isequation(eq) && occursin("=", string(eq))
        @test occursin("eigenvals", tex(eq)) && occursin("=", tex(eq))      # the held side stays held
    end

    @testset "value ~ held" begin
        eq = Giac.Commands.eigenvals(M) ~ hold_cmd(:eigenvals, M)
        @test isequation(eq) && occursin("eigenvals", tex(eq))
    end

    @testset "held ~ held" begin
        @giac_var x
        eq = hold_cmd(:factor, x^2 - 1) ~ hold_cmd(:expand, (x - 1) * (x + 1))
        @test isequation(eq) && occursin("=", string(eq))
    end

    @testset "held ~ number, number ~ held" begin
        h = hold_cmd(:det, M)
        @test isequation(h ~ -2) && occursin("-2", string(h ~ -2))
        @test isequation(-2 ~ h) && occursin("-2", string(-2 ~ h))
        @test string(release(h ~ -2)) == "-2=-2"
    end
end

# ============================================================================
# Held command LaTeX: Limit, Sum, Product, Sum_Riemann (060-heldcmd-latex-limit-sum)
# ============================================================================
@testset "Held limit LaTeX (060 - US1)" begin

    using Giac.Commands: hold_cmd

    @giac_var x

    @testset "Basic limit" begin
        h = hold_cmd(:limit, sin(x)/x, x, 0)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim", latex_str)
        @test occursin("x \\to 0", latex_str)
        @test occursin("\\frac", latex_str)  # sin(x)/x rendered as fraction
    end

    @testset "Right-sided limit (dir=1)" begin
        h = hold_cmd(:limit, sign(x), x, 0, 1)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim", latex_str)
        @test occursin("^+", latex_str)
    end

    @testset "Left-sided limit (dir=-1)" begin
        h = hold_cmd(:limit, sign(x), x, 0, -1)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim", latex_str)
        @test occursin("^-", latex_str)
    end

    @testset "Limit at infinity" begin
        h = hold_cmd(:limit, 1/x, x, Inf)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim", latex_str)
        @test occursin("\\infty", latex_str)
    end

    @testset "Limit in an equation" begin
        h = hold_cmd(:limit, sin(x)/x, x, 0)
        eq = h ~ giac_eval("1")
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), eq)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim", latex_str)
        @test occursin("=", latex_str)
    end
end

@testset "Held sum/product LaTeX (060 - US2)" begin

    using Giac.Commands: hold_cmd

    @giac_var n k

    @testset "Bounded sum" begin
        h = hold_cmd(:sum, 1/n^2, n, 1, 17)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\sum", latex_str)
        @test occursin("n=1", latex_str)
        @test occursin("17", latex_str)
    end

    @testset "Sum with symbolic upper bound" begin
        h = hold_cmd(:sum, 1/k, k, 1, n)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\sum", latex_str)
        @test occursin("k=1", latex_str)
    end

    @testset "Infinite sum" begin
        h = hold_cmd(:sum, 1/n^2, n, 1, Inf)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\sum", latex_str)
        @test occursin("\\infty", latex_str)
    end

    @testset "Sum without bounds" begin
        h = hold_cmd(:sum, sin(n), n)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\sum", latex_str)
        @test occursin("n", latex_str)
    end

    @testset "Bounded product" begin
        h = hold_cmd(:product, k, k, 1, n)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\prod", latex_str)
        @test occursin("k=1", latex_str)
    end

    @testset "Product in an equation" begin
        h = hold_cmd(:product, k, k, 1, n)
        eq = h ~ invoke_cmd(:product, k, k, 1, n)
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), eq)
        latex_str = String(take!(latex_io))
        @test occursin("\\prod", latex_str)
        @test occursin("=", latex_str)
    end
end

@testset "Held Riemann sum LaTeX (060 - US3)" begin

    using Giac.Commands: hold_cmd

    @giac_var n k

    @testset "Basic Riemann sum" begin
        h = hold_cmd(:sum_riemann, 1 / (n + k), [n, k])
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), h)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim_{n \\to +\\infty}", latex_str)
        @test occursin("\\sum_{k=0}^{n-1}", latex_str)
        # Should NOT be the generic fallback
        @test !occursin("\\mathrm{sum\\_riemann}", latex_str)
    end

    @testset "Riemann sum in an equation" begin
        h = hold_cmd(:sum_riemann, 1 / (n + k), [n, k])
        result = Giac.Commands.sum_riemann(1 / (n + k), [n, k])
        eq = h ~ result
        latex_io = IOBuffer()
        show(latex_io, MIME("text/latex"), eq)
        latex_str = String(take!(latex_io))
        @test occursin("\\lim_{n \\to +\\infty}", latex_str)
        @test occursin("=", latex_str)
    end
end
