# Held commands: hold_cmd builds an unevaluated command as an expression, written in its notation;
# release evaluates it

@testset "Held commands" begin
    @giac_var x t s n z a
    f = x^2
    F = 1 / (s + 1)
    L(e) = Giac.giac_latex(e)

    @testset "a held command is an expression" begin
        h = hold_cmd(:integrate, x, x)
        @test h isa GiacExpr && Giac.symb_funcname(h) == "integrate"
        @test string.(Giac.arguments(h)) == ["x", "x"]
        @test Giac.symb_funcname(hold_cmd(:ifactor, 120)) == "ifactor"
        @test string(hold_cmd(:rand)) == "rand()"
        @test string(hold_cmd(:diff, x, :x)) == "diff(x,x)"
        @test string(hold_cmd(:factor, "x^2-1")) == "factor(x^2-1)"
        @test string(hold_cmd(:sum, [1, 2, 3])) == "sum([1,2,3])"
        @test string(hold_cmd("integrate", x, x)) == "integrate(x,x)"
        if !isempty(Giac.VALID_COMMANDS)
            @test_throws GiacError hold_cmd(:this_command_does_not_exist_xyz)
        end
    end

    @testset "written in its notation" begin
        @test L(hold_cmd(:integrate, f, x)) == "\\int x^{2} \\, \\mathrm{d}x"
        @test L(hold_cmd(:integrate, f, x, 0, 1)) == "\\int_{0}^{1} x^{2} \\, \\mathrm{d}x"
        @test L(hold_cmd(:diff, f, x)) == "\\frac{\\mathrm{d}}{\\mathrm{d} x}\\left(x^{2}\\right)"
        @test L(hold_cmd(:diff, f, x, 2)) == "\\frac{\\mathrm{d}^{2}}{\\mathrm{d} x^{2}}\\left(x^{2}\\right)"
        @test L(hold_cmd(:laplace, exp(-a * t), t, s)) == "\\mathcal{L}\\left\\{e^{-a t}\\right\\}(s)"
        @test occursin("\\mathcal{L}^{-1}", L(hold_cmd(:invlaplace, F, s, t)))
        @test occursin("\\mathcal{Z}\\left\\{", L(hold_cmd(:ztrans, f, n, z)))
        @test occursin("\\mathcal{Z}^{-1}", L(hold_cmd(:invztrans, F, z, n)))
        @test L(hold_cmd(:limit, sin(x) / x, x, 0)) == "\\lim_{x \\to 0} \\frac{\\sin x}{x}"
        @test L(hold_cmd(:sum, 1 / n^2, n, 1, Inf)) == "\\sum_{n=1}^{\\infty} \\frac{1}{n^{2}}"
        @test L(hold_cmd(:factor, x^2 - 1)) == "\\mathrm{factor}\\left(x^{2} - 1\\right)"
    end

    @testset "in arithmetic, equations and substitutions" begin
        h = hold_cmd(:integrate, f, x)
        @test L(h ~ release(h)) == "\\int x^{2} \\, \\mathrm{d}x = \\frac{x^{3}}{3}"
        @test L(2 * hold_cmd(:limit, sin(x) / x, x, 0) + 1) == "2 \\lim_{x \\to 0} \\frac{\\sin x}{x} + 1"
        S = Giac.substitute(hold_cmd(:sum, a * n, n, 1, 3), a => 2)
        @test string(S) == "sum(2*n,n,1,3)" && string(release(S)) == "12"
    end

    @testset "release" begin
        @test string(release(hold_cmd(:factor, x^2 - 1))) == string(invoke_cmd(:factor, x^2 - 1))
        @test string(release(hold_cmd(:integrate, x, x))) == "x^2/2"
        @test string(release(hold_cmd(:ifactor, 120))) == "2^3*3*5"        # GIAC's form, as it returns it
        @test string(release(hold_cmd(:integrate, f, x) + x)) == "x^3/3+x"  # inside an expression
        @test string(release(x + 1)) == "x+1"
    end
end
