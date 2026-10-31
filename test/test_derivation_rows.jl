# The rows of a derivation: naming a step, functions, implications with conditions

@testset "Derivation rows" begin
    @giac_var x k::positive a::positive A B n ħ::positive m::positive
    rows(d) = [l * " = " * r for (l, r) in d.rows]

    @testset "name := expr" begin
        d = @derivation begin
            k2 := diff(k^3, k)
        end
        @test rows(d) == ["\\frac{\\mathrm{d}}{\\mathrm{d} k}\\left(k^{3}\\right) = 3 k^{2}"]
        @test string(k2) == "3*k^2"
    end

    @testset "implications and their conditions" begin
        # (a `where` clause is parsed from a string, as a file is: Slate's cell analysis rejects it)
        d = Core.eval(@__MODULE__, Meta.parse("""
        let
        @giac_var x k::positive a::positive A B n ħ::positive m::positive
        @derivation begin
            φ(x) = A * cos(k * x) + B * sin(k * x)
            φ(0) ~ 0 => A ~ 0
            φ(x) ~ φ(x)
            φ(a) ~ 0 => k ~ n * Giac.giac_eval("pi") / a where n::integer, n != 0
            E = ħ^2 * k^2 / (2m)
        end
        end"""))
        r = rows(d)
        @test r[1] == "\\varphi\\left(x\\right) = A \\cos\\left(k x\\right) + B \\sin\\left(k x\\right)"
        @test r[2] == "\\varphi\\left(0\\right) = A = 0"                       # its own result: stated once
        @test r[3] == "\\varphi\\left(x\\right) = B \\sin\\left(k x\\right)"    # A = 0 applied to φ
        @test r[4] == "\\varphi\\left(a\\right) = B \\sin\\left(k a\\right) = 0 \\quad\\Rightarrow\\quad k = \\frac{\\pi n}{a}, \\quad n \\in \\mathbb{Z}, n \\ne 0"
        @test r[5] == "E = \\frac{\\pi^{2} n^{2} \\hbar^{2}}{2 a^{2} m}"       # k = nπ/a applied
    end

    @testset "equations as values" begin
        @giac_var ω
        d = @derivation begin
            r := k ~ m * ω^2                  # a written equation: GIAC's `=`
            eq = k^2 ~ a                      # a bound equation shows as itself
        end
        @test rows(d) == ["k = m \\omega^{2}", "k^{2} = a"]
        @test string(r) == "k=m*ω^2"
    end

    @testset "a named rule is not applied" begin
        d = @derivation begin
            rule := k^2 ~ a => k ~ sqrt(a)
            k^2 + 1 ~ k^2 + 1
        end
        @test rows(d)[2] == "k^{2} + 1 = k^{2} + 1"
        @test string(substitute(k^3, rule)) == string(substitute(k^3, k => sqrt(a)))
    end

    @testset "conditions after a comma, without where" begin
        d = @derivation begin
            k^2 ~ a^2 => k ~ a * n, n::integer, n != 0
        end
        @test endswith(rows(d)[1], "k = a n, \\quad n \\in \\mathbb{Z}, n \\ne 0")
    end

    @testset "solving the condition for a name" begin
        @giac_var b::positive
        d = @derivation begin
            a * b^2 / 2 ~ 1 => b
            b^2 ~ b^2
        end
        @test endswith(rows(d)[1], "\\quad\\Rightarrow\\quad b = \\frac{\\sqrt{2}}{\\sqrt{a}}")
        @test rows(d)[2] == "b^{2} = \\frac{2}{a}"                  # applied from here on
    end

    @testset "conditions in brackets, and tags" begin
        imp = Giac._implication(Giac._rejoin_conditions(:(k ~ 0 => k ~ n where [n::integer, n::nonzero])))
        @test Giac._conditions(imp[2])[2] == Any[:(n::integer), :(n::nonzero)]
        @test Giac._tag_latex("n", :nonzero) == "n \\ne 0"
    end
end
