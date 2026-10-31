# Roots of products: taken apart over factors known not to be negative

@testset "Roots" begin
    @giac_var m::positive ω::positive a::nonnegative x
    N(v) = convert(GiacExpr, v)
    @test string(sqrt(m^2 * ω^2)) == "m*ω"
    @test string(sqrt(4 * m^2)) == "2*m" && string(sqrt(N(12) * m^2)) == "2*sqrt(3)*m"
    @test string(sqrt(N(8))) == "2*sqrt(2)" && string(sqrt(N(2))) == "sqrt(2)"
    @test string(sqrt(m^2 * x)) == "m*sqrt(x)" && string(sqrt(a^2 * m)) == "a*sqrt(m)"
    @test string(sqrt(x^2)) == "abs(x)" && string(sqrt(4 * x^2)) == "2*abs(x)"    # x may be negative
    @test string(sqrt(N(-1))) == "i"
    @test string((m^2 * ω^2)^(1 // 2)) == "m*ω"
    # roots stay apart in the expression, so they cancel; they are written under one radical
    L = Giac.giac_latex
    @test string(sqrt(2m) * sqrt(m)) == "sqrt(2)*m"
    @test L(sqrt(ω^2 * x^2 / (2m))) == "\\frac{\\omega \\left|x\\right|}{\\sqrt{2 m}}"
    @test L(sqrt(8m^3)) == "2 m \\sqrt{2 m}"
    @test L(sqrt(2 / m) * sin(x)) == "\\sqrt{\\frac{2}{m}} \\sin x"          # a number over a root: one root
    @test L(sqrt(3a / (4m))) == "\\frac{1}{2} \\sqrt{\\frac{3 a}{m}}"
    @test L(sqrt(x / 2)) == "\\frac{\\sqrt{x}}{\\sqrt{2}}"                    # x may be negative
    @test L(Giac.Commands.solve(x^2 * m / 2 ~ 1, x)) == "\\left\\{\\sqrt{\\frac{2}{m}},\\; -\\sqrt{\\frac{2}{m}}\\right\\}"
end
