# Roots of products: taken apart over factors known not to be negative

@testset "Roots" begin
    @giac_var m::positive ω::positive a::nonnegative x
    N(v) = convert(GiacExpr, v)
    @test string(sqrt(m^2 * ω^2)) == "m*ω"
    @test string(sqrt(4 * m^2)) == "2*m" && string(sqrt(N(12) * m^2)) == "2*m*sqrt(3)"
    @test string(sqrt(N(8))) == "2*sqrt(2)" && string(sqrt(N(2))) == "sqrt(2)"
    @test string(sqrt(m^2 * x)) == "m*sqrt(x)" && string(sqrt(a^2 * m)) == "a*sqrt(m)"
    @test string(sqrt(x^2)) == "abs(x)" && string(sqrt(4 * x^2)) == "2*abs(x)"    # x may be negative
    @test string(sqrt(N(-1))) == "i"
    @test string((m^2 * ω^2)^(1 // 2)) == "m*ω"
    # what stays under the root is one radical above and one below
    @test Giac.giac_latex(sqrt(ω^2 * x^2 / (2m))) == "\\frac{\\omega \\left|x\\right|}{\\sqrt{2 m}}"
    @test string(sqrt(2m)) == "sqrt(2*m)" && string(sqrt(3a / (4m))) == "sqrt(3*a)/(2*sqrt(m))"
end
