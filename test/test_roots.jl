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
end
