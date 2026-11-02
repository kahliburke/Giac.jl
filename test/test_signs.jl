# the sign of an expression, and simplification by it

using Nemo, JuMP, Clarabel

@testset "signs" begin
    @giac_var x y k::positive a::positive b::positive n::negative
    C = Giac.Commands

    @testset "by structure" begin
        @test sign_of(exp(x) + k) == :positive
        @test sign_of(-k^3 - exp(y)) == :negative
        @test sign_of(log(1 + k)) == :positive
        @test sign_of(x^2) == :nonnegative
        @test sign_of(x^2 + 1) == :positive
        @test sign_of(k * n) == :negative
        @test sign_of(n^2 * k) == :positive
        @test sign_of(abs(x)) == :nonnegative
        @test sign_of(abs(x) + k) == :positive
        @test sign_of(sqrt(k) * (x^2 + 1)) == :positive
        @test sign_of(x - y) == :unknown
        @test sign_of(convert(GiacExpr, 0)) == :zero
        @test sign_of(x; positive = [x]) == :positive
        @test sign_of(x * y; negative = [x, y]) == :positive
    end

    @testset "by positivity of the skeleton" begin
        # mixed coefficient signs, decided by a sum of squares
        @test sign_of(x^2 - 2x * y + y^2) == :nonnegative
        # positive in fact, but a sum of squares only shows it is not negative
        @test sign_of(a^2 - a * b + b^2) == :nonnegative
        # a function call stands for a variable of known sign
        @test sign_of(sin(x)^2 - 2sin(x) + 1) == :nonnegative
        @test sign_of(x^2 - 2x * y + y^2; sos = false) == :unknown
    end

    @testset "assumptions in force" begin
        @giac_var t
        @test sign_of(t) == :unknown
        assuming(t > 0) do
            @test sign_of(t + 1) == :positive
        end
    end

    @testset "simplifying by signs" begin
        @test string(simplify_signs(abs(a * b + 1))) == "a*b+1"
        @test string(simplify_signs(abs(-a - 1))) == "a+1"
        @test string(simplify_signs(max(a^2 + b^2, 2a * b))) == string(a^2 + b^2)
        @test string(simplify_signs(min(k, -k))) == string(-k)
        @test string(simplify_signs(C.invoke_cmd(:sign, -k - 1))) == "-1"
        @test string(simplify_signs(abs(x - y))) == string(abs(x - y))
        e = simplify_signs(sqrt((a - b)^2 + 1) * abs(a * b + 1))
        @test !occursin("abs", string(e))
    end
end
