# Indefinite integrals of exact derivatives of expressions in unknown functions, where GIAC's own
# integrate returns infinity

@testset "Integrating exact derivatives" begin
    @giac_var t η(t) p(t)
    C = Giac.Commands
    D(f) = C.diff(f, t)
    @test string(C.integrate(D(η * p), t)) == "p(t)*η(t)"
    @test string(C.integrate(D(η) * t + η, t)) == "t*η(t)"
    @test string(C.integrate(D(p) * sin(t) + p * cos(t), t)) == "sin(t)*p(t)"
    @test string(C.integrate(D(η^2 * p), t)) == "p(t)*η(t)^2"
    @test string(C.integrate(D(η * p) + t^2, t)) == "p(t)*η(t)+t^3/3"   # the rest to GIAC
    @test string(C.integrate(D(η) * p, t)) == "integrate(diff(η(t),t)*p(t),t)"   # no closed form
    @test string(C.integrate(t^2, t)) == "t^3/3"
end
