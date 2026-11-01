# express as a change of variables, and stopping a long GIAC computation

@testset "express and interruption" begin
    C = Giac.Commands
    @giac_var t m x1 x2 y u L p r a b

    @testset "affine definitions: a change of variables" begin
        D = C.discriminant(m * (t - y) * (t - u) - (t - x1) * (t - x2), t)
        e = express(D, L => x2 - x1, p => y - x1, r => u - x1)
        @test iszero(C.expand(e - ((p - r)^2 * m^2 - 2 * (p * L + r * L - 2 * p * r) * m + L^2)))
        @test !occursin("x1", string(e))                      # translation invariant: x₁ drops out
        @test string(express(a + b, L => a + b)) == "L"
        @test string(express(2a - 4, L => a - 2)) == "2*L"
        @test Giac._affine_change([L], [a * b], [a, b]) === nothing        # not affine: Gröbner
        @test Giac._affine_change([L, p], [a + b, 2a + 2b], [a, b]) === nothing   # dependent
    end

    @testset "a timeout stops GIAC" begin
        if Threads.nthreads() > 1
            # the Gröbner route to the same rewrite is slow enough to be stopped
            D = C.discriminant(m * (t - y) * (t - u) - (t - x1) * (t - x2), t)
            vals = [x2 - x1, y - x1, u - x1]
            order = vcat([x2, x1, y, u], [L, p, r])
            slow() = C.invoke_cmd(:greduce, C.numer(C.normal(D)),
                                  C.invoke_cmd(:gbasis, GiacExpr[v - n for (n, v) in zip([L, p, r], vals)], order, giac_eval("plex")), order)
            @test_throws Giac.GiacTimeout Giac.with_timeout(slow, 1)
            @test string(C.factor(x1^2 - 1)) == "(x1-1)*(x1+1)"            # GIAC works again after
            # interrupting the waiting task, as Ctrl-C does, stops the computation too
            if Giac._giac_threaded()
                job = Threads.@spawn try
                    slow()
                catch e
                    e
                end
                sleep(1)
                schedule(job, InterruptException(); error = true)
                @test fetch(job) isa InterruptException
                @test string(C.factor(x1^2 - 1)) == "(x1-1)*(x1+1)"
            end
        end
        @test Giac.with_timeout(() -> C.expand((a + b)^2), 10) == C.expand((a + b)^2)
    end
end
