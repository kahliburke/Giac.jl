# GIAC from several threads at once, with the garbage collector freeing expressions meanwhile

@testset "Threads" begin
    @giac_var x y

    @testset "concurrent arithmetic, rewriting and typesetting" begin
        work(k) = begin
            ok = 0
            for j in 1:60
                e = Giac.Commands.expand((x + k + j)^2 - x^2)
                ok += string(Giac.Commands.normal(e - (2 * (k + j) * x + (k + j)^2))) == "0"
                ok += !isempty(Giac.giac_latex(e + sin(y)^2 + cos(y)^2))
                ok += string(Giac.Rules.rewrite(sin(y)^2 + cos(y)^2, Giac.Rules.trig_simplify())) == "1"
                j % 20 == 0 && GC.gc(false)
            end
            ok
        end
        @test all(==(180), fetch.([Threads.@spawn(work(k)) for k in 1:8]))
    end

    @testset "display names declared from several threads" begin
        tasks = [Threads.@spawn begin
                     for j in 1:200
                         Giac._set_tex_name!("thr_$(k)_$(j)", "x_{$k,$j}")
                         Giac._tex_name("thr_$(k)_$(j)")
                     end
                 end for k in 1:8]
        foreach(fetch, tasks)
        @test all(Giac._tex_name("thr_$(k)_$(j)") == "x_{$k,$j}" for k in 1:8, j in 1:200)
    end

    @testset "freed objects are queued, then freed under the lock" begin
        for _ in 1:2000
            Giac.Commands.expand((x + 1)^3)
        end
        GC.gc()
        Giac.with_giac_lock(() -> nothing)          # empties the queue
        @test Giac._DOOMED_COUNT[] == 0
    end
end
