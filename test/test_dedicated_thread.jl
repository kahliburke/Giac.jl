# GIAC on an OS thread of its own, with DedicatedThreads loaded (Julia 1.12 and later). It has to
# be loaded before the first GIAC computation, so this runs in a fresh process with one thread, in
# a copy of the test environment with DedicatedThreads added.

@testset "a dedicated GIAC thread" begin
    env = mktempdir()
    cp(dirname(Base.active_project()), env; force = true)
    script = """
    import Pkg
    Pkg.activate($(repr(env)); io = devnull)
    Pkg.add("DedicatedThreads"; io = devnull)
    using Giac, DedicatedThreads
    pool = Base.get_extension(Giac, :GiacDedicatedThreadsExt).POOL
    @giac_var k::positive
    simplified = string(Giac.Commands.simplify(sqrt(k^2)))
    on_pool = Threads.threadid(Giac._GIAC_TASK[]) in DedicatedThreads.threadids(pool[])
    print(simplified, " ", on_pool, " ", Threads.nthreads())
    """
    out = read(`$(Base.julia_cmd()) -t 1 --startup-file=no -e $script`, String)
    @test out == "k true 1"          # an assumption holds, on the pool's thread, beside one thread
end
