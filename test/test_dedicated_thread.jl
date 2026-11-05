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
    slow() = giac_eval("gbasis([x^7+y^6+z^5+w^3-1, x^5+y^7+z^6+w^4-1, x^6+y^5+z^7+w^5-1, x+y+z+w^2-2], [x,y,z,w], plex)")
    stopped = try
        Giac.with_timeout(slow, 1); false
    catch e
        e isa Giac.GiacTimeout
    end
    print(simplified, " ", on_pool, " ", stopped, " ", Threads.nthreads())
    """
    out = read(`$(Base.julia_cmd()) -t 1 --startup-file=no -e $script`, String)
    # an assumption holds, on the pool's thread, and a timeout stops GIAC, beside one thread
    @test out == "k true true 1"
end
