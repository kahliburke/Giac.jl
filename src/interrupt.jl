# Stopping a GIAC computation. GIAC checks a flag of its own (`giac::ctrl_c`) in its long loops
# (Gröbner bases, factoring, integration) and gives up with "Stopped by user interruption" when
# it is set. A computation runs inside a C++ call, where Julia's own interrupts do not reach, so
# the flag is set from another thread.

using Libdl

const _GIAC_FLAGS = Ref{Tuple{Ptr{UInt8}, Ptr{UInt8}}}((C_NULL, C_NULL))

# GIAC's `ctrl_c` and `interrupted` flags, looked up in the loaded libgiac
function _giac_flags()
    flags = _GIAC_FLAGS[]
    flags[1] == C_NULL || return flags
    lib = findfirst(l -> occursin(r"libgiac[.\d]*\.(dylib|so|dll)", basename(l)), Libdl.dllist())
    lib === nothing && error("libgiac is not loaded")
    h = Libdl.dlopen(Libdl.dllist()[lib])
    flags = (Ptr{UInt8}(Libdl.dlsym(h, :_ZN4giac6ctrl_cE)), Ptr{UInt8}(Libdl.dlsym(h, :_ZN4giac11interruptedE)))
    _GIAC_FLAGS[] = flags
    return flags
end

"""
    Giac.interrupt()

Stop the GIAC computation running now; it throws a `GiacError` saying it was stopped by user
interruption. Call it from another task on another thread: the computation itself holds its
thread until it returns. See [`Giac.with_timeout`](@ref) to stop one after a time.
"""
interrupt() = (unsafe_store!(_giac_flags()[1], 0x01); nothing)

_clear_interrupt() = (foreach(p -> unsafe_store!(p, 0x00), _giac_flags()); nothing)

"""
    GiacTimeout(seconds)

Thrown by [`Giac.with_timeout`](@ref) when the computation was stopped for taking longer.
"""
struct GiacTimeout <: Exception
    seconds::Float64
end
Base.showerror(io::IO, e::GiacTimeout) = print(io, "GiacTimeout: the GIAC computation was stopped after ", e.seconds, " s")

"""
    Giac.with_timeout(f, seconds)

Run `f()`, stopping a GIAC computation in it that is still running after `seconds`, with a
[`GiacTimeout`](@ref). Needs Julia started with more than one thread (`julia -t 2`), as the
timer runs beside the computation.

```julia
Giac.with_timeout(30) do
    Giac.Commands.gbasis(eqs, vars)
end
```
"""
function with_timeout(f, seconds::Real)
    Threads.nthreads() > 1 || @warn "with_timeout needs Julia started with more than one thread; the computation runs without a limit" maxlog = 1
    done = Threads.Atomic{Bool}(false)
    fired = Threads.Atomic{Bool}(false)
    watcher = Threads.@spawn begin
        deadline = time() + seconds
        while !done[] && time() < deadline
            sleep(min(0.05, max(deadline - time(), 0.001)))
        end
        done[] || (fired[] = true; interrupt())
    end
    try
        return f()
    catch e
        fired[] && throw(GiacTimeout(Float64(seconds)))
        rethrow()
    finally
        done[] = true
        wait(watcher)
        fired[] && _clear_interrupt()
    end
end
