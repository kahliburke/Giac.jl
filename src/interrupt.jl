# Stopping a GIAC computation. GIAC checks a flag of its own (`giac::ctrl_c`) in its long loops
# (Gröbner bases, factoring, integration) and gives up with "Stopped by user interruption" when
# it is set. A computation runs inside a C++ call, where Julia's own interrupts do not reach, so
# the flag is set from another thread.
#
# Limitation: the wrapper does not mark GIAC's computations safe for Julia's garbage collector, so
# while one runs, any thread that needs to collect waits for it. A timeout or cancel check that is
# due while such a wait is on acts only when the computation returns.

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
[`GiacTimeout`](@ref). Only this task's calls are stopped. Needs Julia started with more than one
thread (`julia -t 2`), as the timer runs beside the computation. Julia cannot collect garbage
while GIAC computes, so if a collection is pending the stop waits for the computation.

```julia
Giac.with_timeout(30) do
    Giac.Commands.gbasis(eqs, vars)
end
```
"""
function with_timeout(f, seconds::Real)
    Threads.nthreads() > 1 || @warn "with_timeout needs Julia started with more than one thread; the computation runs without a limit" maxlog = 1
    expired = Threads.Atomic{Bool}(false)
    timer = Timer(_ -> (expired[] = true), seconds)
    try
        return task_local_storage(f, :giac_cancel, expired)
    catch e
        expired[] && e isa InterruptException && throw(GiacTimeout(Float64(seconds)))
        rethrow()
    finally
        close(timer)
    end
end

# ---------------------------------------------------------------------------------------------
# Cancelling GIAC work from a job runner
# ---------------------------------------------------------------------------------------------

"""
    Giac.cancel_checks

Functions a job runner registers so that cancelling a job stops the GIAC computation in it. Each
is called in the task that makes a GIAC call and returns `nothing`, or a function of no
arguments that says whether that task's work has been cancelled. While the call runs on the GIAC
thread, that function is polled, and when it says so the computation is interrupted and the call
throws an `InterruptException`. Needs Julia started with more than one thread. For Kaimon:

```julia
push!(Giac.cancel_checks, function ()
    id = string(get(task_local_storage(), :gate_request_id, ""))
    isempty(id) ? nothing : () -> KaimonGate.is_cancelled(; job_id = id)
end)
```
"""
const cancel_checks = Function[]

# whether the calling task's work is cancelled, as a function another task can poll; nothing
# when no check applies
function _cancel_predicate()
    own = get(task_local_storage(), :giac_cancel, nothing)       # with_timeout's
    own === nothing && isempty(cancel_checks) && return nothing
    preds = Any[p for p in (c() for c in cancel_checks) if p !== nothing]
    own === nothing || push!(preds, () -> own[])
    isempty(preds) && return nothing
    _start_cancel_watcher()
    return () -> any(p -> p(), preds)
end

# A call handed to the GIAC thread. Cancelling it stops it when it runs and skips it when it
# waits. The running job and GIAC's interruption flag change together under `_RUN_LOCK`, so an
# interruption meant for one job never reaches the next.
mutable struct _GiacJob
    check::Any                          # a function saying whether the caller's work is cancelled
    @atomic cancelled::Bool
    _GiacJob(check) = new(check, false)
end

const _RUNNING = Ref{Union{Nothing, _GiacJob}}(nothing)
const _RUN_LOCK = ReentrantLock()

# the job starts: false if it was cancelled before it could
function _start_job(job::_GiacJob)
    job.check !== nothing && (try job.check() catch; false end) && (@atomic job.cancelled = true)
    @atomic(job.cancelled) && return false
    lock(_RUN_LOCK) do
        _clear_interrupt()
        _RUNNING[] = job
    end
    return true
end

# the job ended: whether it was cancelled while it ran
function _finish_job(job::_GiacJob)
    lock(_RUN_LOCK) do
        _RUNNING[] = nothing
        _clear_interrupt()
    end
    return @atomic job.cancelled
end

function _cancel_job!(job::_GiacJob)
    @atomic job.cancelled = true
    lock(_RUN_LOCK) do
        _RUNNING[] === job && interrupt()
    end
    return
end

const _WATCHER = Ref{Union{Nothing, Task}}(nothing)
const _WATCHER_LOCK = ReentrantLock()

# a task that polls the running job's cancel check, started the first time a check is in use
function _start_cancel_watcher()
    _WATCHER[] === nothing || return
    lock(_WATCHER_LOCK) do
        _WATCHER[] === nothing || return
        _WATCHER[] = Threads.@spawn while true
            sleep(0.1)
            job = _RUNNING[]
            (job === nothing || job.check === nothing || @atomic(job.cancelled)) && continue
            (try job.check() catch; false end) && _cancel_job!(job)
        end
    end
    return
end
