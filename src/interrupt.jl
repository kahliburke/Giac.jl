# Stopping a GIAC computation. GIAC checks a flag of its own (`giac::ctrl_c`) in its long loops
# (Gröbner bases, factoring, integration) and gives up with "Stopped by user interruption" when
# it is set. A computation runs inside a C++ call, where Julia's own interrupts do not reach, so
# the flag is set from another thread.
#
# A wrapper that runs GIAC's computations in GC-safe regions (with `giac_interrupt`) lets the
# other threads collect garbage meanwhile. With an older one, a thread that needs to collect
# waits for the computation, and a timeout or cancel check due then acts only when it returns.

using Libdl

# whether the wrapper sets GIAC's flags itself, and runs computations GC-safe
_wrapper_interrupts() = isdefined(GiacCxxBindings, :giac_interrupt)

const _GIAC_FLAGS = Ref{Tuple{Ptr{UInt8}, Ptr{UInt8}}}((C_NULL, C_NULL))

# GIAC's `ctrl_c` and `interrupted` flags, looked up in the loaded libgiac (older wrappers)
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
function interrupt()
    _wrapper_interrupts() ? GiacCxxBindings.giac_interrupt() : unsafe_store!(_giac_flags()[1], 0x01)
    return nothing
end

function _clear_interrupt()
    _wrapper_interrupts() ? GiacCxxBindings.giac_clear_interrupt() : foreach(p -> unsafe_store!(p, 0x00), _giac_flags())
    return nothing
end

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
[`GiacTimeout`](@ref). Only this task's calls are stopped. Needs GIAC on a thread of its own, as
the timer runs beside the computation: Julia started with more than one thread (`julia -t 2`), or
DedicatedThreads loaded. With a wrapper older than the
one running GIAC GC-safe, a pending garbage collection makes the stop wait for the computation.

```julia
Giac.with_timeout(30) do
    Giac.Commands.gbasis(eqs, vars)
end
```
"""
function with_timeout(f, seconds::Real)
    _giac_threaded() || @warn "with_timeout needs GIAC on a thread of its own (Julia started with more than one thread, or DedicatedThreads loaded); the computation runs without a limit" maxlog = 1
    # a deadline the cancel watcher compares with the clock: Julia's timers need its event loop,
    # which may not be run while every other thread is idle. Inside another `with_timeout`, the
    # earlier of the two deadlines holds.
    deadline = min(time() + seconds, get(task_local_storage(), :giac_cancel, Inf))
    before, mode = _assigned_names(), string(giac_eval("complex_mode()"))
    try
        return task_local_storage(f, :giac_cancel, deadline)
    catch e
        # GIAC's own temporary assumptions and complex mode, such as simplify's on the variables it
        # works on, are left in place when it is stopped or fails
        foreach(n -> giac_eval("purge($n)"), setdiff(_assigned_names(), before))
        string(giac_eval("complex_mode()")) == mode || giac_eval("complex_mode($mode)")
        time() >= deadline && e isa InterruptException && throw(GiacTimeout(Float64(seconds)))
        rethrow()
    end
end

# the names GIAC has values or assumptions for
function _assigned_names()
    s = string(giac_eval("VARS()"))
    return Set(filter(!isempty, split(strip(s, ['[', ']']), ',')))
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
    own === nothing || push!(preds, () -> time() >= own)
    isempty(preds) && return nothing
    # a job is polled only on the GIAC thread's side; with one thread GIAC runs where it is called
    # and nothing can poll beside it
    _giac_threaded() || return nothing
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
    job.check !== nothing && (try Base.invokelatest(job.check) catch; false end) && (@atomic job.cancelled = true)
    @atomic(job.cancelled) && return false
    lock(_RUN_LOCK) do
        _clear_interrupt()
        _RUNNING[] = job
    end
    job.check === nothing || lock(() -> notify(_WATCH), _WATCH)
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
const _WATCH = Threads.Condition()          # notified when a job with a cancel check starts

# A tenth of a second's wait while `job` runs, that needs neither Julia's event loop, which may not
# be run while every other thread is idle and GIAC computes, nor holds up a garbage collection. It
# waits a millisecond at a time and gives way to other tasks between, as the thread it holds is
# shared: a GIAC call's caller may be waiting there for its reply. It ends when the job does.
function _nap(job)
    @static if VERSION >= v"1.12"
        for _ in 1:100
            @ccall gc_safe = true usleep(1_000::Cuint)::Cint
            yield()
            _RUNNING[] === job || break
        end
    else
        sleep(0.1)
    end
    return nothing
end

_watched(job) = job !== nothing && job.check !== nothing && !@atomic(job.cancelled)

# A task that polls the running job's cancel check while there is one, and otherwise waits for
# one to start. Checks run in the latest world: one may be a closure defined after the task
# started. Started the first time a check is in use, on the interactive threads when there are
# any: the GIAC task may hold the only thread of the default pool while it computes, and a watcher
# there would not run until the computation it is to stop had ended.
function _start_cancel_watcher()
    _WATCHER[] === nothing || return
    lock(_WATCHER_LOCK) do
        _WATCHER[] === nothing || return
        pool = Threads.nthreads(:interactive) > 0 ? :interactive : :default
        _WATCHER[] = Threads.@spawn pool while true
            lock(_WATCH) do
                while !_watched(_RUNNING[])
                    wait(_WATCH)
                end
            end
            job = _RUNNING[]
            _nap(job)
            _RUNNING[] === job && _watched(job) && (try Base.invokelatest(job.check) catch; false end) && _cancel_job!(job)
        end
    end
    return
end
