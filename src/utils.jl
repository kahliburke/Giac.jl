# Utilities for Giac.jl
# Thread safety, type conversion, and helper functions

"""
Global lock for serializing GIAC library calls when they run on the calling thread (see
`with_giac_lock`). GIAC is not thread-safe, so all calls must be serialized.
"""
const GIAC_LOCK = ReentrantLock()

# ---------------------------------------------------------------------------------------------
# The GIAC thread
# ---------------------------------------------------------------------------------------------
# GIAC keeps its state per OS thread: an assumption made from one thread isn't there for a call on
# another, and Julia moves tasks between threads. So with several threads every GIAC call runs on
# one of them, in a task pinned there that does one job at a time. That also serializes GIAC,
# which isn't thread-safe. With a single thread, or while precompiling, calls run where they are,
# under `GIAC_LOCK`.

const _GIAC_TASK = Ref{Union{Nothing, Task}}(nothing)
const _GIAC_INBOX = Channel{Tuple{Any, Any, Channel{Any}, Any}}(Inf)
const _GIAC_START_LOCK = ReentrantLock()
const _ACTIVE_FRAME = Ref{Any}(nothing)     # the frame of the task whose job the GIAC thread is running
const _INLINE_DEPTH = Ref(0)                # how deeply `GIAC_LOCK` is held when calls run in place

_on_giac_thread() = current_task() === _GIAC_TASK[]
_giac_threaded() = Threads.maxthreadid() > 1 && ccall(:jl_generating_output, Cint, ()) == 0

# The GIAC thread's task, started on first use on the last thread of the default pool
function _giac_task()
    t = _GIAC_TASK[]
    t === nothing || return t
    return lock(_GIAC_START_LOCK) do
        _GIAC_TASK[] === nothing || return _GIAC_TASK[]
        t = Task(_giac_serve)
        t.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), t, last(Threads.threadpooltids(:default)) - 1)
        _GIAC_TASK[] = t
        schedule(t)
        t
    end
end

# A job runs in the latest world: this task outlives the methods defined after it started (a
# REPL's closures, a Revise edit), which its own world couldn't call
function _giac_serve()
    for (f, frame, reply, job) in _GIAC_INBOX
        # a job cancelled while it waited is not run
        if !_start_job(job)
            put!(reply, (false, InterruptException()))
            continue
        end
        _ACTIVE_FRAME[] = frame
        result = try
            (true, Base.invokelatest(_run_giac_job, f))
        catch e
            (false, e)
        finally
            _ACTIVE_FRAME[] = nothing
        end
        # a job stopped because it was cancelled ends in an InterruptException
        _finish_job(job) && !first(result) && (result = (false, InterruptException()))
        put!(reply, result)
    end
end

function _run_giac_job(f)
    _free_doomed!()          # objects finalizers left to free, see wrapper.jl
    _sync_local_names!()
    return f()
end

# Each task waits for its replies on a channel of its own
_reply_channel() = get!(() -> Channel{Any}(1), task_local_storage(), :giac_reply)::Channel{Any}

"""
    with_giac_lock(f)

Run `f`, which calls into GIAC, on the GIAC thread and return its result (an exception it throws
is thrown here). Code already running there, a GIAC call made from inside another, runs at once.
The GIAC state of names declared in a local scope is made the calling task's first (see
`_sync_local_names!`).
"""
function with_giac_lock(f)
    _on_giac_thread() && return f()
    if !_giac_threaded()
        return lock(GIAC_LOCK) do
            _INLINE_DEPTH[] += 1
            try
                _free_doomed!()
                # an interruption meant for an earlier call does not stop this one
                _INLINE_DEPTH[] == 1 && (_sync_local_names!(); _clear_interrupt())
                f()
            finally
                _INLINE_DEPTH[] -= 1
            end
        end
    end
    _giac_task()
    reply = _reply_channel()
    job = _GiacJob(_cancel_predicate())
    put!(_GIAC_INBOX, (f, _current_frame(), reply, job))
    ok, r = try
        take!(reply)
    catch e
        # the reply is still to come, so this task's next call takes a channel of its own
        delete!(task_local_storage(), :giac_reply)
        if e isa InterruptException
            # an interrupt of the waiting task (Ctrl-C) cancels its job: stopped if it runs,
            # skipped if it waits, so the GIAC thread is free again
            _cancel_job!(job)
            try
                take!(reply)
            catch
            end
        end
        rethrow()
    end
    ok || throw(r)
    return r
end

# Note: to_julia, is_numeric, and is_symbolic have been moved to
# conversion.jl and introspection.jl as part of feature 029-output-handling
