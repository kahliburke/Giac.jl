# ---------------------------------------------------------------------------
# GiacDedicatedThreadsExt — GIAC on an OS thread of its own.
#
# Every GIAC call runs on one thread, in one task (see "The GIAC thread" in src/utils.jl).
# Without this extension that task is pinned to the last thread of the default pool, which other
# tasks share, and with a single thread there is none: calls run in place under a lock. With
# DedicatedThreads loaded, the task gets a pool of one thread that runs nothing else, whatever
# `-t` was.
#
# The thread is chosen when the task starts, at the first GIAC call. GIAC's state belongs to the
# thread it was made on, so a task already started stays where it is: load DedicatedThreads before
# the first computation (`using Giac, DedicatedThreads`).
# ---------------------------------------------------------------------------

module GiacDedicatedThreadsExt

using Giac
using DedicatedThreads: DedicatedPool, spawn

const POOL = Ref{Union{Nothing, DedicatedPool}}(nothing)

function _start(serve)
    POOL[] = DedicatedPool(1; name = :giac)
    return spawn(serve, POOL[])
end

function __init__()
    ccall(:jl_generating_output, Cint, ()) == 0 || return
    if Giac._GIAC_TASK[] !== nothing
        @warn "GIAC already runs on a thread of the default pool, so it stays there. Load DedicatedThreads before the first GIAC computation to give GIAC a thread of its own."
        return
    end
    Giac._GIAC_TASK_STARTER[] = _start
    return
end

end # module
