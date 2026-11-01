# ---------------------------------------------------------------------------
# GiacKaimonGateExt: cancelling a Kaimon job stops its GIAC computation.
#
# Loaded with KaimonGate. Registers a cancel check (see `Giac.cancel_checks`) that reads the job
# id KaimonGate keeps in the evaluating task, so `cancel_eval` interrupts the GIAC call that job
# is waiting on.
# ---------------------------------------------------------------------------

module GiacKaimonGateExt

using Giac
using KaimonGate

function _kaimon_cancelled()
    id = string(get(task_local_storage(), :gate_request_id, ""))
    return isempty(id) ? nothing : () -> KaimonGate.is_cancelled(; job_id = id)
end

function __init__()
    any(c -> c === _kaimon_cancelled, Giac.cancel_checks) || push!(Giac.cancel_checks, _kaimon_cancelled)
    return nothing
end

end # module GiacKaimonGateExt
