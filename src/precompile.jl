# Work done while the package is compiled. GIAC's commands become functions of `Commands` then, so
# that loading the package does not make them again, and the calls a session starts with are
# compiled. GIAC objects made meanwhile are let go before the image is written, as their pointers
# mean nothing in another process.

using PrecompileTools: @compile_workload

function _precompile_workload()
    x, y = giac_eval("x"), giac_eval("y")
    f = (x^2 + 2x * y + 1) / (x - y) + sin(x)^2 - exp(-x) * cos(y)
    string(f), sprint(show, MIME"text/plain"(), f), sprint(show, MIME"text/latex"(), f)
    _node(f)
    C = Commands
    C.simplify(C.diff(f, x))
    C.integrate(1 / (x^2 + 1), x)
    C.factor(x^4 - y^4), C.expand((x + y)^3), C.solve(x^2 - 2, x)
    to_julia(giac_eval("3/4")), to_julia(giac_eval("2.5"))
    M = GiacMatrix([x 1; 0 y])
    LinearAlgebra.det(M), M * M
    return nothing
end

# Let go of every GIAC object the package's tables hold, and free them
function _forget_giac_objects!()
    empty!(_IDENTIFIERS)
    empty!(_gen_objects)
    _gen_counter[] = 0
    empty!(_context_objects)
    _context_counter[] = 1
    _cxxwrap_context[] = nothing
    # the constants as trees until `__init__` makes them again
    for (r, name) in ((Constants._pi, "pi"), (Constants._e, "e"), (Constants._i, "i"))
        r[] = GiacExpr(SymNode(name))
    end
    Constants._giac_constants[] = (Constants._pi[], Constants._e[], Constants._i[])
    GC.gc(true)
    _free_doomed!()
    _initialized[] = false
    # addresses in this process
    _GIAC_FLAGS[] = (C_NULL, C_NULL)
    for r in (_giac_lib_handle, _cxxwrap_lib_handle, _wrapper_lib_handle,
              GiacCxxBindings._giac_handle, GiacCxxBindings._cxxwrap_handle)
        r[] = C_NULL
    end
    return nothing
end

# The globals of the package's modules that hold a GIAC object or an address, which would be stale
# when loaded. A constant address is the package's own (a sentinel); CxxWrap's pointer table is
# filled again when it loads.
function _giac_objects_in_globals()
    found = String[]
    seen = IdSet{Any}()
    holds(v, depth) = begin
        depth > 8 && return false
        v isa Ptr && return depth > 0 && v != C_NULL
        v isa Tuple && return any(e -> holds(e, depth + 1), v)
        # a Regex is compiled again when loaded
        (isbits(v) || v isa Union{Module, Function, Type, Symbol, String, Regex}) && return false
        v in seen && return false
        push!(seen, v)
        v isa GiacExpr && return getfield(v, :ptr) != C_NULL || getfield(v, :replaced) != C_NULL ||
                                holds(getfield(v, :node), depth + 1) || holds(getfield(v, :args), depth + 1)
        v isa GiacMatrix && return getfield(v, :ptr) != C_NULL || holds(getfield(v, :_source), depth + 1)
        v isa GiacContext && return getfield(v, :ptr) ∉ (C_NULL, _SHARED_CONTEXT_PTR)
        parentmodule(typeof(v)) === GiacCxxBindings && return true
        v isa AbstractDict && return any(p -> holds(p.first, depth + 1) || holds(p.second, depth + 1), v)
        v isa Union{AbstractArray, Tuple, AbstractSet} && return any(e -> holds(e, depth + 1), v)
        v isa Base.RefValue && return isassigned(v) && holds(v[], depth + 1)
        T = typeof(v)
        isstructtype(T) || return false
        return any(i -> isdefined(v, i) && holds(getfield(v, i), depth + 1), 1:fieldcount(T))
    end
    modules = [Giac]
    while !isempty(modules)
        m = pop!(modules)
        for name in names(m; all = true)
            isdefined(m, name) && !startswith(string(name), "__cxxwrap") || continue
            v = getfield(m, name)
            if v isa Module
                v !== m && parentmodule(v) === m && push!(modules, v)
            elseif holds(v, 0)
                push!(found, "$m.$name")
            end
        end
    end
    return found
end

if ccall(:jl_generating_output, Cint, ()) == 1 && GiacCxxBindings._have_library
    Base.CoreLogging.with_logger(__init__, Base.CoreLogging.NullLogger())
    @compile_workload begin
        _precompile_workload()
        # the GIAC thread's side, which compiling runs without (calls run in place then)
        precompile(_giac_serve, ())
    end
    _forget_giac_objects!()
    let left = _giac_objects_in_globals()
        isempty(left) || error("GIAC objects left in globals of the package image: ", join(left, ", "))
    end
end
