# GIAC's session settings for the length of a computation. The wrapper applies an override at the
# start of each call into GIAC and puts the session's own value back at its end, whatever happens in
# between, so a setting holds for exactly the calls made inside `with_settings`.

_wrapper_overrides() = isdefined(GiacCxxBindings, :set_setting_override)

"""
    Giac.with_settings(f, settings...)

Run `f` with the GIAC session `settings` in force, each a pair of a setting's name and its value:
`:keep_acosh_asinh => true` (acosh and asinh are not rewritten as logarithms),
`:complex_mode => true`, `:decimal_digits => 30`, `:angle_mode => 1` (degrees) and the others GIAC
keeps per session. The session's own settings are in force before and after, even when `f` throws.
Other tasks' GIAC calls wait until `f` returns. With a wrapper library that has no overrides, `f`
runs with the session's settings.

```julia
Giac.with_settings(:keep_acosh_asinh => true) do
    Giac.Commands.integrate(giac_eval("acosh(x)^2"), giac_eval("x"))
end
```
"""
function with_settings(f, settings::Pair...)
    _wrapper_overrides() || (@warn "this wrapper library cannot override GIAC's settings; running with the session's" maxlog = 1;
                             return f())
    return with_giac_lock() do
        set = String[]
        try
            for (name, value) in settings
                GiacCxxBindings.set_setting_override(String(name), Int(value)) ||
                    throw(ArgumentError("$name is not a GIAC session setting"))
                push!(set, String(name))
            end
            f()
        finally
            foreach(GiacCxxBindings.clear_setting_override, set)
        end
    end
end

"""
    Giac.setting(name) -> Int

The GIAC session setting `name` (`:keep_acosh_asinh`, `:complex_mode`, `:decimal_digits`, …), as
an integer: 0 or 1 for one that is on or off.
"""
function setting(name)
    _wrapper_overrides() || throw(ArgumentError("this wrapper library cannot read GIAC's settings by name"))
    v = with_giac_lock(() -> GiacCxxBindings.get_setting(String(name)))
    v == -1 && throw(ArgumentError("$name is not a GIAC session setting"))
    return Int(v)
end

# f(), with GIAC's epsilon (the tolerance of its numeric algorithms) as it was before
function _keeping_epsilon(f)
    old = string(giac_eval("epsilon"))
    try
        return f()
    finally
        string(giac_eval("epsilon")) == old || giac_eval("epsilon:=$old")
    end
end
