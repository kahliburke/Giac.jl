# Scoped assumptions on variables.

"""
    assuming(f, conditions...)
    assuming(conditions...) do
        …
    end

Run `f()` with Giac assumptions such as `k > 0` in force, then remove them. Several conditions
on one variable combine (`k > 0, k < 1`).

```julia
@giac_var k
assuming(k > 0) do
    sqrt(k^2)            # k
end
sqrt(k^2)                # abs(k)
```

Assumptions are Giac state: inside the block they apply to everything Giac evaluates. On exit,
the variables they mention are purged, which also drops assumptions made on them before the block.
"""
function assuming(f, conditions::GiacExpr...)
    vars = GiacExpr[]
    for c in conditions
        new = filter(v -> !any(isequal(v), vars), collect(GiacExpr, free_symbols(c)))
        # the first condition on a variable replaces what Giac knew; later ones add to it
        isempty(new) ? Commands.additionally(c) : Commands.assume(c)
        append!(vars, new)
    end
    try
        return f()
    finally
        foreach(Commands.purge, vars)
    end
end
