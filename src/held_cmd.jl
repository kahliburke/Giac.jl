# The written forms of held commands (made with `hold_cmd`, see held_forms.jl): ∫, d/dx, ℒ, 𝒵,
# lim, Σ, Π, and a generic `name(args)` for anything else. Each renderer takes the command's
# arguments as given and writes them with `giac_latex`.

# ============================================================================
# Argument LaTeX Rendering Helper
# ============================================================================

"""
    _arg_to_latex(arg) -> String

Convert a single argument to its LaTeX representation for a held command's written form.

For GiacExpr, the expression as [`giac_latex`](@ref) writes it. For other types, produces
a simple string representation suitable for LaTeX.
"""
_arg_to_latex(arg::GiacExpr)::String = giac_latex(arg)

# an integrand or operand: a sum in parentheses
_operand_latex(arg::GiacExpr) = (iscall(arg) && symb_funcname(arg) == "+") ? "\\left(" * giac_latex(arg) * "\\right)" : giac_latex(arg)
_operand_latex(arg) = _arg_to_latex(arg)

function _arg_to_latex(arg::Symbol)::String
    return string(arg)
end

function _arg_to_latex(arg::Number)::String
    return string(arg)
end

function _arg_to_latex(arg::String)::String
    return "\\text{" * arg * "}"
end

function _arg_to_latex(arg::AbstractVector)::String
    elements = [_arg_to_latex(x) for x in arg]
    return "[" * join(elements, ", ") * "]"
end

function _arg_to_latex(arg::GiacMatrix)::String
    # Render matrix as LaTeX using GIAC's latex() on the underlying expression
    latex_result = invoke_cmd(:latex, arg)
    latex_str = string(latex_result)
    if length(latex_str) > 2 && latex_str[1] == '"' && latex_str[end] == '"'
        return latex_str[2:end-1]
    end
    return string(arg.ptr)
end

function _arg_to_latex(arg)::String
    return string(arg)
end

# ============================================================================
# Specialized LaTeX Renderers
# ============================================================================

# integrate(expr, var) → \int expr \, dvar
# integrate(expr, var, a, b) → \int_a^b expr \, dvar
function _latex_integrate(io::IO, args::Tuple)
    if length(args) >= 4
        # Definite integral: integrate(expr, var, lower, upper)
        expr_latex = _operand_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        lower_latex = _arg_to_latex(args[3])
        upper_latex = _arg_to_latex(args[4])
        print(io, "\\int_{", lower_latex, "}^{", upper_latex, "} ", expr_latex, " \\, \\mathrm{d}", var_latex)
    elseif length(args) >= 2
        expr_latex = _operand_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        print(io, "\\int ", expr_latex, " \\, \\mathrm{d}", var_latex)
    elseif length(args) == 1
        expr_latex = _arg_to_latex(args[1])
        print(io, "\\int ", expr_latex)
    else
        print(io, "\\int")
    end
end

# diff(expr, var) → \frac{d}{dvar} expr
# diff(expr, var, n) → \frac{d^n}{dvar^n} expr
function _latex_diff(io::IO, args::Tuple)
    if length(args) >= 3
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        order = args[3]
        order_str = _arg_to_latex(order)
        print(io, "\\frac{d^{", order_str, "}}{d", var_latex, "^{", order_str, "}} ", expr_latex)
    elseif length(args) >= 2
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        print(io, "\\frac{d}{d", var_latex, "} ", expr_latex)
    elseif length(args) == 1
        expr_latex = _arg_to_latex(args[1])
        print(io, "\\frac{d}{d?} ", expr_latex)
    end
end

# laplace/ztransform(expr, source, target) → \mathcal{S}\left\{expr\right\}(target)
function _latex_transform(io::IO, symbol::String, args::Tuple)
    if length(args) >= 3
        expr_latex = _arg_to_latex(args[1])
        target_latex = _arg_to_latex(args[3])
        print(io, "\\mathcal{", symbol, "}\\left\\{", expr_latex, "\\right\\}(", target_latex, ")")
    elseif length(args) >= 1
        expr_latex = _arg_to_latex(args[1])
        print(io, "\\mathcal{", symbol, "}\\left\\{", expr_latex, "\\right\\}")
    else
        print(io, "\\mathcal{", symbol, "}")
    end
end

# invlaplace/invztransform(expr, source, target) → \mathcal{S}^{-1}\left\{expr\right\}(target)
function _latex_inv_transform(io::IO, symbol::String, args::Tuple)
    if length(args) >= 3
        expr_latex = _arg_to_latex(args[1])
        target_latex = _arg_to_latex(args[3])
        print(io, "\\mathcal{", symbol, "}^{-1}\\left\\{", expr_latex, "\\right\\}(", target_latex, ")")
    elseif length(args) >= 1
        expr_latex = _arg_to_latex(args[1])
        print(io, "\\mathcal{", symbol, "}^{-1}\\left\\{", expr_latex, "\\right\\}")
    else
        print(io, "\\mathcal{", symbol, "}^{-1}")
    end
end

# Helper to render infinity values in LaTeX
function _latex_infinity_check(arg)::String
    s = _arg_to_latex(arg)
    # Check for Julia Inf
    if arg isa Number
        if arg == Inf
            return "+\\infty"
        elseif arg == -Inf
            return "-\\infty"
        end
    end
    # Check for GIAC's inf representation (as string from _arg_to_latex)
    if s == "inf" || s == "+inf" || s == "+\\infty"
        return "+\\infty"
    elseif s == "-inf" || s == "-\\infty"
        return "-\\infty"
    end
    return s
end

# limit(expr, var, point) → \lim_{var \to point} expr
# limit(expr, var, point, dir) → \lim_{var \to point^+} expr (dir=1) or point^- (dir=-1)
function _latex_limit(io::IO, args::Tuple)
    if length(args) >= 3
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        point_latex = _latex_infinity_check(args[3])
        dir_str = ""
        if length(args) >= 4
            dir = args[4]
            dir_val = dir isa GiacExpr ? string(dir) : string(dir)
            if dir_val == "1"
                dir_str = "^+"
            elseif dir_val == "-1"
                dir_str = "^-"
            else
                dir_str = "^{" * _arg_to_latex(dir) * "}"
            end
        end
        print(io, "\\lim_{", var_latex, " \\to ", point_latex, dir_str, "} ", expr_latex)
    elseif length(args) >= 2
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        print(io, "\\lim_{", var_latex, "} ", expr_latex)
    elseif length(args) == 1
        expr_latex = _arg_to_latex(args[1])
        print(io, "\\lim ", expr_latex)
    else
        print(io, "\\lim")
    end
end

# sum(expr, var, lower, upper) → \sum_{var=lower}^{upper} expr
# product(expr, var, lower, upper) → \prod_{var=lower}^{upper} expr
function _latex_sum_product(io::IO, symbol::String, args::Tuple)
    if length(args) >= 4
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        lower_latex = _latex_infinity_check(args[3])
        upper_latex = _latex_infinity_check(args[4])
        print(io, symbol, "_{", var_latex, "=", lower_latex, "}^{", upper_latex, "} ", expr_latex)
    elseif length(args) >= 2
        expr_latex = _arg_to_latex(args[1])
        var_latex = _arg_to_latex(args[2])
        print(io, symbol, "_{", var_latex, "} ", expr_latex)
    elseif length(args) == 1
        expr_latex = _arg_to_latex(args[1])
        print(io, symbol, " ", expr_latex)
    else
        print(io, symbol)
    end
end

# sum_riemann(expr, [n, k]) → \lim_{n \to +\infty} \sum_{k=0}^{n-1} expr
function _latex_sum_riemann(io::IO, args::Tuple)
    # the variables may come as a GIAC list
    if length(args) >= 2 && args[2] isa GiacExpr && is_vector(args[2])
        args = (args[1], collect(GiacExpr, args[2]), args[3:end]...)
    end
    if length(args) >= 2 && args[2] isa AbstractVector && length(args[2]) >= 2
        expr_latex = _arg_to_latex(args[1])
        n_var = _arg_to_latex(args[2][1])
        k_var = _arg_to_latex(args[2][2])
        print(io, "\\lim_{", n_var, " \\to +\\infty} \\sum_{", k_var, "=0}^{", n_var, "-1} ", expr_latex)
    elseif length(args) >= 2 && args[2] isa AbstractVector && length(args[2]) == 1
        expr_latex = _arg_to_latex(args[1])
        n_var = _arg_to_latex(args[2][1])
        print(io, "\\lim_{", n_var, " \\to +\\infty} \\sum ", expr_latex)
    elseif length(args) >= 1
        # Fallback: show with generic Riemann label
        expr_latex = _arg_to_latex(args[1])
        print(io, "S_{\\mathrm{Riemann}}\\left(", expr_latex, "\\right)")
    else
        print(io, "S_{\\mathrm{Riemann}}")
    end
end

# Generic fallback: \mathrm{cmdname}\left(arg1, arg2, ...\right)
function _latex_generic(io::IO, cmd::Symbol, args::Tuple)
    print(io, "\\mathrm{", string(cmd), "}")
    if !isempty(args)
        arg_strs = [_arg_to_latex(a) for a in args]
        print(io, "\\left(", join(arg_strs, ", "), "\\right)")
    else
        print(io, "()")
    end
end

# ============================================================================
# The written form of a held command
# ============================================================================

function _print_held_latex(io::IO, cmd::Symbol, args::Tuple)
    if cmd === :integrate
        _latex_integrate(io, args)
    elseif cmd === :diff
        _latex_diff(io, args)
    elseif cmd === :laplace
        _latex_transform(io, "L", args)
    elseif cmd === :invlaplace || cmd === :ilaplace
        _latex_inv_transform(io, "L", args)
    elseif cmd === :ztransform || cmd === :ztrans
        _latex_transform(io, "Z", args)
    elseif cmd === :invztransform || cmd === :invztrans
        _latex_inv_transform(io, "Z", args)
    elseif cmd === :limit
        _latex_limit(io, args)
    elseif cmd === :sum
        _latex_sum_product(io, "\\sum", args)
    elseif cmd === :product
        _latex_sum_product(io, "\\prod", args)
    elseif cmd === :sum_riemann
        _latex_sum_riemann(io, args)
    else
        _latex_generic(io, cmd, args)
    end
end
