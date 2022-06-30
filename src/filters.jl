"""
    create_tophat(h)

Create top-hat filter with filter radius `h`.

The kernel is given by

```math
    G(x, \\xi) =
    \\begin{cases}
        \\frac{1}{2h(x)} \\quad & |\\xi - x| \\leq h(x), \\\\
        0 \\quad & \\text{otherwise},
    \\end{cases}
```

and the local transfer function by

```math
    \\hat{G}_k(x) = \\frac{\\sin \\left ( 2 \\pi k h(x) \\right )}{2 \\pi k h(x)}.
```
"""
function create_tophat(h)
    name = "top_hat"
    G(x, ξ) = (abs(ξ - x) ≤ h(x)) / 2h(x)
    Ĝ(k, x) = k == 0 ? 1.0 : sin(2π * k * h(x)) / (2π * k * h(x))
    (; G, Ĝ, name)
end


"""
    create_gaussian(h)

Create Gaussian filter with filter radius `h`.

The kernel is given by

```math
    G(x, \\xi) = \\sqrt{\\frac{3}{2 \\pi h^2(x)}} \\mathrm{e}^{-\\frac{3 (\\xi - x)^2}{2 h^2(x)}},
```

and the local transfer function by

```math
    \\hat{G}_k(x) = \\mathrm{e}^{-\\frac{2 \\pi^2}{3} k^2 h^2(x)}.
```
"""
function create_gaussian(h)
    name = "gaussian"
    G(x, ξ) = √(3 / 2π) / h(x) * exp(-3(ξ - x)^2 / 2h(x)^2)
    Ĝ(k, x) = exp(-2π^2 / 3 * k^2 * h(x)^2)
    (; G, Ĝ, name)
end


"""
    filter_matrix(F, x, ξ)

Create filter matrix with filter `F` from fine grid `ξ` to coarse grid `x`.
"""
function filter_matrix(F, x, ξ; cutoff = 1e-14)
    W = [mapreduce(ℓ -> F.G(x, ξ + ℓ), +, (-1, 0, 1)) for x in x, ξ ∈ ξ]
    W = W ./ sum(W; dims = 2)
    W[abs.(W).<cutoff] .= 0
    W = sparse(W)
    W
end

"""
    interpolation_matrix(x, y)

Create matrix for interpolating from grid ``x \\in \\mathbb{R}^N`` to ``y \\in \\mathbb{R}^M``.
"""
function interpolation_matrix(L, x, y)
    @assert issorted(x)
    @assert issorted(y)
    @assert x[end] - x[1] ≤ L
    @assert y[end] - y[1] ≤ L
    N = length(x)
    M = length(y)
    a = [x[end] - L; x[1:end]]'
    b = [x[1:end]; x[1] + L]'
    Ia = a .< y .≤ b
    Ib = circshift(Ia, (0, 1))
    A = @. (b - y) / (b - a)
    B = circshift(@.((y - a) / (b - a)), (0, 1))
    P = @. A * Ia + B * Ib
    IP = P[:, 2:end]
    IP[:, end] += P[:, 1]
    IP
end

# if false
# IP = interpolation_matrix(1, x, ξ)
# a = sin.(2π .* x)
# plot(x, a; marker = :o)
# plot!(ξ, IP * a)
# end

# if false
# IP = interpolation_matrix(1, ξ, x)
# a = sin.(2π .* ξ)
# plot(ξ, a)
# plot!(x, IP * a; marker = :o)
# end
