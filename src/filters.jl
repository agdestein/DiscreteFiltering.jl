"""
    create_tophat(h)

Create top-hat filter width filter radius `h`.

The kernel is given by

```math
    G(x, \\xi) =
    \\begin{cases}
        \\frac{1}{2h(x)} \\quad & |x - \\xi| \\leq h(x), \\\\
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
    G(x, ξ) = (abs(x - ξ) ≤ h(x)) / 2h(x)
    Ĝ(k, x) = k == 0 ? 1.0 : sin(2π * k * h(x)) / (2π * k * h(x))
    (; G, Ĝ, name)
end


"""
    create_gaussian(h)

Create Gaussian filter, with filter radius `h`.

The kernel is given by

```math
    G(x, \\xi) = \\sqrt{\\frac{3}{2 \\pi h^2(x)}} \\euler^{-\\frac{3 (x - \\xi)^2}{2 h^2(x)}},
```

and the local transfer function by

```math
    \\hat{G}_k(x) = \\euler^{-\\frac{2 \\pi^2}{3} k^2 h^2(x)}.
```
"""
function create_gaussian(h)
    name = "gaussian"
    G(x, ξ) = √(3 / 2π) / h(x) * exp(-3(x - ξ)^2 / 2h(x)^2)
    Ĝ(k, x) = exp(-2π^2 / 3 * k^2 * h(x)^2)
    (; G, Ĝ, name)
end


"""
    filter_matrix(F, x, ξ)

Create filter matrix with filter `F` from fine grid `ξ` to coarse grid `x`.
"""
function filter_matrix(F, x, ξ)
    W = [mapreduce(ℓ -> F.G(x, ξ + ℓ), +, (-1, 0, 1)) for x in x, ξ ∈ ξ]
    W = W ./ sum(W; dims = 2)
    W[abs.(W).<1e-14] .= 0
    W = sparse(W)
    W
end
