if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using Pkg;
Pkg.activate(".");
cd("examples")

using DiscreteFiltering
using JLD2
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using DiffEqFlux
using Plots
using LaTeXStrings

# Create signal
create_signal(K) =
    map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) / (abs(k) + 5)^1.5, (-K):K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-5 / 6 * max(abs(k), 5), -K:K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-0.5 * max(abs(k), 10)), -K:K)

# Filter width
h₀ = 1 / 30
# h(x) = (1 + 3 * sin(π * x) * exp(-2x^2)) * h₀ / 2
h(x) = (1 + 1 / 3 * sin(2π * x)) * h₀
dh(x) = 2π / 3 * cos(2π * x) * h₀

# Top-hat filter
filtername = "top_hat"
G(x, ξ) = (abs(x - ξ) ≤ h(x)) / 2h(x)
Ĝ(k, x) = k == 0 ? 1.0 : sin(2π * k * h(x)) / (2π * k * h(x))

# Gaussian filter
filtername = "gaussian"
G(x, ξ) = √(6 / π) / h(x) * exp(-6(x - ξ)^2 / h(x)^2)
Ĝ(k, x) = exp(-4π^2 / 3 * k^2 * h(x)^2)

# Continuous data
K = 50
ntrain = 1000
ntest = 100

c_train = reduce(hcat, (create_signal(K) for _ = 1:ntrain))
c_test = reduce(hcat, (create_signal(K) for _ = 1:ntest))

jldsave("output/coefficients.jld2"; c_train, c_test)
c_train, c_test = load("output/coefficients.jld2", "c_train", "c_test")

# Fine discretization
Nfine = 10000
ξfine = LinRange(0, 1, Nfine + 1)[2:end]
Δξfine = 1 / Nfine
efine = [exp(2π * im * k * x) for x ∈ ξfine, k ∈ (-K):K]
ēfine = [Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ (-K):K]
Φ = efine'ēfine .* Δξfine
# Φinv = inv(Φ)
Φinv = (Φ'Φ + 1e-8I) \ Φ'
Â = -2π * im * Φ * Diagonal((-K):K) * Φinv

# DNS discretization
N = 1000
ξ = LinRange(0, 1, N + 1)[2:end]
Δξ = 1 / N
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Time (T: one period)
T = 1.0
t_train = LinRange(0, 1T, 51)[2:end]
t_test = LinRange(0, 2T, 21)[2:end]

# DNS solution: approximate
dns_train = create_data_dns(Aᴺ, c_train, ξ, t_train)
dns_test = create_data_dns(Aᴺ, c_test, ξ, t_test)

# DNS solution: exact
dns_train = create_data_exact(Aᴺ, c_train, ξ, t_train)
dns_test = create_data_exact(Aᴺ, c_test, ξ, t_test)

# Coarse discretization
MM = round.(Int, 10 .^ LinRange(1, log10(500), 11))
# MM = [10, 50, 100, 150, 200, 250, 300]
eAᴹ = zeros(length(MM))
eĀ_classic = zeros(length(MM))
eWAR = zeros(length(MM))
eĀ_ls = zeros(length(MM))
eĀ_intrusive = zeros(length(MM))
eĀ_fourier = zeros(length(MM))
MM

for (i, M) ∈ enumerate(MM)
    println("i, M = $i, $M")
end

i, M = 1, 10
i, M = 2, 15
i, M = 3, 22
i, M = 4, 32
i, M = 5, 48
i, M = 6, 71
i, M = 7, 105
i, M = 8, 155
i, M = 9, 229
i, M = 10, 338
i, M = 11, 500

# for (i, M) ∈ enumerate(MM)
for (i, M) ∈ collect(enumerate(MM))[1:8]

    gr()

    # LES Discretization
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    Aᴹ = circulant(M, [-1, 1], [1.0, -1.0] / 2Δx)
    Aᴹ_mat = Matrix(Aᴹ)

    # # Filter
    # h₀ = 3Δx
    # # h(x) = h₀ / 2 * (1 + 3 * sin(π * x) * exp(-2x^2))
    # h(x) = h₀ * (1 + 1 / 3 * sin(2π * x))
    # # G(x, ξ) = abs(x - ξ) ≤ h(x)
    # G(x, ξ) = exp(-6(x - ξ)^2 / h(x)^2)

    W = [mapreduce(ℓ -> G(x, ξ + ℓ), +, (-1, 0, 1)) for x in x, ξ ∈ ξ]
    W = W ./ sum(W; dims = 2)
    W[abs.(W) .< 1e-14] .= 0
    W = sparse(W)
    plotmat(W)
    # plotmat(log.(W))

    train = create_data_filtered(W, Aᴺ, dns_train)
    test = create_data_filtered(W, Aᴺ, dns_test)

    # Fit non-intrusively using least squares on snapshots matrices
    Ū = reshape(train.ū, M, :)
    ∂Ū∂t = reshape(train.∂ū∂t, M, :)
    λ = 1e-3
    # Ā_ls = (∂Ū∂t * Ū') / (Ū * Ū')
    # Ā_ls = ∂Ū∂t * Ū' / (Ū * Ū' + λ * I)
    Ā_ls = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
    plotmat(Ā_ls)

    # λ = 1e-10 * M
    # Wmat = Matrix(W)
    # R = (W'W + λ * I) \ Wmat'
    # # R =  Wmat' / (W*W' + λ * I)
    # plotmat(R)

    U = reshape(dns_train.u, N, :)
    # λ = 1e-8 * size(U, 2)
    λ = 1e-6
    R = (U * Ū') / (Ū * Ū' + λ * I)
    plotmat(R)

    WAR = W * Aᴺ * R
    plotmat(WAR)

    e = [exp(2π * im * k * x) for x ∈ x, k ∈ (-K):K]
    Ā_fourier = real.(e * Â * e') .* Δx
    plotmat(Ā_fourier)

    D = circulant(M, [-1, 0, 1], [1.0, -2.0, 1.0] / Δx^2)
    Ā_classic = Aᴹ + Diagonal(1 / 3 .* h.(x) .* dh.(x)) * D
    plotmat(Ā_classic)

    # loss = create_loss(train.ū₀, train.ū, t_train, Aᴹ; λ = 1e-10)
    # loss(Aᴹ)
    # loss(WAR)
    # plotmat(first(Zygote.gradient(loss, Matrix(Aᴹ))))
    # plotmat(first(Zygote.gradient(loss, WAR)))
    # result_ode = DiffEqFlux.sciml_train(loss, Aᴹ_mat, ADAM(0.01); cb = (A, l) -> (println(l); false), maxiters = 500)
    # Ā_intrusive = result_ode.u
    # plotmat(Ā_intrusive)

    Ā_intrusive = Aᴹ_mat

    # create_testloss(ū₀, ū, t) = (A -> relerr(A, ū₀, ū, t; abstol = 1e-10, reltol = 1e-8))
    create_testloss(ū₀, ū, t) = (A -> relerr(A, ū₀, ū, t))

    Ā_intrusive = fit_intrusive(
        Aᴹ_mat,
        train.ū₀,
        train.ū,
        t_train;
        α = 0.001,
        β₁ = 0.9,
        β₂ = 0.999,
        ϵ = 1e-8,
        λ = 1e-10,
        nbatch = 100,
        niter = 500,
        nepoch = 1,
        initial = Ā_intrusive,
        # testloss = create_loss(test.ū₀[:, 1:10], test.ū[:, 1:10, :], t_test, Aᴹ_mat; λ = 0),
        testloss = create_testloss(test.ū₀[:, 1:10], test.ū[:, 1:10, :], t_test),
        ntestloss = 20,
    )

    plotmat(Ā_intrusive - Aᴹ)

    eAᴹ[i] = relerr(Aᴹ, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    # eĀ_classic[i] = relerr(Ā_classic, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eWAR[i] = relerr(WAR, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eĀ_ls[i] = relerr(Ā_ls, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eĀ_intrusive[i] =
        relerr(Ā_intrusive, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    # eĀ_fourier[i] = relerr(Ā_fourier, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    @show(
        i,
        M,
        eAᴹ[i],
        # eĀ_classic[i],
        eWAR[i],
        eĀ_ls[i],
        eĀ_intrusive[i],
        # eĀ_fourier[i],
    )
end

jldsave("output/$filtername/A_intrusive_$M.jld2"; Ā_intrusive)
Ā_intrusive = load("output/$filtername/A_intrusive_$M.jld2", "Ā_intrusive")

# aa = jldopen("output/$filtername/errors.jld2", "r")

jldsave(
    "output/$filtername/errors.jld2";
    eAᴹ,
    eĀ_classic,
    eWAR,
    eĀ_ls,
    eĀ_intrusive,
    eĀ_fourier,
)

(
    eAᴹ,
    # eĀ_classic,
    eWAR,
    eĀ_ls,
    eĀ_intrusive,
    # eĀ_fourier,
) = load(
    "output/$filtername/errors.jld2",
    "eAᴹ",
    # "eĀ_classic",
    "eWAR",
    "eĀ_ls",
    "eĀ_intrusive",
    # "eĀ_fourier",
)

p = plot(;
    xlabel = L"M",
    xscale = :log10,
    yscale = :log10,
    xlims = (10, 1000),
    ylims = (1e-5, 2),
    yticks = 10.0 .^ (-5:0),
    minorgrid = true,
    legend = :bottomleft,
    legend_font_halign = :left,
);
plot!(p, MM, eAᴹ; marker = :diamond, label = L"\mathbf{A}^{(M)}");
# plot!(p, MM, eĀ_classic; label = L"$\mathbf{A}^{(M)} + \mathbf{M}$");
plot!(p, MM, eWAR; marker = :rect, label = L"\mathbf{W} \mathbf{A}^{(N)} \mathbf{R}");
plot!(p, MM, eĀ_ls; marker = :circle, label = L"$\bar{\mathbf{A}}$, least squares");
plot!(p, MM, eĀ_intrusive; marker = :xcross, label = L"$\bar{\mathbf{A}}$, intrusive");
# plot!(p, MM, eĀ_fourier; label = L"$\bar{\mathbf{A}}$, Fourier");
p

gr()
pgfplotsx()

[
    :none,
    :auto,
    :circle,
    :rect,
    :star5,
    :diamond,
    :hexagon,
    :cross,
    :xcross,
    :utriangle,
    :dtriangle,
    :rtriangle,
    :ltriangle,
    :pentagon,
    :heptagon,
    :octagon,
    :star4,
    :star6,
    :star7,
    :star8,
    :vline,
    :hline,
    :+,
    :x,
].figsave(
    p,
    "convergence_$filtername";
    size = (400, 300),
)
