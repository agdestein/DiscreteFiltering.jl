if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using Pkg
Pkg.activate(".")
cd("examples")

using DiscreteFiltering
using JLD2
using LinearAlgebra
using SparseArrays
using Plots
using LaTeXStrings
using Random

# Create signal
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) / (abs(k) + 5)^1.5, -K:K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-5 / 6 * max(abs(k), 5), -K:K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-0.5 * max(abs(k), 10)), -K:K)
create_magnitude(k) = (1 + 0.2 * randn()) / (5 + abs(k))^1.5
function create_signal(K)
    c₀ = create_magnitude(0)
    c₊ = map(k -> create_magnitude(k) * exp(2π * im * rand()), 1:K)
    c₋ = reverse(conj.(c₊))
    [c₋; c₀; c₊]
end

# Filter width
h₀ = 1 / 30
# h(x) = (1 + 3 * sin(π * x) * exp(-2x^2)) * h₀ / 2
h(x) = (1 + 1 / 3 * sin(2π * x)) * h₀
dh(x) = 2π / 3 * cos(2π * x) * h₀

# Top-hat filter
tophat = create_tophat(h)
F = tophat

# Gaussian filter
gaussian = create_gaussian(h)
F = gaussian

# Continuous data
K = 50
ntrain = 1000
ntest = 100

Random.seed!(0)
c_train = reduce(hcat, (create_signal(K) for _ = 1:ntrain))
c_test = reduce(hcat, (create_signal(K) for _ = 1:ntest))

# jldsave("output/coefficients.jld2"; c_train, c_test)
c_train, c_test = load("output/coefficients.jld2", "c_train", "c_test")

# Fine discretization
Nfine = 10000
ξfine = LinRange(0, 1, Nfine + 1)[2:end]
Δξfine = 1 / Nfine
efine = [exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
ēfine = [F.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ = efine'ēfine .* Δξfine
# Φinv = inv(Φ)
Φinv = (Φ'Φ + 1e-8I) \ Φ'
Â = -2π * im * Φ * Diagonal(-K:K) * Φinv

Φtophat = Φ
Φgauss = Φ

# DNS discretization
N = 1000
ξ = LinRange(0, 1, N + 1)[2:end]
Δξ = 1 / N
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Time (T: one period)
T = 1.0
t_train = LinRange(0, 1T, 51)
t_test = LinRange(0, 2T, 21)

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
eĀ_exp = zeros(length(MM))
eĀ_ls = zeros(length(MM))
eĀ_int = zeros(length(MM))
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

i, M = 0, 100

# for (i, M) ∈ enumerate(MM)
for (i, M) ∈ collect(enumerate(MM))[1:8]

    gr()

    F = tophat
    F = gaussian

    # LES Discretization
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    Aᴹ = circulant(M, [-1, 1], [1.0, -1.0] / 2Δx)
    Aᴹ_mat = Matrix(Aᴹ)

    W = filter_matrix(F, x, ξ)
    plotmat(W; aspect_ratio = N / M)
    # plotmat(log.(W))

    W_tophat = filter_matrix(tophat, x, ξ)
    W_gaussian = filter_matrix(gaussian, x, ξ)

    plotmat(W_tophat; aspect_ratio = N / M)
    plotmat(W_gaussian; aspect_ratio = N / M)

    train = create_data_filtered(W, Aᴺ, dns_train)
    test = create_data_filtered(W, Aᴺ, dns_test)

    # Fit non-intrusively using least squares on snapshots matrices
    Ū = reshape(train.ū, M, :)
    ∂Ū∂t = reshape(train.∂ū∂t, M, :)
    λ = 1e-5
    # Ā_ls = (∂Ū∂t * Ū') / (Ū * Ū')
    # Ā_ls = ∂Ū∂t * Ū' / (Ū * Ū' + λ * I)
    Ā_ls = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
    plotmat(Ā_ls)
    # plotmat(Ā_ls - Aᴹ)

    Ā_ls_tophat = Ā_ls
    Ā_ls_gaussian = Ā_ls

    # λ = 1e-8 * M
    # Wmat = Matrix(W)
    # R = (W'W + λ * I) \ Wmat'
    # # R =  Wmat' / (W*W' + λ * I)
    # plotmat(R)

    U = reshape(dns_train.u, N, :)
    # λ = 1e-8 * size(U, 2)
    λ = 1e-8
    R = (U * Ū') / (Ū * Ū' + λ * I)
    plotmat(R; aspect_ratio = M / N)

    Rtophat = R
    Rgauss = R

    Ā_exp = W * Aᴺ * R
    plotmat(Ā_exp)
    # plotmat(Ā_exp - Aᴹ)

    Ā_exp_tophat = Ā_exp
    Ā_exp_gaussian = Ā_exp

    e = [exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]
    Ā_fourier = real.(e * Â * e') .* Δx
    plotmat(Ā_fourier)

    D = circulant(M, [-1, 0, 1], [1.0, -2.0, 1.0] / Δx^2)
    Ā_classic = Aᴹ + Diagonal(1 / 3 .* h.(x) .* dh.(x)) * D
    plotmat(Ā_classic)

    # loss = create_loss(train.ū, t_train, Aᴹ; λ = 1e-10)
    # loss(Aᴹ)
    # loss(WAR)
    # plotmat(first(Zygote.gradient(loss, Matrix(Aᴹ))))
    # plotmat(first(Zygote.gradient(loss, WAR)))
    # result_ode = DiffEqFlux.sciml_train(loss, Aᴹ_mat, ADAM(0.01); cb = (A, l) -> (println(l); false), maxiters = 500)
    # Ā_intrusive = result_ode.u
    # plotmat(Ā_intrusive)

    Ā_int = Aᴹ_mat

    # create_testloss(ū, t) = (A -> relerr(A, ū, t; abstol = 1e-10, reltol = 1e-8))
    create_testloss(ū, t) = (A -> relerr(A, ū, t))

    Ā_int = fit_intrusive(
        Aᴹ_mat,
        train.ū,
        t_train;
        α = 0.001,
        β₁ = 0.9,
        β₂ = 0.999,
        ϵ = 1e-8,
        λ = 1e-2,
        nbatch = 20,
        niter = 5000,
        nepoch = 1,
        initial = Ā_int,
        # testloss = create_loss(test.ū₀[:, 1:10], test.ū[:, 1:10, :], t_test, Aᴹ_mat; λ = 0),
        testloss = create_testloss(test.ū[:, 1:10, :], t_test),
        ntestloss = 5,
    )

    plotmat(Ā_int - Aᴹ)
    plotmat(Ā_int)

    eAᴹ[i] = relerr(Aᴹ, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    # eĀ_classic[i] = relerr(Ā_classic, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eĀ_exp[i] = relerr(Ā_exp, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eĀ_ls[i] = relerr(Ā_ls, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eĀ_int[i] = relerr(Ā_int, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    # eĀ_fourier[i] = relerr(Ā_fourier, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    @show(
        i,
        M,
        eAᴹ[i],
        # eĀ_classic[i],
        eĀ_exp[i],
        eĀ_ls[i],
        eĀ_int[i],
        # eĀ_fourier[i],
    );
end

# jldsave("output/$filtername/A_intrusive_$M.jld2"; Ā_intrusive)
Ā_intrusive = load("output/$filtername/A_intrusive_$M.jld2", "Ā_intrusive")

# jldsave(
#     "output/$filtername/errors.jld2";
#     eAᴹ,
#     eĀ_classic,
#     eWAR,
#     eĀ_ls,
#     eĀ_intrusive,
#     eĀ_fourier
# )

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
]

figsave(p, "convergence_$filtername"; size = (400, 300))
