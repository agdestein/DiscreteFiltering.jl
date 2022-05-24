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
n_train = 1000
n_valid = 20
n_test = 100

Random.seed!(0)
c_train = reduce(hcat, (create_signal(K) for _ = 1:n_train))
c_valid = reduce(hcat, (create_signal(K) for _ = 1:n_valid))
c_test = reduce(hcat, (create_signal(K) for _ = 1:n_test))

# jldsave("output/coefficients.jld2"; c_train, c_valid, c_test)
c_train, c_valid, c_test = load("output/coefficients.jld2", "c_train", "c_valid", "c_test")

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

ēfine_tophat = [tophat.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ_tophat = efine'ēfine_tophat .* Δξfine
ēfine_gaussian = [gaussian.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ_gaussian = efine'ēfine_gaussian .* Δξfine

# DNS discretization
N = 1000
ξ = LinRange(0, 1, N + 1)[2:end]
Δξ = 1 / N
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Time (T: one period)
T = 1.0
t_train = LinRange(0, 1T, 51)
t_valid = LinRange(0, 2T, 11)
t_test = LinRange(0, 2T, 21)

# DNS solution: approximate
dns_train = create_data_dns(Aᴺ, c_train, ξ, t_train)
dns_valid = create_data_dns(Aᴺ, c_valid, ξ, t_valid)
dns_test = create_data_dns(Aᴺ, c_test, ξ, t_test)
float(Base.summarysize(dns_train))

# DNS solution: exact
dns_train = create_data_exact(Aᴺ, c_train, ξ, t_train)
dns_valid = create_data_exact(Aᴺ, c_valid, ξ, t_valid)
dns_test = create_data_exact(Aᴺ, c_test, ξ, t_test)


M = 100

gr()

F = tophat
F = gaussian

# LES Discretization
x = LinRange(0, 1, M + 1)[2:end]
Δx = 1 / M
Aᴹ = circulant(M, [-1, 1], [1.0, -1.0] / 2Δx)
Aᴹ_mat = Matrix(Aᴹ)

# Discrete filter
W = filter_matrix(F, x, ξ)
plotmat(W; aspect_ratio = N / M)
plotmat(log.(W); aspect_ratio = N / M)

W_tophat = filter_matrix(tophat, x, ξ)
W_gaussian = filter_matrix(gaussian, x, ξ)

plotmat(W_tophat; aspect_ratio = N / M)
plotmat(W_gaussian; aspect_ratio = N / M)

# Filtered data sets
train = create_data_filtered(W, Aᴺ, dns_train)
valid = create_data_filtered(W, Aᴺ, dns_valid)
test = create_data_filtered(W, Aᴺ, dns_test)

# Snapshot matrices
U = reshape(dns_train.u, N, :)
# ∂U∂t = reshape(train.∂u∂t, M, :)
Ū = reshape(train.ū, M, :)
∂Ū∂t = reshape(train.∂ū∂t, M, :)

# Explicit reconstruction approach
# Wmat = Matrix(W)
λ_exp = 0.0
r_exp = Inf
Ā_exp = nothing
for λ ∈ 10.0 .^ (-8:-2)
    # R = (W'W + λ * I) \ Wmat'
    # # R =  Wmat' / (W*W' + λ * I)
    R = (U * Ū') / (Ū * Ū' + λ * I)
    # plotmat(R; aspect_ratio = M / N)
    Ā = W * Aᴺ * R
    # plotmat(Ā)
    r = relerr(Ā, valid.ū, t_valid; abstol = 1e-4 / M, reltol = 1e-2 / M)
    if r < r_exp
        λ_exp = λ
        r_exp = r
        Ā_exp = Ā
    end
end
plotmat(Ā_exp)
@show λ_exp r_exp; plotmat(Ā_exp)

# M = 229
0.006069 # 1e-6 1e-3
0.00131  # 1e-6 1e-4
0.001376 # 1e-7 1e-5
0.001406 # 1e-8 1e-6
0.001408 # 1e-9 1e-7
0.001408 # 1e-10 1e-8

# Fit non-intrusively using least squares on snapshots matrices
λ_ls = 0.0
r_ls = Inf
Ā_ls = nothing
for λ ∈ 10.0 .^ (-8:-2)
    # Ā = (∂Ū∂t * Ū') / (Ū * Ū')
    # Ā = ∂Ū∂t * Ū' / (Ū * Ū' + λ * I)
    Ā = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
    # plotmat(Ā)
    # plotmat(Ā - Aᴹ)
    r = relerr(Ā, valid.ū, t_valid) # ; abstol = 1e-10, reltol = 1e-8)
    if r < r_ls
        λ_ls = λ
        r_ls = r
        Ā_ls = Ā
    end
end
@show λ_ls r_ls; plotmat(Ā_ls)

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
state = nothing

# create_testloss(ū, t) = (A -> relerr(A, ū, t; abstol = 1e-10, reltol = 1e-8))
create_testloss(ū, t) = (A -> relerr(A, ū, t))

state = fit_intrusive(
    Aᴹ_mat,
    train.ū,
    t_train;
    α = 0.001,
    β₁ = 0.9,
    β₂ = 0.999,
    ϵ = 1e-8,
    λ = λ_ls,
    nbatch = 10,
    niter = 1000,
    nepoch = 1,
    initial = state,
    # testloss = create_loss(test.ū₀[:, 1:10], test.ū[:, 1:10, :], t_test, Aᴹ_mat; λ = 0),
    testloss = create_testloss(test.ū[:, 1:10, :], t_test),
    ntestloss = 5,
)
Ā_int = state.A_min

plotmat(Ā_int - Aᴹ)
plotmat(Ā_int)

eAᴹ = relerr(Aᴹ, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
# eĀ_classic = relerr(Ā_classic, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
eĀ_exp = relerr(Ā_exp, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
eĀ_ls = relerr(Ā_ls, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
eĀ_int = relerr(Ā_int, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
# eĀ_fourier = relerr(Ā_fourier, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

@show(
    M,
    eAᴹ,
    # eĀ_classic,
    eĀ_exp,
    eĀ_ls,
    eĀ_int,
    # eĀ_fourier,
);

