if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

# DNS discretization
N = 101
ξ = LinRange(0, 1, N + 1)[2:end]
Δξ = 1 / N
Aᴺ₂ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ) # 2nd order 
Aᴺ₄ = circulant(N, -2:2, [-1, 8, 0, -8, 1] / 12Δξ) # 4th order 
Aᴺ₆ = circulant(N, -3:3, [1, -9, 45, 0, -45, 9, -1] / 60Δξ) # 6th order 
Aᴺ₈ = circulant(N, -4:4, [-3, 32, -168, 672, 0, -672, 168, -32, 3] / 840Δξ) # 8th order 
Aᴺ = Aᴺ₆

K = 50
M = 2K + 1
x = LinRange(0, 1, M + 1)[2:end]
Δx = 1 / M

if iseven(M)
    K_cos = M ÷ 2
    K_sin = M ÷ 2 - 1
else
    K_cos = M ÷ 2
    K_sin = M ÷ 2
end

e_cos = [cos(2π * k * x) for x ∈ ξ, k ∈ 0:K_cos]
e_sin = [sin(2π * k * x) for x ∈ ξ, k ∈ 1:K_sin]
e = [e_cos e_sin]

ē_cos = [F.Ĝ(k, x) * cos(2π * k * x) for x ∈ x, k ∈ 0:K_cos]
ē_sin = [F.Ĝ(k, x) * sin(2π * k * x) for x ∈ x, k ∈ 1:K_sin]
ē = [ē_cos ē_sin] .+ 1e-3 .* randn.()

dē_cos = [2π * k * F.Ĝ(k, x) * sin(2π * k * x) for x ∈ x, k ∈ 0:K_cos]
dē_sin = [-2π * k * F.Ĝ(k, x) * cos(2π * k * x) for x ∈ x, k ∈ 1:K_sin]
dē = [dē_cos dē_sin] .+ 1e-3 .* randn.()

λ = 1e-10
W = (ē * e') / (e * e' + λ * I)
# W = filter_matrix(F, x, ξ)
plotmat(W; aspect_ratio = N / M)

# ē = W * e

plot(x, ē_cos[:, 1:5])

λ = 1e-1
# λ = 0
R = (e * ē') / (ē * ē' + λ * I)
plotmat(R; aspect_ratio = M / N)

plotmat(W * R)
plotmat(R * W)

Ā_int = W * Aᴺ * R
plotmat(Ā_int)

λ = 1e0
Ā_df = (dē * ē' + λ * Aᴹ) / (ē * ē' + λ * I)
plotmat(Ā_df)
plotmat(Ā_df - Aᴹ)
