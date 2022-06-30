if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

savedir = "figures/"

# F = tophat
# F = gaussian

# Number of simulations
# n_sim = 11
n_sim = 7

# Coarse discretization
# MM = round.(Int, 10 .^ LinRange(1, log10(500), n_sim))
MM = round.(Int, 10 .^ LinRange(1, 5 / 2, n_sim))
# MM = [10, 50, 100, 150, 200, 250, 300]

eAᴹ = zeros(n_sim)
eAᴹ₂ = zeros(n_sim)
eAᴹ₄ = zeros(n_sim)
eAᴹ₆ = zeros(n_sim)
eAᴹ₈ = zeros(n_sim)
eAᴹ₁₀ = zeros(n_sim)
eĀ_classic = zeros(n_sim)
eĀ_int = zeros(n_sim)
eĀ_df = zeros(n_sim)
eĀ_emb = zeros(n_sim)
eĀ_fourier = zeros(n_sim)
MM

λ_range = 10.0 .^ (-14:0)

operators = (;
    int = (;
        λ = fill(0.0, n_sim),
        r = fill(Inf, n_sim),
        Ā = [zeros(M, M) for M ∈ MM],
        R = [zeros(N, M) for M ∈ MM],
    ),
    df = (; λ = fill(0.0, n_sim), r = fill(Inf, n_sim), Ā = [zeros(M, M) for M ∈ MM]),
    emb = (; λ = fill(0.0, n_sim), r = fill(Inf, n_sim), Ā = [zeros(M, M) for M ∈ MM]),
);

for (i, M) ∈ enumerate(MM)
    println("i, M = $i, $M")
end

i, M = 1, 10
i, M = 2, 18
i, M = 3, 32
i, M = 4, 56
i, M = 5, 100
i, M = 6, 178
i, M = 7, 316

for (i, M) ∈ enumerate(MM)
    # for (i, M) ∈ collect(enumerate(MM))[1:4]

    # F = tophat
    # F = gaussian

    # LES Discretization
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    Aᴹ₂ = circulant(M, -1:1, [1, 0, -1] / 2Δx) # 2nd order
    Aᴹ₄ = circulant(M, -2:2, [-1, 8, 0, -8, 1] / 12Δx) # 4th order
    Aᴹ₆ = circulant(M, -3:3, [1, -9, 45, 0, -45, 9, -1] / 60Δx) # 6th order
    Aᴹ₈ = circulant(M, -4:4, [-3, 32, -168, 672, 0, -672, 168, -32, 3] / 840Δx) # 8th order 
    Aᴹ₁₀ = circulant(
        M,
        -5:5,
        [
            5334336,
            -66679200,
            400075200,
            -1600300800,
            5601052800,
            0,
            -5601052800,
            1600300800,
            -400075200,
            66679200,
            -5334336,
        ] / 6721263360Δx,
    ) # 10th order

    # Aᴹ = Aᴹ₂
    # Aᴹ = Aᴹ₄
    Aᴹ = Aᴹ₆
    # Aᴹ = Aᴹ₈
    # Aᴹ = Aᴹ₁₀
    Aᴹ_mat = Matrix(Aᴹ)
    # plotmat(Aᴹ)
    # plotmat(Aᴹ₂)
    # plotmat(Aᴹ₄)
    # plotmat(Aᴹ₆)
    # plotmat(Aᴹ₈)
    # plotmat(Aᴹ₁₀)

    # Diffusion operator
    Dᴹ₂ = circulant(M, -1:1, [1, -2, 1] / Δx^2) # 2th order
    Dᴹ₄ = circulant(M, -2:2, [-1, 16, -30, 16, -1] / 12Δx^2) # 4th order
    Dᴹ₆ = circulant(M, -3:3, [2, -27, 270, -490, 270, -27, 2] / 180Δx^2) # 6th order
    Dᴹ₈ =
        circulant(M, -4:4, [-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9] / 5040Δx^2) # 8th order
    # Dᴹ = Dᴹ₂
    # Dᴹ = Dᴹ₄
    Dᴹ = Dᴹ₆
    # Dᴹ = Dᴹ₈

    # plotmat(Dᴹ₆)
    # norm(Dᴹ * sin.(2π .* x) .+ 4π^2 .* sin.(2π .* x)) / norm(4π^2 .* sin.(2π .* x))

    # Discrete filter
    W = filter_matrix(F, x, ξ)
    # plotmat(W; aspect_ratio = N / M)
    # plotmat(log.(W))

    W_tophat = filter_matrix(tophat, x, ξ)
    W_gaussian = filter_matrix(gaussian, x, ξ)

    # plotmat(W_tophat; aspect_ratio = N / M)
    # plotmat(W_gaussian; aspect_ratio = N / M)

    # Filtered data sets
    train = create_data_filtered(W, Aᴺ, dns_train)
    valid = create_data_filtered(W, Aᴺ, dns_valid)
    test = create_data_filtered(W, Aᴺ, dns_test)

    # Snapshot matrices
    noise = 0e-3
    U = reshape(dns_train.u, N, :) .+ noise .* randn.()
    # dUdt = reshape(train.dudt, M, :)
    Ū = reshape(train.ū, M, :) .+ noise .* randn.()
    dŪdt = reshape(train.dūdt, M, :) .+ noise .* randn.()
    n_sample = size(Ū, 2)

    # # Unfiltered approach
    # eAᴹ[i] = relerr(Aᴹ, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eAᴹ₂[i] = relerr(Aᴹ₂, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eAᴹ₄[i] = relerr(Aᴹ₄, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eAᴹ₆[i] = relerr(Aᴹ₆, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eAᴹ₈[i] = relerr(Aᴹ₈, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    eAᴹ₁₀[i] = relerr(Aᴹ₁₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # Explicit reconstruction approach
    # IP = interpolation_matrix(1, x, ξ)
    Wmat = Matrix(W)
    operators.int.r[i] = Inf
    for λ ∈ λ_range
        λ = n_sample * λ
        # R = (W'W + λ * I) \ Wmat'
        R = Wmat' / (W * W' + λ * I)
        # R = (U * Ū') / (Ū * Ū' + λ * I)
        # R = (U * Ū' + λ * IP) / (Ū * Ū' + λ * I)
        # plotmat(R; aspect_ratio = M / N)
        Ā = W * Aᴺ * R
        # plotmat(Ā)
        # r = relerr(Ā, long.ū, t_long; abstol = 1e-8 / M, reltol = 1e-6 / M)
        r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
        if r < operators.int.r[i]
            operators.int.λ[i] = λ
            operators.int.r[i] = r
            operators.int.Ā[i] = Ā
            operators.int.R[i] = R
        end
    end
    @show operators.int.λ[i] operators.int.r[i]

    # plotmat(W; aspect_ratio = N / M)
    # plotmat(operators.int.R[i]; aspect_ratio = M / N)
    # plotmat(IP; aspect_ratio = M / N)
    # plotmat(W * Aᴺ * IP)
    # plotmat(Aᴹ)
    # plotmat(operators.int.Ā[i])
    # plotmat(operators.int.Ā[i] - Aᴹ)

    eĀ_int[i] = relerr(operators.int.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # # Fit non-intrusively using least squares on snapshots matrices
    # operators.df.r[i] = Inf
    # λ_diff_range = [1e-9]
    # # λ_diff_range = [1e-12]
    # for λ_conv ∈ λ_range, λ_diff ∈ λ_diff_range
    #     λ₁ = n_sample * λ_conv
    #     λ₂ = n_sample * λ_diff / M
    #     # Ā = (dŪdt * Ū') / (Ū * Ū')
    #     # Ā = dŪdt * Ū' / (Ū * Ū' + λ * I)
    #     Ā = (dŪdt * Ū' + λ₁ * Aᴹ + λ₂ * Dᴹ₆) / (Ū * Ū' + (λ₁ + λ₂) * I)
    #     # plotmat(Ā)
    #     # plotmat(Ā - Aᴹ)
    #     r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
    #     if r < operators.df.r[i]
    #         operators.df.λ[i] = λ₁
    #         operators.df.r[i] = r
    #         operators.df.Ā[i] = Ā
    #     end
    # end
    # @show operators.df.λ[i] operators.df.r[i]
    # # plotmat(operators.df.Ā[i])
    # # plotmat(operators.df.Ā[i] - Aᴹ)
    #
    # eĀ_df[i] = relerr(operators.df.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    #
    # # D = circulant(M, [-1, 0, 1], [1.0, -2.0, 1.0] / Δx^2)
    # # Ā_classic = Aᴹ + Diagonal(1 / 3 .* h.(x) .* dh.(x)) * D
    # # plotmat(Ā_classic)
    # #
    # # eĀ_classic[i] = relerr(Ā_classic, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
    #
    # fit = create_loss_fit(
    #     train.ū,
    #     t_train;
    #     n_sample = 1000,
    #     n_time = 50,
    #     reltol = 1e-7,
    #     abstol = 1e-9,
    #     sensealg = BacksolveAdjoint(),
    #     # sensealg = InterpolatingAdjoint(),
    #     # sensealg = QuadratureAdjoint(),
    # )
    # prior = create_loss_prior(Matrix(Aᴹ))
    # stab = create_loss_prior(Matrix(Dᴹ))
    #
    # # Initial state
    # state = create_initial_state(Matrix(Aᴹ))
    # operators.emb.r[i] = Inf
    #
    # λ_diff_range = [1e-9 / M]
    # # λ_diff_range = [0.0]
    # for λ_conv ∈ λ_range[1], λ_diff ∈ λ_diff_range
    #     loss = create_loss_mixed((fit, prior, stab), (1.0, λ_conv, λ_diff))
    # #     display(plotmat(first(DiscreteFiltering.Zygote.gradient(loss, Matrix(Aᴹ)))))
    # # end
    # #
    #     state = fit_embedded(
    #         state,
    #         loss;
    #         α = 0.01,
    #         n_iter = 1000,
    #         testloss = A -> relerr(A, valid.ū, t_valid),
    #         ntestloss = 10,
    #         doplot = true,
    #     )
    #     Ā = state.A_min
    #     r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
    #     if r < operators.emb.r[i]
    #         operators.emb.λ[i] = λ_conv
    #         operators.emb.r[i] = r
    #         operators.emb.Ā[i] = Ā
    #     end
    # end
    # @show operators.emb.λ[i] operators.emb.r[i]
    # # plotmat(operators.emb.Ā[i])
    # # plotmat(Aᴹ)
    # # plotmat(state.A)
    # # plotmat(state.A - Aᴹ)
    # # plotmat(Ā_emb - Aᴹ)
    # # plotmat(Ā_emb)
    # # plot(state.hist_i, state.hist_r)
    # # plotmat(operators.emb.Ā[i])
    # # plotmat(operators.emb.Ā[i] - Aᴹ)
    #
    # eĀ_emb[i] = relerr(operators.emb.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # e = [int(2π * im * k * x) for x ∈ x, k ∈ -K:K]
    # Ā_fourier = real.(e * Â * e') .* Δx
    # plotmat(Ā_fourier)

    # eĀ_fourier[i] = relerr(Ā_fourier, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    @show(
        i,
        M,
        # eAᴹ[i],
        eAᴹ₂[i],
        eAᴹ₄[i],
        eAᴹ₆[i],
        eAᴹ₈[i],
        eAᴹ₁₀[i],
        # eĀ_classic[i],
        eĀ_int[i],
        eĀ_df[i],
        eĀ_emb[i],
        eĀ_fourier[i],
    )

end

filename = "output/K$(K)/convergence/operators_$(F.name).jld2"
# @info("Saving operators to \"$(pwd())/$filename\"")
# jldsave(filename; operators)

@info("Loading operators from \"$filename\"")
operators = load("output/K$(K)/convergence/operators_$(F.name).jld2", "operators")

filename = "output/K$(K)/convergence/errors_$(F.name).jld2";
# @info("Saving errors to \"$(pwd())/$filename\"")
# jldsave(
#     "output/K$(K)/convergence/errors_$(F.name).jld2";
#     eAᴹ₂,
#     eAᴹ₄,
#     eAᴹ₆,
#     eAᴹ₈,
#     eAᴹ₁₀,
#     eĀ_classic,
#     eĀ_int,
#     eĀ_df,
#     eĀ_emb,
#     eĀ_fourier,
# )

@info("Loading errors from \"$(pwd())/$filename\"")
(eAᴹ₂, eAᴹ₄, eAᴹ₆, eAᴹ₈, eAᴹ₁₀, eĀ_classic, eĀ_int, eĀ_df, eĀ_emb, eĀ_fourier) = load(
    filename,
    "eAᴹ₂",
    "eAᴹ₄",
    "eAᴹ₆",
    "eAᴹ₈",
    "eAᴹ₁₀",
    "eĀ_classic",
    "eĀ_int",
    "eĀ_df",
    "eĀ_emb",
    "eĀ_fourier",
)

p = plot(;
    xlabel = L"M",
    xscale = :log10,
    yscale = :log10,
    # xlims = (10, 1000),
    # xticks = (MM, [L"%$M" for M ∈ MM]),
    # xticks = MM,
    # xticks = (MM, string.(MM)),
    # ylims = (1e-3, 2),
    yticks = 10.0 .^ (-3:0),
    minorgrid = true,
    legend = :bottomleft,
    legend_font_halign = :left,
);
# vline!(p, [2K]; label = L"2 K", linestyle = :dash);
# plot!(p, MM, eAᴹ; marker = :diamond, label = L"\mathbf{A}^{(M)}");
# plot!(p, MM, eAᴹ₂; marker = :diamond, label = L"\mathbf{A}_2^{(M)}");
# plot!(p, MM, eAᴹ₄; marker = :diamond, label = L"\mathbf{A}_4^{(M)}");
# plot!(p, MM, eAᴹ₆; marker = :diamond, label = L"\mathbf{A}_6^{(M)}");
plot!(p, MM, eAᴹ₆; marker = :diamond, label = L"\mathbf{A}^{(M)}");
# plot!(p, MM, eAᴹ₈; marker = :diamond, label = L"\mathbf{A}_8^{(M)}");
# plot!(p, MM, eAᴹ₁₀; marker = :diamond, label = L"\mathbf{A}_{10}^{(M)}");
# plot!(p, MM, eĀ_classic; label = L"$\mathbf{A}^{(M)} + \mathbf{M}$");
plot!(p, MM, eĀ_int; marker = :rect, label = L"$\bar{\mathbf{A}}$, intrusive");
plot!(p, MM, eĀ_df; marker = :circle, label = L"$\bar{\mathbf{A}}$, derivative fit");
plot!(p, MM, eĀ_emb; marker = :xcross, label = L"$\bar{\mathbf{A}}$, embedded");
# plot!(p, MM, eĀ_fourier; marker = :cross, label = L"$\bar{\mathbf{A}}$, Fourier");
p

figsave(p, "convergence_$(F.name)"; savedir, size = (400, 300))

gr()
pgfplotsx()
