# Number of simulations
n_sim = 11

# Coarse discretization
MM = round.(Int, 10 .^ LinRange(1, log10(500), n_sim))
# MM = [10, 50, 100, 150, 200, 250, 300]
eAᴹ = zeros(length(MM))
eĀ_classic = zeros(length(MM))
eĀ_exp = zeros(length(MM))
eĀ_ls = zeros(length(MM))
eĀ_int = zeros(length(MM))
eĀ_fourier = zeros(length(MM))
MM

λ_range = 10 .^ (-8:0.5:0)

operators = (;
    exp = (;
        λ = fill(0.0, n_sim),
        r = fill(0.0, n_sim),
        Ā = fill(zeros(0, 0), n_sim),
        R = fill(zeros(0, 0), n_sim),
    ),
    ls = (; λ = fill(0.0, n_sim), r = fill(0.0, n_sim), Ā = fill(zeros(0, 0), n_sim)),
    int = (; λ = fill(0.0, n_sim), r = fill(0.0, n_sim), Ā = fill(zeros(0, 0), n_sim)),
);


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
for (i, M) ∈ collect(enumerate(MM))[1:7]

    # F = tophat
    # F = gaussian

    # LES Discretization
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    Aᴹ = circulant(M, [-1, 1], [1.0, -1.0] / 2Δx)
    # Aᴹ = circulant(M, -2:2, [-1, 8, 0, -8, 1] / 12Δx)
    # Aᴹ = circulant(M, -2:0, [-1, 4, -3] / 2Δx)
    Aᴹ_mat = Matrix(Aᴹ)
    plotmat(Aᴹ)

    # Discrete filter
    W = filter_matrix(F, x, ξ)
    # plotmat(W; aspect_ratio = N / M)
    # plotmat(log.(W))

    # W_tophat = filter_matrix(tophat, x, ξ)
    # W_gaussian = filter_matrix(gaussian, x, ξ)

    # plotmat(W_tophat; aspect_ratio = N / M)
    # plotmat(W_gaussian; aspect_ratio = N / M)

    # Filtered data sets
    train = create_data_filtered(W, Aᴺ, dns_train)
    valid = create_data_filtered(W, Aᴺ, dns_valid)
    test = create_data_filtered(W, Aᴺ, dns_test)

    # Snapshot matrices
    U = reshape(dns_train.u, N, :)
    # ∂U∂t = reshape(train.∂u∂t, M, :)
    Ū = reshape(train.ū, M, :)
    ∂Ū∂t = reshape(train.∂ū∂t, M, :)

    # # Unfiltered approach
    # eAᴹ[i] = relerr(Aᴹ, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # # Explicit reconstruction approach
    # # Wmat = Matrix(W)
    # operators.exp.r[i] = Inf
    # for λ ∈ λ_range
    #     # R = (W'W + λ * I) \ Wmat'
    #     # # R =  Wmat' / (W*W' + λ * I)
    #     R = (U * Ū') / (Ū * Ū' + λ * I)
    #     # plotmat(R; aspect_ratio = M / N)
    #     Ā = W * Aᴺ * R
    #     # plotmat(Ā)
    #     r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
    #     if r < operators.exp.r[i]
    #         operators.exp.λ[i] = λ
    #         operators.exp.r[i] = r
    #         operators.exp.Ā[i] = Ā
    #         operators.exp.R[i] = R
    #     end
    # end
    # @show operators.exp.λ[i] operators.ls.r[i]
    # plotmat(operators.exp.Ā[i])
    #
    # eĀ_exp[i] = relerr(operators.exp.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # # Fit non-intrusively using least squares on snapshots matrices
    # operators.ls.r[i] = Inf
    # for λ ∈ λ_range
    #     # Ā = (∂Ū∂t * Ū') / (Ū * Ū')
    #     # Ā = ∂Ū∂t * Ū' / (Ū * Ū' + λ * I)
    #     Ā = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
    #     # plotmat(Ā)
    #     # plotmat(Ā - Aᴹ)
    #     r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
    #     if r < operators.ls.r[i]
    #         operators.ls.λ[i] = λ
    #         operators.ls.r[i] = r
    #         operators.ls.Ā[i] = Ā
    #     end
    # end
    # @show operators.ls.λ[i] operators.ls.r[i]
    # plotmat(operators.ls.Ā[i])
    #
    # eĀ_ls[i] = relerr(operators.ls.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # D = circulant(M, [-1, 0, 1], [1.0, -2.0, 1.0] / Δx^2)
    # Ā_classic = Aᴹ + Diagonal(1 / 3 .* h.(x) .* dh.(x)) * D
    # plotmat(Ā_classic)
    #
    # eĀ_classic[i] = relerr(Ā_classic, test.ū₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # loss = create_loss(train.ū, t_train, Aᴹ; λ = 1e-10)
    # loss(Aᴹ)
    # loss(WAR)
    # plotmat(first(Zygote.gradient(loss, Matrix(Aᴹ))))
    # plotmat(first(Zygote.gradient(loss, WAR)))
    # result_ode = DiffEqFlux.sciml_train(loss, Aᴹ_mat, ADAM(0.01); cb = (A, l) -> (println(l); false), maxiters = 500)
    # Ā_intrusive = result_ode.u
    # plotmat(Ā_intrusive)

    # create_testloss(ū, t) = (A -> relerr(A, ū, t; abstol = 1e-10, reltol = 1e-8))
    create_testloss(ū, t) = (A -> relerr(A, ū, t; abstol = 1e-5 / size(A, 1), reltol = 1e-3 / size(A, 1)))
    
    # 
    state = nothing

    # operators.int.Ā[i] = Aᴹ_mat
    operators.int.r[i] = Inf
    # for λ ∈ λ_range
    for λ ∈ [1e-5]
        state = fit_intrusive(
            Aᴹ_mat,
            train.ū,
            t_train;
            α = 0.001,
            β₁ = 0.9,
            β₂ = 0.999,
            ϵ = 1e-8,
            λ,
            nbatch = 100,
            niter = 1000,
            initial = state,
            # testloss = create_loss(test.ū₀[:, 1:10], test.ū[:, 1:10, :], t_test, Aᴹ_mat; λ = 0),
            testloss = create_testloss(valid.ū, t_valid),
            ntestloss = 1,
            ntime = 20,
            ntimebatch = 10,
            doplot = true,
            reltol = 1e-6,
            abstol = 1e-8,
        )
        Ā = state.A_min
        r = relerr(Ā, valid.ū, t_valid; abstol = 1e-5 / M, reltol = 1e-3 / M)
        if r < operators.int.r[i]
            operators.int.λ[i] = λ
            operators.int.r[i] = r
            operators.int.Ā[i] = Ā
        end
    end
    @show operators.int.λ[i] operators.int.r[i]
    # plotmat(operators.int.Ā[i])

    # plotmat(Ā_int - Aᴹ)
    # plotmat(Ā_int)

    eĀ_int[i] = relerr(operators.int.Ā[i], test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

    # e = [exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]
    # Ā_fourier = real.(e * Â * e') .* Δx
    # plotmat(Ā_fourier)
    #
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
    )
end

# jldsave("output/convergence/operators_$(F.name).jld2"; operators)
operators = load("output/convergence/operators_$(F.name).jld2", "operators")

# jldsave(
#     "output/convergence/errors_$(F.name).jld2";
#     eAᴹ,
#     eĀ_classic,
#     eĀ_exp,
#     eĀ_ls,
#     eĀ_int,
#     eĀ_fourier,
# )

(
    eAᴹ,
    eĀ_classic,
    eĀ_exp,
    eĀ_ls,
    eĀ_int,
    eĀ_fourier,
) = load(
    "output/convergence/errors_$(F.name).jld2",
      "eAᴹ"       ,
      "eĀ_classic",
      "eĀ_exp"    ,
      "eĀ_ls"     ,
      "eĀ_int"    ,
      "eĀ_fourier",
)

p = plot(;
    xlabel = L"M",
    xscale = :log10,
    yscale = :log10,
    xlims = (10, 1000),
    ylims = (1e-4, 2),
    yticks = 10.0 .^ (-4:0),
    minorgrid = true,
    legend = :bottomleft,
    legend_font_halign = :left,
);
# vline!(p, [2K]; label = L"2 K", linestyle = :dash);
plot!(p, MM, eAᴹ; marker = :diamond, label = L"\mathbf{A}^{(M)}");
# plot!(p, MM, eĀ_classic; label = L"$\mathbf{A}^{(M)} + \mathbf{M}$");
plot!(p, MM, eĀ_exp; marker = :rect, label = L"$\bar{\mathbf{A}}$, explicit");
plot!(p, MM, eĀ_ls; marker = :circle, label = L"$\bar{\mathbf{A}}$, least squares");
plot!(p, MM, eĀ_int; marker = :xcross, label = L"$\bar{\mathbf{A}}$, intrusive");
# plot!(p, MM, eĀ_fourier; label = L"$\bar{\mathbf{A}}$, Fourier");
p

figsave(p, "convergence_$(F.name)"; size = (400, 300))

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

