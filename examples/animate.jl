if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

gr()

savedir = "output"

c = c_train
i = 1
tplot = LinRange(0, 1, 101)
# sol = S!(state.A, test.ū[:, iplot, 1], tplot)
ylims = extrema([u(c[:, i], ξ, 0) for ξ ∈ ξ])
# anim = @animate for (it, t) ∈ enumerate(tplot)
for (it, t) ∈ enumerate(tplot)
    tstr = @sprintf("%.2f", t)
    τ = 0
    # τ = t
    p = plot(;
        ylims,
        xlabel = "x",
        # xlabel = "x - t",
        # title = "Data sample",
        title = "Solution, t = $tstr",
        legend = :bottomleft,
        size = (400, 300),
        dpi = 200,
    )
    span = x -> [x - h(x), x + h(x)]
    vspan!(p, span(1/4); fillalpha = 0.1, color = 1, label = nothing);
    vspan!(p, span(3/4); fillalpha = 0.1, color = 1, label = nothing);
    plot!(p, ξ, ξ -> u(c[:, i], ξ + τ, t); color = 1, label = "Unfiltered")
    plot!(p, ξ, ξ -> ū(tophat.Ĝ, c[:, i], ξ + τ, t); color = 2, label = "Top-hat")
    plot!(p, ξ, ξ -> ū(gaussian.Ĝ, c[:, i], ξ + τ, t); color = 3, label = "Gaussian")
    # scatter!(p, x, sol[it])
    display(p); sleep(0.005)
    p
end

# gif(anim, "output/filtered.gif")
gif(anim, joinpath(savedir, "animations/filtered.gif"))
# gif(anim, "output/closure.gif")


# Spectra
c = c_train;
i = 1
# k = 0:kmax
k = -K:K
ik = K+1:2K+1
# anim = @animate for t ∈ LinRange(0, T, 101)
for t ∈ LinRange(0, T, 101)
    func = abs
    Et = [exp(-2π * im * k * t) for k ∈ -K:K]
    p = plot(;
        xlabel = "k",
        legend = :bottomleft,
        yscale = :log10,
        # xlims = (-20, 20),
        ylims = (1e-8, 1e-1),
        title = "Fourier coefficients, t = $(@sprintf "%0.2f" t)",
        size = (400, 300),
        dpi = 200,
    )
    scatter!(p, k[ik], func.(Et .* c[:, i])[ik]; label = "Unfiltered", linestyle = :dash)
    # sticks!(
    #     p, k, func.(Φ * (Et .* c[:, i]));
    #     label = "Filtered",
    #     marker = :c,
    # )
    scatter!(p, k[ik], func.(Φ_tophat * (Et .* c[:, i]))[ik]; label = "Top-Hat", marker = :c)
    scatter!(p, k[ik], func.(Φ_gaussian * (Et .* c[:, i]))[ik]; label = "Gaussian", marker = :c)
    p
    display(p); sleep(0.005)
end

# gif(anim, "output/fourier_coefficients_absolute.gif")
gif(anim, joinpath(savedir, "animations/coefficients.gif"))

shiftback(x) = mod(x + π, 2π) - π

# anim = @animate for t ∈ LinRange(0, T/20, 101)
for t ∈ LinRange(0, T / 10, 201)
    Et = [exp(-2π * im * k * t) for k ∈ -K:K]
    p = plot(;
        xlabel = "k",
        legend = :topright,
        ylims = (-π, π),
        title = "Fourier coefficients (phase shift), t = $(@sprintf "%0.3f" t)",
    )
    scatter!(
        p,
        k,
        angle.(Et .* c[:, i]);
        # p, k, shiftback.(angle.(Et .* c[:, i]) .+ 2π .* k .* t);
        label = "Unfiltered",
        linestyle = :dash,
    )
    scatter!(
        p,
        k,
        angle.(Φ * (Et .* c[:, i]));
        # p, k, shiftback.(angle.(Φ * (Et .* c[:, i])) .+ 2π .* k .* t);
        label = "Filtered",
        marker = :c,
    )
    p
    display(p); sleep(0.005)
end
# gif(anim, "output/fourier_coefficients_phase.gif")

# F, tit = tophat, "Top-hat"
F, tit = gaussian, "Gaussian"
# λ_range = [1e-8]
λ_range = 10.0 .^ (-12:2:0)
# anim = @animate for M ∈ 1:100
# for M ∈ 1:100
for M ∈ [1, 10, 50, 100]
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    W = filter_matrix(F, x, ξ)
    U = reshape(dns_train.u, N, :)
    Ū = W * U
    U_valid = reshape(dns_valid.u, N, :)
    Ū_valid = W * U_valid
    R_min = nothing
    r_min = Inf
    λ_min = 0.0
    for λ ∈ λ_range
        R = (U * Ū') / (Ū * Ū' + λ * I)
        r = norm(R * Ū_valid - U_valid) / norm(U_valid)
        if r < r_min
            λ_min = λ
            r_min = r
            R_min = R
        end
    end
    @info "Found R for" M λ_min r_min
    R = R_min
    i = 1
    sample = dns_train.u[:, i, 1]
    p = plot(;
        xlabel = "x",
        legend = :bottomleft,
        legend_font_halign = :left,
        title = "$tit filter, M = $M, N = $N",
        xlims = (0, 1),
        ylims = (-0.3, 0.5),
        size = (400, 300),
        dpi = 200,
    )
    plot!(p, ξ, sample; label = "u")
    plot!(p, x, W * sample; label = "Wu")
    plot!(p, ξ, R * (W * sample); label = "RWu")
    display(p); sleep(0.005)
    p
end

# gif(anim, "output/reconstruction_$(F.name).gif")
gif(anim, joinpath(savedir, "animations/reconstruction_$(F.name).gif"))


# F, tit = tophat, "Top-hat"
F, tit = gaussian, "Gaussian"
# λ_range = [1e-8]
λ_range = 10.0 .^ (-12:2:0)
# anim = @animate for M ∈ unique(round.(Int, 10.0 .^ LinRange(log10(2), log10(200), 200)))
# for M ∈ unique(round.(Int, 10.0 .^ LinRange(log10(2), log10(200), 100)))
# anim = @animate for M ∈ 1:100
# for M ∈ 1:100
for M = [1, 50, 100]
    x = LinRange(0, 1, M + 1)[2:end]
    Δx = 1 / M
    W = filter_matrix(F, x, ξ)
    U = reshape(dns_train.u, N, :)
    Ū = W * U
    U_valid = reshape(dns_valid.u, N, :)
    Ū_valid = W * U_valid
    R_min = nothing
    r_min = Inf
    λ_min = 0.0
    for λ ∈ λ_range
        R = (U * Ū') / (Ū * Ū' + λ * I)
        r = norm(R * Ū_valid - U_valid) / norm(U_valid)
        if r < r_min
            λ_min = λ
            r_min = r
            R_min = R
        end
    end
    @info "Found R for" M λ_min r_min
    R = R_min
    i = 1
    k = 0:K
    # Kmax = 3K ÷ 2
    Kmax = K
    # Kmax = 100
    kmax = 0:Kmax
    kmin = 0:min(M ÷ 2, Kmax)
    sample = dns_train.u[:, i, 1]
    p = plot(;
        xlabel = L"k",
        legend = :bottomleft,
        legend_font_halign = :left,
        title = "$tit filter, M = $M, N = $N",
        xlims = (0, Kmax),
        ylims = (9e-8, 2e-1),
        # ylims = (9e-11, 2e-1),
        yticks = 10.0 .^ (-7:-1),
        # yticks = 10.0 .^ (-10:-1),
        yscale = :log10,
        size = (400, 300),
        dpi = 200,
    )
    scatter!(p, kmax, abs.(fft(sample)[kmax.+1]) ./ N; label = "u", marker = :circle)
    scatter!(p, kmin, abs.(fft(W * sample)[kmin.+1]) ./ M; label = "Wu", marker = :rect)
    scatter!(
        p,
        kmax,
        abs.(fft(R * (W * sample))[kmax.+1]) ./ N;
        label = "RWu",
        marker = :diamond,
    )
    display(p); sleep(0.005)
    p
end

# gif(anim, "output/fft_$(F.name).gif")
gif(anim, joinpath(savedir, "animations/fft_$(F.name).gif"))
