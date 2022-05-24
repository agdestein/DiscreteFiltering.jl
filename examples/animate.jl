using LaTeXStrings

gr()

anim = @animate for t ∈ LinRange(0, 1, 201)
    tstr = @sprintf("%.3f", t)
    p = plot(; ylims = (-0.7, 0.7), xlabel = "x", title = "Filtered solutions, t = $tstr")
    for i = 1:3
        # plot!(p, ξ, train.u[i].(ξ, t); label = L"u_%$i", color = i, linestyle = :dash)
        plot!(p, ξ, train.ū[i].(ξ, t); label = L"\bar{u}_%$i", color = i)
    end
    p
end
gif(anim, "output/filtered.gif")
# gif(anim, "filtered_unfiltered.gif")

anim = @animate for t ∈ LinRange(0, 1, 201)
    tstr = @sprintf("%.3f", t)
    p = plot(;
        ylims = (-0.7, 0.7),
        xlabel = "x - t",
        title = "Filtered solutions, t = $tstr",
    )
    for i = 1:3
        # plot!(p, ξ, train.u[i].(ξ .+ t, t); label = L"u_%$i", color = i, linestyle = :dash)
        plot!(p, ξ, train.ū[i].(ξ .+ t, t); label = L"\bar{u}_%$i", color = i)
    end
    p
end
gif(anim, "output/closure.gif")
# gif(anim, "closure_unfiltered.gif")

anim = @animate for λ ∈ 10 .^ LinRange(-8, 2, 101)
    λ = λ * size(Ū, 2)
    Ā_ls = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
    plotmat(Ā_ls)
end
gif(anim, "output/A_ls.gif")


# Spectra
c = c_test;
i = 1
# k = 0:kmax
k = (-K):K
# anim = @animate for t ∈ LinRange(0, T, 101)
for t ∈ LinRange(0, T, 501)
    func = abs
    Et = [exp(-2π * im * k * t) for k ∈ (-K):K]
    p = plot(;
        xlabel = "k",
        legend = :topright,
        yscale = :log10,
        # xlims = (-20, 20),
        ylims = (1e-5, 1e-1),
        title = "Fourier coefficients (absolute value), t = $(@sprintf "%0.2f" t)",
    )
    scatter!(p, k, func.(Et .* c[:, i]); label = "Unfiltered", linestyle = :dash)
    # sticks!(
    #     p, k, func.(Φ * (Et .* c[:, i]));
    #     label = "Filtered",
    #     marker = :c,
    # )
    scatter!(p, k, func.(Φtophat * (Et .* c[:, i])); label = "Top-Hat", marker = :c)
    scatter!(p, k, func.(Φgauss * (Et .* c[:, i])); label = "Gaussian", marker = :c)
    p
    display(p)
    sleep(0.005)
end
# gif(anim, "output/fourier_coefficients_absolute.gif")

shiftback(x) = mod(x + π, 2π) - π

# anim = @animate for t ∈ LinRange(0, T/20, 101)
for t ∈ LinRange(0, T / 10, 201)
    Et = [exp(-2π * im * k * t) for k ∈ (-K):K]
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
    display(p)
    sleep(0.02)
end
# gif(anim, "output/fourier_coefficients_phase.gif")

F = tophat
# F = gaussian
# λ_range = [1e-8]
λ_range = 10.0 .^ (-12:2:0)
anim = @animate for M ∈ unique(round.(Int, 10.0 .^ LinRange(log10(2), log10(200), 200)))
# for M ∈ round.(Int, 10.0 .^ LinRange(log10(2), log10(200), 100))
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
        xlabel = L"x",
        legend = :bottomleft,
        legend_font_halign = :left,
        title = "$(titlecase(F.name)) filter, M = $M, N = $N",
        xlims = (0, 1),
        ylims = (-0.3, 0.5),
    );
    plot!(p, ξ, sample; label = "u");
    plot!(p, x, W * sample; label = "Wu");
    plot!(p, ξ, R * (W * sample); label = "RWu");
    # display(p); sleep(0.01)
    p
end

gif(anim, "output/reconstruction_$(F.name).gif")
