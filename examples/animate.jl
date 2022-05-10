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
