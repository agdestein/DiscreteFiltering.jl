using Plots, LaTeXStrings

gr()
pgfplotsx()

figsave(
    # pplotmat(Aᴹ), "convection/AM"; title = L"\mathbf{A}^{(M)}",
    # pplotmat(WAR), "convection/WAR"; title = L"\mathbf{W} \mathbf{A}^{(N)} \mathbf{R}",
    pplotmat(Ā_ls - Aᴹ),
    "convection/Abar_least_squares";
    title = L"\bar{\mathbf{A}}^{\mathrm{LS}} - \mathbf{A}^{(M)}",
    # pplotmat(Ā - Aᴹ), "convection/Abar_intrusive"; title = L"\bar{\mathbf{A}}^\mathrm{intrusive} - \mathbf{A}^{(M)}",
    suffices = ("png",),
    dpi = 200,
    size = (450, 400),
    thickness_scaling = 1.5,
)


# Filter
p = plot(
    ξ,
    h;
    xlabel = L"x",
    ylims = (0.0, 0.05),
    legend = nothing,
    # title = L"h(x)",
)

figsave(p, "filter_width"; size = (400, 300))

# Transfer function
p = plot(; xlabel = L"x", legend = :bottomright);
for k ∈ 0:5:25
    plot!(p, ξ, x -> Ĝ(k, x); label = L"k = %$k")
end
p

p = plot(; xlabel = L"k", legend = :bottomleft);
k = LinRange(0, 50, 1000)
for x ∈ LinRange(0, 0.5, 6)
    plot!(p, k, k -> Ĝ(k, x); label = L"x = %$x")
end
p

figsave(p, "local_transfer_functions"; size = (400, 300))

p = let
    x = LinRange(0.0, 1.0, 50)
    # k = LinRange(0.0, K, 100)
    k = 0:K
    surface(
        k,
        x,
        Ĝ.(k', x);
        zlims = (-0.25, 1),
        zticks = -0.25:0.25:1,
        xlabel = L"k",
        ylabel = L"x",
        legend = false,
    )
end

figsave(p, "transfer_function_top_hat"; size = (400, 300))#, suffices = ("pdf",))
figsave(p, "transfer_function_gauss"; size = (400, 300))#, suffices = ("pdf",))

# Spectrum
i = 1
coeffs = coeffs_train;
dataset = train;
k = 0:kmax
e = abs2.(fft(dataset.u[i].(ξ, 0.0)))[1:(kmax + 1)];
e /= e[1];
ē = abs2.(fft(dataset.ū₀_data[:, i]))[1:(kmax + 1)];
ē /= ē[1];
p = plot(; xlabel = L"k", yscale = :log10, legend = :topright);
plot!(p, k, e; label = L"u");
plot!(p, k, ē; label = L"\bar{u}");
# plot!(p, k[5:end], exp.(-5 / 3 .* (k[5:end] .- 5)); linestyle = :dash, label = L"-\frac{5}{3} k");
p

c = c_train
p = plot(;
    xlabel = L"k",
    yscale = :log10,
    legend = false,
    # legend = :topright,
    # legend_font_halign = :left,
    # title = "e(k)",
);
for i = 1:3
    scatter!(p, (-K):K, abs.(c[:, i]); label = "Unfiltered $i", color = i, marker = :d)
    scatter!(p, (-K):K, abs.(Φ * c[:, i]); label = "Filtered $i", color = i)
end
p

figsave(p, "convection/spectra"; size = (400, 300))#, suffices = ("pdf",))

# Filtered and unfiltered solutions
p = plot(;
    xlabel = L"x",
    legend = false,
    # title = "Filtered and unfiltered signals",
);
for i = 1:3
    # plot!(p, ξ, train.u[i].(ξ, 0.0); linestyle = :dash, color = i)
    # plot!(p, ξ, train.ū[i].(ξ, 0.0); color = i)
    plot!(p, ξ, ξ -> u(c_train[:, i], ξ, 0.0); linestyle = :dash, color = i)
    plot!(p, ξ, ξ -> ū(c_train[:, i], ξ, 0.0); color = i)
    # scatter!(p, x, train.ū[i].(x, 0.0); color = i)
end
p

figsave(p, "initial_conditions"; size = (400, 300))#, suffices = ("pdf",))

# Superposition
c, dataset = c_test, test;
i = 1
p = plot(;
    xlims = (0, 1),
    xlabel = L"x - t",
    # xlabel = L"x - t \pmod 1",
    legend = :topright,
    legend_font_halign = :left,
    # title = "ū(x+t, t)",
);
# plot!(p, ξ, dataset.u[i].(ξ, 0.0); label = L"u(x, 0)", linestyle = :dash);
plot!(p, ξ, x -> u(c[:, i], x, 0.0); label = L"u(x, 0)", linestyle = :dash);
for (j, t) ∈ enumerate(LinRange(0, T / 2, 3))
    plot!(
        # p, ξ, dataset.ū[i].(ξ .+ t, t);
        p,
        ξ,
        x -> ū(c[:, i], x .+ t, t);
        color = j,
        # label = L"t = %$t"
        label = L"\bar{u}(x, %$t)",
    )
end
p

figsave(p, "superposed"; size = (400, 300))

# Spectra
c, dataset = c_test, test;
i = 1
k = (-K):K
p = plot(;
    # xlims = (0, 1),
    xlabel = L"k",
    legend = :topright,
    legend_font_halign = :left,
    # title = "ū(x+t, t)",
    yscale = :log10,
    yticks = 10.0 .^ (-5:0),
);
scatter!(
    p,
    k,
    abs.(c[:, i]);
    # label = L"\hat{u}(k, 0)",
    label = L"u(x, 0)",
    marker = :d,
);
for (j, t) ∈ enumerate(LinRange(0, T / 2, 3))
    Et = [exp(-2π * im * k * t) for k ∈ (-K):K]
    scatter!(
        p,
        k,
        abs.(Φ * (Et .* c[:, i]));
        color = j,
        # marker = :c,
        # label = L"\hat{\bar{u}}(k, %$t)"
        label = L"\bar{u}(x, %$t)",
    )
end
p

figsave(p, "superposed_spectra"; size = (400, 300))

# Plot prediction at T/2
t = T / 2
c, dataset = c_test, test;
iplot = 1:3
y⁻ = minimum(dataset.ū₀[:, iplot])
y⁺ = maximum(dataset.ū₀[:, iplot])
Δy = y⁺ - y⁻
p = plot(xlabel = "\$x\$", legend = false, xlims = (0, 1));
for (ii, i) ∈ enumerate(iplot)
    plot!(p, ξ, x -> ū(c[:, i], x, t); color = ii)
    # Ause = Aᴹ
    # Ause = WAR
    Ause = Ā_ls
    # Ause = Ā_fourier
    # Ause = Ā
    scatter!(
        p,
        x,
        S(Ause, dataset.ū₀[:, i], [t])[end];
        markeralpha = 0.5,
        markersize = 3,
        label = "i = $i, fit",
        color = ii,
    )
end
p

figsave(p, "initial"; size = (400, 300))
figsave(p, "final"; size = (400, 300))

# Error evolution
c_long = reduce(hcat, (create_signal(K) for _ = 1:20))
t_long = 10 .^ LinRange(-2, 2, 51);
dns_long = create_data_dns(Aᴺ, c_long, ξ, t_long)
# dns_long = create_data_exact(Aᴺ, c_test, ξ, t_long)
long = create_data_filtered(W, Aᴺ, dns_long);
e_long_Aᴹ = relerrs(Aᴹ, long.ū₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_WAR = relerrs(WAR, long.ū₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_ls = relerrs(Ā_ls, long.ū₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_intrusive =
    relerrs(Ā_intrusive, long.ū₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Ā_fourier = relerrs(Ā_fourier, long.ū₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);

t₀, t₁ = extrema(t_train)
p = plot(;
    xlabel = L"t",
    # ylabel = "Mean relative error",
    xscale = :log10,
    yscale = :log10,
    legend = :bottomright,
    # xticks = 10.0 .^ (-2:2),
    yticks = 10.0 .^ (-6:0),
    xlims = extrema(t_long),
    ylims = (5e-5, 2.0e0),
    legend_font_halign = :left,
    minorgrid = true,
);
vspan!(p, [t₀, t₁]; fillalpha = 0.1, label = "Training interval");
# vline!(p, t_train; label = "Training snapshots", linestyle = :dash);
plot!(p, t_long, e_long_Aᴹ; label = L"\mathbf{A}^{(M)}");
plot!(p, t_long, e_long_WAR; label = L"\mathbf{W} \mathbf{A}^{(N)} \mathbf{R}");
plot!(p, t_long, e_long_Ā_ls; label = L"$\bar{\mathbf{A}}$, least squares");
plot!(p, t_long, e_long_Ā_intrusive; label = L"$\bar{\mathbf{A}}$, intrusive");
# plot!(p, t_long, e_long_Ā_fourier; label = L"$\bar{\mathbf{A}}$, Fourier");
p

figsave(p, "comparison_$filtername"; size = (400, 300))


# Energy evolution
tnew = 6T
i = 7
tplot = LinRange(0, tnew, 1000)
saveat = LinRange(0, tnew, 2000)
# sol = S_mem(Aᴹ, train.ūₕ₀[:, i], saveat)
sol = S(WAR, train.ū₀_data[:, i], saveat)
# sol = S_mem(Ā, train.ū₀_data[:, i], saveat)
# sol = S_mem(Ā_ls, train.ū₀_data[:, i], saveat)
E(t) = 1 / 2M * sum(abs2, train.u[i].(x, t))
Ē(t) = 1 / 2M * mapreduce(apply_filter(x -> train.u[i](x, t)^2, ℱ, domain), +, x)
Eu = E.(tplot)
Eū = [1 / 2M * sum(abs2, sol(t)) for t ∈ tplot]
Eup = [1 / 2M * sum(abs2, train.u[i].(x, t) .- sol(t)) for t ∈ tplot]

p = plot(; xlabel = L"t", legend_font_halign = :left, legend = :topright);
plot!(p, tplot, E; label = L"E(u)");
plot!(p, tplot, Eū; label = L"E(\bar{u})");
plot!(p, tplot, Eū + Eup; label = L"E(\bar{u}) + E(u')");
p

figsave(p, "energy"; size = (400, 300))

scatter(x, sin.(25 .* x))

##
# d = train;
d = test;
p1 = plot(; xlims = (a, b), xlabel = "x", title = "Solution (test data)");
j = 0
k = 9
i = 6
for k in [1, 3, 6, 9]
    # for i = 1:3
    j += 1
    lab = L"\bar{u}(%$(tstops[k]))"
    scatter!(
        x,
        d.ūₕ[k][:, i];
        # label = "$lab exact",
        label = nothing,
        color = j,
        markeralpha = 0.5,
    )
    plot!(
        x,
        S(Aᴹ, d.ū₀_data[:, i], [tstops[k]])[end];
        linestyle = :dot,
        label = nothing,
        color = j,
    )
    plot!(x, S(Ā, d.ū₀_data[:, i], [tstops[k]])[end]; label = lab, color = j)
end
plot(p1)

p2 = plot(; xlims = (a, b), xlabel = L"x", title = "Filter width");
xline(x, y) = plot!(x, [y, y]; label = nothing, color = 1);
for (i, x) ∈ enumerate(x)
    hᵢ = h(x)
    x⁻ = x - hᵢ
    x⁺ = x + hᵢ
    if x⁻ < a
        xline([a, x⁺], i)
        xline([b - (a - x⁻), b], i)
    elseif x⁺ > b
        xline([x⁻, b], i)
        xline([a, a + (x⁺ - b)], i)
    else
        xline([x⁻, x⁺], i)
    end
    scatter!([x], [i]; label = nothing, color = 1)
end
plot(p2)

plot(p1, p2; layout = (2, 1))
