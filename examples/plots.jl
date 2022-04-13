using Plots, LaTeXStrings

Plots.gr()
Plots.pgfplotsx()

function figsave(
    p, name;
    savedir = "figures/",
    suffices = ("pdf", "tikz"),
    kwargs...
)
    for suffix ∈ suffices
        Plots.savefig(
            Plots.plot(p; kwargs...),
            joinpath(savedir, "$name.$suffix"),
        )
    end
end

Plots.gr()

figsave(
    # pplotmat(Aᴹ), "convection/AM"; title = L"\mathbf{A}^{(M)}",
    # pplotmat(WAR), "convection/WAR"; title = L"\mathbf{W} \mathbf{A}^{(N)} \mathbf{R}",
    pplotmat(Ā_ls - Aᴹ), "convection/Abar_least_squares"; title = L"\bar{\mathbf{A}}^{\mathrm{LS}} - \mathbf{A}^{(M)}",
    # pplotmat(Ā - Aᴹ), "convection/Abar_intrusive"; title = L"\bar{\mathbf{A}}^\mathrm{intrusive} - \mathbf{A}^{(M)}",
    suffices = ("png",),
    dpi = 200,
    size = (450, 400),
    thickness_scaling = 1.5
)


# Filter
p = Plots.plot(ξ, h; xlabel = "x", ylims = (0.0, 0.06), legend = :topright)


# Spectrum
dataset = train;
k = 0:kmax
e = abs2.(fft(dataset.u₀[1].(ξ)))[1:kmax+1];
e /= e[1];
ē = abs2.(fft(dataset.ūₕ₀[:, 1]))[1:kmax+1];
ē /= ē[1];
p = Plots.plot(; xlabel = "k", yscale = :log10, legend = :topright);
Plots.plot!(p, k, e; label = L"u");
Plots.plot!(p, k, ē; label = L"\bar{u}");
Plots.plot!(p, k[5:end], exp.(-5 / 3 .* (k[5:end] .- 5)); linestyle = :dash, label = L"-\frac{5}{3} k");
p

p = Plots.plot(; xlabel = "k", yscale = :log10, legend = :topright,
    # title = "e(k)",
);
for i = 1:5
    Plots.plot!(p, 0:kmax, dataset.c[i]; label = "e(k)", linestyle = :dash)
end
p

# Filtered and unfiltered solutions
fig = Figure();
ax = Axis(fig[1, 1]; xlabel = "x", legend = false);
for i = 1:3
    lines!(ax, ξ, train.u₀[i]; linestyle = :dash)
    lines!(ax, ξ, train.ū₀[i]; color = Cycled(i))
end
fig


# Superposition
dataset = test;
i = 8
p = Plots.plot(;
    xlims = (a, b),
    xlabel = L"x",
    legend = :topright
    # title = "ū(x+t, t)",
);
Plots.plot!(p, ξ, dataset.u₀[i].(ξ); label = L"u(x, 0)", linestyle = :dash);
for (j, t) ∈ enumerate(LinRange(0, T / 2, 3))
    Plots.plot!(p, ξ, dataset.ū(t)[i].(ξ .+ t);
        color = j,
        label = L"t = %$t"
        # label = L"\bar{u}(x, %$t)",
    )
end
p

figsave(p, "convection/superposed"; size = (400, 300))


# Spectra
dataset = test;
i = 8
k = 0:kmax
p = Plots.plot(;
    # xlims = (a, b),
    xlabel = L"k",
    legend = :topright,
    # title = "ū(x+t, t)",
    yscale = :log10
);
Plots.plot!(p, k, abs.(fft(dataset.u(0.0)[i].(ξ))[1:kmax+1]); label =
    L"\hat{u}(k, 0)", linestyle = :dash);
for (j, t) ∈ enumerate(LinRange(0, T / 2, 3))
    Plots.plot!(
        p, k, abs.(fft(dataset.ū(t)[i].(ξ))[1:kmax+1]);
        color = j,
        label = L"\hat{\bar{u}}(k, %$t)"
        # label = L"\bar{u}(x, %$t)",
    )
end
p

figsave(p, "convection/superposed_spectra"; size = (400, 300))

# Plot prediction at T/2
iplot = 1:5
y⁻ = minimum(train.ūₕ₀[:, iplot])
y⁺ = maximum(train.ūₕ₀[:, iplot])
Δy = y⁺ - y⁻
p = Plots.plot(xlabel = "\$x\$", legend = false, xlims = (a, b));
for (ii, i) ∈ enumerate(iplot)
    Plots.plot!(p, ξ, train.ū(T)[i].(ξ); color = ii)
    # Ause = Aᴹ
    Ause = WAR
    # Ause = Ā_ls
    # Ause = Ā
    Plots.scatter!(p, x, S(Ause, train.ūₕ₀[:, i], T);
        markeralpha = 0.5, markersize = 3, label = "i = $i, fit", color = ii)
end
p

figsave(p, "convection/initial"; size = (400, 300))
figsave(p, "convection/final"; size = (400, 300))

# Error evolution
t₀, t₁ = extrema(tstops)
data_exact(set) = t -> mapreduce(u -> u.(x), hcat, set.ū(t))
tplot = 10 .^ LinRange(-2, 2, 500)
# tplot = LinRange(0, 20T, 200)[2:end]
# train_exact = data_exact(train).(tplot)
test_exact = data_exact(test).(tplot)
eAᴹ = relerrs(Aᴹ, test.ūₕ₀, test_exact, tplot; abstol = 1e-10, reltol = 1e-8);
# eAᴹ_left = relerrs(Aᴹ_left, test.ūₕ₀, test_exact, tplot; abstol = 1e-10, reltol = 1e-8);
eWAR = relerrs(WAR, test.ūₕ₀, test_exact, tplot; abstol = 1e-10, reltol = 1e-8);
eĀ = relerrs(Ā, test.ūₕ₀, test_exact, tplot; abstol = 1e-10, reltol = 1e-8);
eĀ_ls = relerrs(Ā_ls, test.ūₕ₀, test_exact, tplot; abstol = 1e-10, reltol = 1e-8);

p = Plots.plot(;
    xlabel = L"t / T",
    # ylabel = "Mean relative error",
    xscale = :log10,
    yscale = :log10,
    legend = :bottomright,
    xticks = 10.0 .^ (-2:2),
    yticks = 10.0 .^ (-5:0),
    xlims = extrema(tplot),
    ylims = (5e-6, 2.0),
    legend_font_halign = :left,
    minorgrid = true
);
Plots.vspan!(p, [t₀, t₁]; fillalpha = 0.1, label = "Training interval");
# Plots.vline!(p, tstops; label = "Training snapshots", linestyle = :dash);
Plots.plot!(p, tplot, eAᴹ; label = L"\mathbf{A}^{(M)}");
# Plots.plot!(p, tplot, eAᴹ_left; label = L"\mathbf{A}_l^{(M)}");
Plots.plot!(p, tplot, eWAR; label = L"\mathbf{W} \mathbf{A}^{(N)} \mathbf{R}");
Plots.plot!(p, tplot, eĀ; label = L"$\bar{\mathbf{A}}$, intrusive");
Plots.plot!(p, tplot, eĀ_ls; label = L"$\bar{\mathbf{A}}$, least squares");
# Plots.plot!(p, tplot, 1e-3 .* tplot .^ 1)
p

figsave(p, "convection/comparison"; size = (400, 300))


# Energy evolution
tnew = 6T
i = 7
tplot = LinRange(0, tnew, 1000)
saveat = LinRange(0, tnew, 2000)
# sol = S_mem(Aᴹ, train.ūₕ₀[:, i], saveat)
sol = S_mem(WAR, train.ūₕ₀[:, i], saveat)
# sol = S_mem(Ā, train.ūₕ₀[:, i], saveat)
# sol = S_mem(Ā_ls, train.ūₕ₀[:, i], saveat)
E(t) = 1 / 2M * sum(abs2, train.u(t)[i].(x))
Ē(t) = 1 / 2M * sum(apply_filter(x -> train.u(t)[i](x)^2, ℱ, domain).(x))
Eu = E.(tplot)
Eū = [1 / 2M * sum(abs2, sol(t)) for t ∈ tplot]
Eup = [1 / 2M * sum(abs2, train.u(t)[i].(x) .- sol(t)) for t ∈ tplot]

p = Plots.plot(; xlabel = L"t", legend = :topright);
Plots.plot!(p, tplot, E; label = L"E(u)");
Plots.plot!(p, tplot, Eū; label = L"E(\bar{u})");
Plots.plot!(p, tplot, Eū + Eup; label = L"E(\bar{u}) + E(u')");
p

figsave(p, "convection/energy"; size = (400, 300))


##
# d = train;
d = test;
p1 = Plots.plot(; xlims = (a, b), xlabel = "x", title = "Solution (test data)");
j = 0
k = 9
i = 6
for k = [1, 3, 6, 9]
    # for i = 1:3
    j += 1
    lab = L"\bar{u}(%$(tstops[k]))"
    Plots.scatter!(x, d.ūₕ[k][:, i];
        # label = "$lab exact",
        label = nothing,
        color = j,
        markeralpha = 0.5
    )
    Plots.plot!(x, S(Aᴹ, d.ūₕ₀[:, i], tstops[k]); linestyle = :dot, label = nothing, color = j)
    Plots.plot!(x, S(Ā, d.ūₕ₀[:, i], tstops[k]); label = lab, color = j)
end
Plots.plot(p1)

p2 = Plots.plot(; xlims = (a, b), xlabel = L"x", title = "Filter width");
xline(x, y) = Plots.plot!(x, [y, y]; label = nothing, color = 1);
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
    Plots.scatter!([x], [i]; label = nothing, color = 1)
end
Plots.plot(p2)

Plots.plot(p1, p2; layout = (2, 1))
