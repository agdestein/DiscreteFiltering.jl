if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

gr()
pgfplotsx()

savedir = "figures/"

F = tophat
F = gaussian

R = operators.exp.R[i]
Ā_exp = operators.exp.Ā[i]
Ā_ls = operators.ls.Ā[i]
Ā_int = operators.int.Ā[i]

name = "W"
p = plotmat(
    W;
    # title = L"\mathbf{W}",
    aspect_ratio = N / M,
)

name = "R"
p = plotmat(
    R;
    # title = L"\mathbf{R}",
    aspect_ratio = M / N,
)

name = "AM"
p = plotmat(
    Aᴹ;
    # title = L"\mathbf{A}^{(M)}",
    aspect_ratio = :equal,
)

name = "Abar_exp"
p = plotmat(
    Ā_exp;
    # title = L"\bar{\mathbf{A}}^{\mathrm{exp}}",
    aspect_ratio = :equal,
)

name = "Abar_ls"
p = plotmat(
    Ā_ls;
    # title = L"\bar{\mathbf{A}}^{\mathrm{LS}}",
    aspect_ratio = :equal,
)

name = "Abar_int"
p = plotmat(Ā_int;
    # title = L"\bar{\mathbf{A}}^{\mathrm{int}}",
    aspect_ratio = :equal,
)

figsave(
    p,
    "matrices/$(name)_$(F.name)";
    savedir,
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
    plot!(p, ξ, x -> F.Ĝ(k, x); label = L"k = %$k")
end
p

p = plot(; xlabel = L"k", legend = :bottomleft);
k = LinRange(0, 50, 1000)
for x ∈ LinRange(0, 0.5, 6)
    plot!(p, k, k -> F.Ĝ(k, x); label = L"x = %$x")
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
        F.Ĝ.(k', x);
        zlims = (-0.25, 1),
        zticks = -0.25:0.25:1,
        xlabel = L"k",
        ylabel = L"x",
        legend = false,
        # title = "Local transfer function",
    )
end

figsave(p, "transfer_function_top_hat"; size = (400, 300))#, suffices = ("pdf",))
figsave(p, "transfer_function_gauss"; size = (400, 300))#, suffices = ("pdf",))

# Filtered and unfiltered solutions
i = 1
p = plot(;
    xlabel = L"x",
    legend = :bottomleft,
    legend_font_halign = :left,
    # title = "Filtered and unfiltered signals",
);
plot!(p, ξ, ξ -> u(c_train[:, i], ξ, 0.0); label = "Unfiltered");
plot!(p, ξ, ξ -> ū(tophat.Ĝ, c_train[:, i], ξ, 0.0); label = "Top-hat");
# plot!(p, ξ, ξ -> u(Φtophat * c_train[:, i], ξ, 0.0); label = "Top-hat");
plot!(p, ξ, ξ -> ū(gaussian.Ĝ, c_train[:, i], ξ, 0.0); label = "Gaussian");
# plot!(p, ξ, ξ -> u(Φgauss * c_train[:, i], ξ, 0.0); label = "Gaussian");
p

figsave(p, "initial_conditions"; size = (400, 300))#, suffices = ("pdf",))

W_tophat = W
R_tophat = R

W_gaussian = W
R_gaussian = R

i = 1
p = plot(;
    xlabel = L"x",
    legend = :topright,
    legend_font_halign = :left,
    # title = "Filtered and unfiltered signals",
);
plot!(p, ξ, dns_train.u[:, i, 1]; label = "Unfiltered");
plot!(p, x, W_tophat * dns_train.u[:, i, 1]; label = "Top-hat");
# plot!(p, x, W_gaussian * dns_train.u[:, i, 1]; label = "Gaussian");
plot!(p, ξ, R_tophat * W_tophat * dns_train.u[:, i, 1]; label = "Recon Top-hat");
# plot!(p, ξ, R_gaussian * W_gaussian * dns_train.u[:, i, 1]; label = "Recon Gaussian");
p

figsave(p, "initial_conditions_discrete_$M"; savedir = "output", suffices = ("pdf",))
# figsave(p, "initial_conditions_discrete_$M"; size = (400, 300))#, suffices = ("pdf",))

p = plot(; yscale = :log10);
scatter!(p, abs.(fft(Wtophat * dns_train.u[:, i, 1])); label = "Top-hat");
scatter!(p, abs.(fft(Wgauss * dns_train.u[:, i, 1])); label = "Gaussian");
p

# Spectra
i = 1
j = K+1:2K+1
# j = K+1:K+50
k = -K:K
c = c_test[:, i]
a = abs.(c)
a_tophat = abs.(Φ_tophat * c)
a_gaussian = abs.(Φ_gaussian * c)
yloglims = (-7, -1)
p = plot(;
    # xlims = (0, 1),
    xlabel = L"k",
    legend = :bottomleft,
    legend_font_halign = :left,
    # title = "ū(x+t, t)",
    # xscale = :log10,
    yscale = :log10,
    # xticks = -K:10:K,
    ylims = 10.0 .^ yloglims,
    yticks = 10.0 .^ range(yloglims...),
);
scatter!(p, k[j], a[j]; label = "Unfiltered");
scatter!(p, k[j], a_tophat[j]; label = "Top-hat");
scatter!(p, k[j], a_gaussian[j]; label = "Gaussian");
# vline!(p, [1 / 4h₀]; label = "Cutoff");
p

figsave(p, "coefficients"; size = (400, 300))

# Plot prediction at T/2
t = 1.5
iplot = 1:3
y⁻ = minimum(test.ū[:, iplot, :])
y⁺ = maximum(test.ū[:, iplot, :])
Δy = y⁺ - y⁻
p = plot(xlabel = "\$x\$", legend = false, xlims = (0, 1));
for (ii, i) ∈ enumerate(iplot)
    plot!(p, ξ, x -> ū(F.Ĝ, c_test[:, i], x, t); color = ii)
    # Ause = Aᴹ
    Ause = Ā_exp
    # Ause = Ā_ls
    # Ause = Ā_int
    # Ause = Ā_fourier
    scatter!(
        p,
        x,
        S!(Ause, test.ū[:, i, 1], [0, t])[end];
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
t_long = 10 .^ LinRange(-2, 2, 501);
# dns_long = create_data_dns(Aᴺ, c_long, ξ, t_long)
dns_long = create_data_exact(c_long, ξ, t_long)
long = create_data_filtered(W, Aᴺ, dns_long);
# e_long_Aᴹ = relerrs(Aᴹ, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Aᴹ₂ = relerrs(Aᴹ₂, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Aᴹ₄ = relerrs(Aᴹ₄, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Aᴹ₆ = relerrs(Aᴹ₆, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Aᴹ₈ = relerrs(Aᴹ₈, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_exp = relerrs(Ā_exp, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_ls = relerrs(Ā_ls, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_int = relerrs(Ā_int, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Ā_fourier = relerrs(Ā_fourier, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);

t₀, t₁ = extrema(t_train)
p = plot(;
    xlabel = L"t",
    # ylabel = "Mean relative error",
    xscale = :log10,
    yscale = :log10,
    legend = :bottomright,
    # xticks = 10.0 .^ (-2:2),
    yticks = 10.0 .^ (-8:0),
    xlims = extrema(t_long),
    ylims = (1e-3, 2.0e0),
    legend_font_halign = :left,
    minorgrid = true,
);
# vspan!(p, [t₀, t₁]; fillalpha = 0.1, label = "Training interval");
# vline!(p, t_train; label = "Training snapshots", linestyle = :dash);
# plot!(p, t_long, e_long_Aᴹ; label = L"\mathbf{A}^{(M)}");
plot!(p, t_long, e_long_Aᴹ₂; label = L"\mathbf{A}_2^{(M)}");
plot!(p, t_long, e_long_Aᴹ₄; label = L"\mathbf{A}_4^{(M)}");
plot!(p, t_long, e_long_Aᴹ₆; label = L"\mathbf{A}_6^{(M)}");
plot!(p, t_long, e_long_Aᴹ₈; label = L"\mathbf{A}_8^{(M)}");
plot!(p, t_long, e_long_Ā_exp; label = L"$\bar{\mathbf{A}}$, explicit");
plot!(p, t_long, e_long_Ā_ls; label = L"$\bar{\mathbf{A}}$, least squares");
plot!(p, t_long, e_long_Ā_int; label = L"$\bar{\mathbf{A}}$, intrusive");
# plot!(p, t_long, e_long_Ā_fourier; label = L"$\bar{\mathbf{A}}$, Fourier");
p

figsave(p, "comparison_$(F.name)"; savedir = "output", suffices = ("pdf",))
# figsave(p, "comparison_$(F.name)"; size = (400, 300))
