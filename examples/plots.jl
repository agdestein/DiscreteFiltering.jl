if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

gr()
pgfplotsx()

savedir = "figures/"

F = tophat
F = gaussian

R = operators.int.R[i]
Ā_int = operators.int.Ā[i]
Ā_df = operators.df.Ā[i]
Ā_emb = operators.emb.Ā[i]

sum(Aᴹ; dims = 2)
sum(Ā_int; dims = 2)
sum(Ā_df ; dims = 2)
sum(Ā_emb; dims = 2)

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

name = "Abar_int"
p = plotmat(
    Ā_int;
    # title = L"\bar{\mathbf{A}}^{\mathrm{int}}",
    aspect_ratio = :equal,
)

name = "Abar_df"
p = plotmat(
    Ā_df;
    # title = L"\bar{\mathbf{A}}^{\mathrm{LS}}",
    aspect_ratio = :equal,
)

name = "Abar_emb"
p = plotmat(
    Ā_emb;
    # title = L"\bar{\mathbf{A}}^{\mathrm{emb}}",
    aspect_ratio = :equal,
)

name = "Abar_int_diff"
p = plotmat(
    Ā_int - Aᴹ;
    # title = L"\bar{\mathbf{A}}^{\mathrm{emb}}",
    aspect_ratio = :equal,
)

name = "Abar_df_diff"
p = plotmat(
    Ā_df - Aᴹ;
    # title = L"\bar{\mathbf{A}}^{\mathrm{emb}}",
    aspect_ratio = :equal,
)

name = "Abar_emb_diff"
p = plotmat(
    Ā_emb - Aᴹ;
    # title = L"\bar{\mathbf{A}}^{\mathrm{emb}}",
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

pgfplotsx()

# Filter
p = plot(
    ξ,
    h;
    xlabel = L"x",
    ylims = (0.0, 0.03),
    legend = nothing,
    # title = L"Filter width $h(x)$",
)

figsave(p, "filter_width"; savedir, size = (400, 300))

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
span = x -> [x - h(x), x + h(x)]
vspan!(p, span(1/4); fillalpha = 0.1, color = 1, label = L"x \pm h(x)");
vspan!(p, span(3/4); fillalpha = 0.1, color = 1, label = nothing);
plot!(p, ξ, ξ -> u(c_train[:, i], ξ, 0.0); color = 1, label = "Unfiltered");
plot!(p, ξ, ξ -> ū(tophat.Ĝ, c_train[:, i], ξ, 0.0); color = 2, label = "Top-hat");
# plot!(p, ξ, ξ -> u(Φtophat * c_train[:, i], ξ, 0.0); label = "Top-hat");
plot!(p, ξ, ξ -> ū(gaussian.Ĝ, c_train[:, i], ξ, 0.0); color = 3, label = "Gaussian");
# plot!(p, ξ, ξ -> u(Φgauss * c_train[:, i], ξ, 0.0); label = "Gaussian");
p

figsave(p, "initial_conditions"; savedir, size = (400, 300))#, suffices = ("pdf",))

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
j = K+2:2K+1
# j = K+1:K+50
k = -K:K
c = c_test[:, i]
a = abs.(c)
a_tophat = abs.(Φ_tophat * c)
a_gaussian = abs.(Φ_gaussian * c)
yloglims = (-8, -1)
p = plot(;
    # xlims = (0, 1),
    xlabel = L"k",
    legend = :bottomleft,
    legend_font_halign = :left,
    # title = "ū(x+t, t)",
    xscale = :log10,
    yscale = :log10,
    # xticks = -K:10:K,
    ylims = 10.0 .^ yloglims,
    yticks = 10.0 .^ range(yloglims...),
    minorgrid = true,
);
scatter!(p, k[j], a[j]; label = "Unfiltered");
scatter!(p, k[j], a_tophat[j]; label = "Top-hat");
scatter!(p, k[j], a_gaussian[j]; label = "Gaussian");
# vline!(p, [1 / 4h₀]; label = "Cutoff");
p

figsave(p, "coefficients"; savedir, size = (400, 300))
figsave(p, "coefficients_log"; savedir, size = (400, 300))

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
    Ause = Ā_int
    # Ause = Ā_df
    # Ause = Ā_emb
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

figsave(p, "initial"; savedir, size = (400, 300))
figsave(p, "final"; savedir, size = (400, 300))

# Long term evolution
c_disp = reduce(hcat, [create_signal(K)])
t_disp = LinRange(0, 100, 2001);
# dns_disp = create_data_dns(Aᴺ, c_disp, ξ, t_disp)
dns_disp = create_data_exact(c_disp, ξ, t_disp)
disp = create_data_filtered(W, Aᴺ, dns_disp);
# disp_Aᴹ = S!(Aᴹ, disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Aᴹ₂   = S!(Aᴹ₂,   disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Aᴹ₄   = S!(Aᴹ₄,   disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Aᴹ₆   = S!(Aᴹ₆,   disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Aᴹ₈   = S!(Aᴹ₈,   disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Aᴹ₁₀  = S!(Aᴹ₁₀,  disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Ā_int = S!(Ā_int, disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Ā_df  = S!(Ā_df,  disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
disp_Ā_emb = S!(Ā_emb, disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);
# disp_Ā_fourier = S!(Ā_fourier, disp.ū[:, :, 1], t_disp; abstol = 1e-10, reltol = 1e-8);

# shift(x, t) = x
shift(x, t) = mod(x - t, T)
function plotsort!(p, x, y; kwargs...)
    I = sortperm(x)
    plot!(p, x[I], y[I]; kwargs...)
end

ylims = extrema(dns_disp.u[:, :, 1])
# for (it, t) ∈ enumerate(t_disp)
for (it, t) ∈ collect(enumerate(t_disp))[1:200]
    p = plot(;
        # xlabel = "x",
        xlabel = "x - t",
        xlims = (0, 1),
        ylims,
        title = @sprintf("t = %.2f", t),
    )
    plotsort!(p, shift.(ξ, t), dns_disp.u[:, :, it]; label = "Unfiltered")
    plotsort!(p, shift.(x, t), disp.ū[:, :, it]; label = "Filtered")
    # plotsort!(p, shift.(x, t), disp_Aᴹ₂[it]; label = "Aᴹ₂")
    # plotsort!(p, shift.(x, t), disp_Aᴹ₄[it]; label = "Aᴹ₄")
    # plotsort!(p, shift.(x, t), disp_Aᴹ₆[it]; label = "Aᴹ₆")
    # plotsort!(p, shift.(x, t), disp_Aᴹ₈[it]; label = "Aᴹ₈")
    # plotsort!(p, shift.(x, t), disp_Ā_int[it]; label = "Ā_int")
    plotsort!(p, shift.(x, t), disp_Ā_df[it]; label = "Ā_df")
    # plotsort!(p, shift.(x, t), disp_Ā_emb[it]; label = "Ā_emb")
    display(p); sleep(0.02)
end

û = abs.(fft(dns_disp.u[:, 1, 1])[1:K]) / N
for (it, t) ∈ collect(enumerate(t_disp))[1:200]
    p = plot(;
        xlabel = "k",
        legend = :bottomleft,
        yscale = :log10,
        # xlims = (-20, 20),
        ylims = (1e-7, 1e0),
        title = "Fourier coefficients, t = $(@sprintf "%0.2f" t)",
        # size = (400, 300),
        # dpi = 200,
    )
    scatter!(p, 1:K, û; label = "Unfiltered", marker = :circle)
    # sticks!(
    #     p, k, func.(Φ * (Et .* c[:, i]));
    #     label = "Filtered",
    #     marker = :c,
    # )
    # scatter!(p, k[ik], func.(Φ_tophat * (Et .* c[:, i]))[ik]; label = "Top-Hat", marker = :c)
    # scatter!(p, k[ik], func.(Φ_gaussian * (Et .* c[:, i]))[ik]; label = "Gaussian", marker = :c)
    k = 1:min(M ÷ 2, K)
    # data = disp_Ā_int
    # data = disp_Ā_df
    data = disp_Ā_emb
    scatter!(p, k, abs.(fft(data[it])[k]) / M; label = "Filtered", marker = :rect)
    scatter!(p, 1:K, abs.(fft(R * data[it])[1:K]) / N; label = "Reconstruction", marker = :diamond)
    p
    display(p); sleep(0.005)
end

# Error evolution

R = operators.int.R[i]
Ā_int = operators.int.Ā[i]
Ā_df = operators.df.Ā[i]
Ā_emb = operators.emb.Ā[i]

long = create_data_filtered(W, Aᴺ, dns_long);
# long.ū[:, :, 1] .+= 1e-4 .* randn.()
e_long_Aᴹ = relerrs(Aᴹ, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Aᴹ₂ = relerrs(Aᴹ₂, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Aᴹ₄ = relerrs(Aᴹ₄, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Aᴹ₆ = relerrs(Aᴹ₆, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Aᴹ₈ = relerrs(Aᴹ₈, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Aᴹ₁₀ = relerrs(Aᴹ₁₀, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_int = relerrs(Ā_int, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Ā_int = relerrs(Ā_int + 1e-7 * Dᴹ, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_df = relerrs(Ā_df, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
e_long_Ā_emb = relerrs(Ā_emb, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);
# e_long_Ā_fourier = relerrs(Ā_fourier, long.ū, t_long; abstol = 1e-10, reltol = 1e-8);

iplot = 2:length(t_long)
t₀, t₁ = extrema(t_train[2:end])
p = plot(;
    xlabel = L"t",
    # ylabel = "Mean relative error",
    xscale = :log10,
    yscale = :log10,
    legend = :topleft,
    xlims = extrema(t_long[iplot]),
    # xticks = 10.0 .^ (-2:2),
    # ylims = (6e-3, 1.0e0),
    ylims = (4e-4, 1.0e0),
    # yticks = 10.0 .^ (-2:0),
    yticks = 10.0 .^ (-3:0),
    legend_font_halign = :left,
    minorgrid = true,
);
vspan!(p, [t₀, t₁]; fillalpha = 0.1, label = "Training interval");
plot!(p, t_long[iplot], e_long_Aᴹ[iplot]; color = 1, label = L"\mathbf{A}^{(M)}");
# plot!(p, t_long[iplot], e_long_Aᴹ₂[iplot]; label = L"\mathbf{A}_2^{(M)}");
# plot!(p, t_long[iplot], e_long_Aᴹ₄[iplot]; label = L"\mathbf{A}_4^{(M)}");
# plot!(p, t_long[iplot], e_long_Aᴹ₆[iplot]; label = L"\mathbf{A}_6^{(M)}");
# plot!(p, t_long[iplot], e_long_Aᴹ₈[iplot]; label = L"\mathbf{A}_8^{(M)}");
# plot!(p, t_long[iplot], e_long_Aᴹ₁₀[iplot]; label = L"\mathbf{A}_{10}^{(M)}");
plot!(p, t_long[iplot], e_long_Ā_int[iplot]; color = 2, label = L"$\bar{\mathbf{A}}$, intrusive");
plot!(p, t_long[iplot], e_long_Ā_df[iplot]; color = 3, label = L"$\bar{\mathbf{A}}$, derivative fit");
plot!(p, t_long[iplot], e_long_Ā_emb[iplot]; color = 4, label = L"$\bar{\mathbf{A}}$, embedded");
# plot!(p, t_long[iplot], e_long_Ā_fourier[iplot]; label = L"$\bar{\mathbf{A}}$, Fourier");
p

# figsave(p, "comparison_$(F.name)"; savedir = "output", suffices = ("pdf",))
figsave(p, "comparison_$(F.name)"; savedir, size = (400, 300))

pgfplotsx()
