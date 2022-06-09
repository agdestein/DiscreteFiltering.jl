R = operators.exp.R[i]
Ā_exp = operators.exp.Ā[i]
Ā_ls = operators.ls.Ā[i]
Ā_int = operators.int.Ā[i]

# eAᴹ[i] = relerr(Aᴹ, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_Aᴹ₂ = spectral_relerr(Aᴹ₂, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_Aᴹ₄ = spectral_relerr(Aᴹ₄, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_Aᴹ₆ = spectral_relerr(Aᴹ₆, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_Aᴹ₈ = spectral_relerr(Aᴹ₈, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_Aᴹ₁₀ = spectral_relerr(Aᴹ₁₀, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_exp = spectral_relerr(Ā_exp, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_ls = spectral_relerr(Ā_ls, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)
specerrs_int = spectral_relerr(Ā_int, test.ū, t_test; abstol = 1e-10, reltol = 1e-8)

# i = length(t_test)-1
for (i, t) in enumerate(t_test)
    p = plot(;
        # yscale = :log10,
        xlabel = "k",
        ylims = (0, 0.5),
    )
    # plot!(p, specerrs_Aᴹ₂[1:M÷2, i]; label = "Aᴹ₂");
    # plot!(p, specerrs_Aᴹ₄[1:M÷2, i]; label = "Aᴹ₄");
    # plot!(p, specerrs_Aᴹ₆[1:M÷2, i]; label = "Aᴹ₆");
    plot!(p, specerrs_Aᴹ₈[1:M÷2, i]; label = "Aᴹ₈");
    plot!(p, specerrs_Aᴹ₁₀[1:M÷2, i]; label = "Aᴹ₁₀")
    plot!(p, specerrs_exp[1:M÷2, i]; label = "Explicit");
    plot!(p, specerrs_ls[1:M÷2, i]; label = "Least squares")
    plot!(p, specerrs_int[1:M÷2, i]; label = "Intrusive")
    p
    display(p); sleep(0.1)
end
