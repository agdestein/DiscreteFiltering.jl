if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

R = operators.int.R[i]
Ā_int = operators.int.Ā[i]
Ā_df = operators.df.Ā[i]
Ā_emb = operators.emb.Ā[i]

Ā_fourier = zeros(M, M)

plotmat(Ā_int)
plotmat(Ā_df)
plotmat(Ā_emb)

eigᴹ = eigen(Matrix(Aᴹ))
eigᴹ₂ = eigen(Matrix(Aᴹ₂))
eigᴹ₄ = eigen(Matrix(Aᴹ₄))
eigᴹ₆ = eigen(Matrix(Aᴹ₆))
eigᴹ₈ = eigen(Matrix(Aᴹ₈))
eigᴹ₁₀ = eigen(Matrix(Aᴹ₁₀))
eig_int = eigen(Ā_int)
eig_df = eigen(Ā_df)
eig_emb = eigen(Ā_emb)
eig_fourier = eigen(Ā_fourier)

inds = sortperm(imag.(eigᴹ₂.values))
norm.(eachcol(real.(eigᴹ₂.vectors[:, inds])))
plot(x, real.(eigᴹ₂.vectors[:, inds[3]]))
plotmat(imag.(eigᴹ₂.vectors[1:10, inds[1:10]]))
eigᴹ₂.values[inds[1:10]]

P = eigᴹ.vectors
P = eigᴹ₂.vectors
P = eigᴹ₄.vectors
P = eigᴹ₆.vectors
P = eigᴹ₈.vectors
P = eigᴹ₁₀.vectors
P = eig_int.vectors
P = eig_df.vectors
P = eig_emb.vectors
P = eig_fourier.vectors

mat = abs.(P'P)
plotmat(mat)

p = plot(; title = "Eigenvalues", legend = :bottomleft);
# scatter!(p, eigᴹ₁₀.values; marker = :diamond, label = L"\mathbf{A}_{10}^{(M)}");
# scatter!(p, eigᴹ₈.values; marker = :diamond, label = L"\mathbf{A}_8^{(M)}");
# scatter!(p, eigᴹ₆.values; marker = :diamond, label = L"\mathbf{A}_6^{(M)}");
# scatter!(p, eigᴹ₄.values; marker = :diamond, label = L"\mathbf{A}_4^{(M)}");
# scatter!(p, eigᴹ₂.values; marker = :diamond, label = L"\mathbf{A}_2^{(M)}");
scatter!(p, eigᴹ.values; marker = :diamond, label = L"\mathbf{A}^{(M)}");
scatter!(p, eig_int.values; marker = :rect, label = L"$\bar{\mathbf{A}}$, intrusive");
scatter!(p, eig_df.values; marker = :circle, label = L"$\bar{\mathbf{A}}$, derivative fit");
scatter!(p, eig_emb.values; marker = :xcross, label = L"$\bar{\mathbf{A}}$, embedded");
# scatter!(p, eig_fourier.values; marker = :xcross, label = L"$\bar{\mathbf{A}}$, Fourier");
p

p = plot(; legend = :topleft)
func = x -> sort(imag.(x))
# func = x -> real.(x[inds])
scatter!(p, func(eigᴹ.values); marker = :diamond, label = L"\mathbf{A}^{(M)}")
# scatter!(p, func(eigᴹ₂.values); marker = :diamond, label = L"\mathbf{A}_2^{(M)}")
# scatter!(p, func(eigᴹ₄.values); marker = :diamond, label = L"\mathbf{A}_4^{(M)}")
# scatter!(p, func(eigᴹ₆.values); marker = :diamond, label = L"\mathbf{A}_6^{(M)}")
# scatter!(p, func(eigᴹ₈.values); marker = :diamond, label = L"\mathbf{A}_8^{(M)}")
# scatter!(p, func(eigᴹ₁₀.values); marker = :diamond, label = L"\mathbf{A}_{10}^{(M)}")
scatter!(p, func(eig_int.values); marker = :rect, label = L"$\bar{\mathbf{A}}$, intrusive");
scatter!(
    p,
    func(eig_df.values);
    marker = :circle,
    label = L"$\bar{\mathbf{A}}$, derivative fit",
);
scatter!(p, func(eig_emb.values); marker = :xcross, label = L"$\bar{\mathbf{A}}$, embedded");
p

inds = 30
func = real
p = plot(; legend = :topleft)
plot!(p, x, func.(eigᴹ.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₂.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₄.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₆.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₈.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₁₀.vectors[:, inds]))
plot!(p, x, func.(eig_int.vectors[:, inds]))
plot!(p, x, func.(eig_df.vectors[:, inds]))
plot!(p, x, func.(eig_emb.vectors[:, inds]))
p

svdᴹ = svd(Matrix(Aᴹ))
svdᴹ₂ = svd(Matrix(Aᴹ₂))
svdᴹ₄ = svd(Matrix(Aᴹ₄))
svdᴹ₆ = svd(Matrix(Aᴹ₆))
svdᴹ₈ = svd(Matrix(Aᴹ₈))
svdᴹ₁₀ = svd(Matrix(Aᴹ₁₀))
svd_int = svd(Ā_int)
svd_df = svd(Ā_df)
svd_emb = svd(Ā_emb)

p = plot(; legend = :topright)
scatter!(p, svdᴹ.S; marker = :diamond, label = L"\mathbf{A}^{(M)}")
# scatter!(p, svdᴹ₂.S; marker = :diamond, label = L"\mathbf{A}_2^{(M)}")
# scatter!(p, svdᴹ₄.S; marker = :diamond, label = L"\mathbf{A}_4^{(M)}")
# scatter!(p, svdᴹ₆.S; marker = :diamond, label = L"\mathbf{A}_6^{(M)}")
# scatter!(p, svdᴹ₈.S; marker = :diamond, label = L"\mathbf{A}_8^{(M)}")
# scatter!(p, svdᴹ₁₀.S; marker = :diamond, label = L"\mathbf{A}_{10}^{(M)}")
scatter!(p, svd_int.S; marker = :rect, label = L"$\bar{\mathbf{A}}$, intrusive");
scatter!(p, svd_df.S; marker = :circle, label = L"$\bar{\mathbf{A}}$, derivative fit");
scatter!(p, svd_emb.S; marker = :xcross, label = L"$\bar{\mathbf{A}}$, embedded");
p
