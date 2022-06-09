if isdefined(@__MODULE__, :LanguageServer)
    include("dns.jl")
end

R = operators.exp.R[i]
Ā_exp = operators.exp.Ā[i]
Ā_ls = operators.ls.Ā[i]
Ā_int = operators.int.Ā[i]

eigᴹ₂ = eigen(Matrix(Aᴹ₂))
eigᴹ₄ = eigen(Matrix(Aᴹ₄))
eigᴹ₆ = eigen(Matrix(Aᴹ₆))
eigᴹ₈ = eigen(Matrix(Aᴹ₈))
eig_exp = eigen(Ā_exp)
eig_ls = eigen(Ā_ls)
eig_int = eigen(Ā_int)
eig_fourier = eigen(Ā_fourier)

inds = sortperm(imag.(eigᴹ₂.values))
norm.(eachcol(real.(eigᴹ₂.vectors[:, inds])))
plot(x, real.(eigᴹ₂.vectors[:, inds[3]]))
plotmat(imag.(eigᴹ₂.vectors[1:10, inds[1:10]]))
eigᴹ₂.values[inds[1:10]]

P = eigᴹ₂.vectors
P = eigᴹ₄.vectors
P = eigᴹ₆.vectors
P = eigᴹ₈.vectors
P = eig_exp.vectors
P = eig_ls.vectors
P = eig_int.vectors
P = eig_fourier.vectors

mat = abs.(P'P)
plotmat(mat)

p = plot(; title = "Eigenvalues", legend = :bottomleft);
scatter!(p, eigᴹ₈.values; marker = :diamond, label = L"\mathbf{A}_8^{(M)}");
scatter!(p, eigᴹ₆.values; marker = :diamond, label = L"\mathbf{A}_6^{(M)}");
scatter!(p, eigᴹ₄.values; marker = :diamond, label = L"\mathbf{A}_4^{(M)}");
scatter!(p, eigᴹ₂.values; marker = :diamond, label = L"\mathbf{A}_2^{(M)}");
scatter!(p, eig_exp.values; marker = :rect, label = L"$\bar{\mathbf{A}}$, explicit");
scatter!(p, eig_ls.values; marker = :circle, label = L"$\bar{\mathbf{A}}$, least squares");
scatter!(p, eig_int.values; marker = :xcross, label = L"$\bar{\mathbf{A}}$, intrusive");
# scatter!(p, eig_fourier.values; marker = :xcross, label = L"$\bar{\mathbf{A}}$, Fourier");
p

p = plot(; legend = :topleft)
func = x -> sort(imag.(x))
# func = x -> real.(x[inds])
scatter!(p, func(eigᴹ₂.values); marker = :diamond, label = L"\mathbf{A}_2^{(M)}")
scatter!(p, func(eigᴹ₄.values); marker = :diamond, label = L"\mathbf{A}_4^{(M)}")
scatter!(p, func(eigᴹ₆.values); marker = :diamond, label = L"\mathbf{A}_6^{(M)}")
scatter!(p, func(eigᴹ₈.values); marker = :diamond, label = L"\mathbf{A}_8^{(M)}")
scatter!(p, func(eig_exp.values); marker = :rect, label = L"$\bar{\mathbf{A}}$, explicit");
scatter!(p, func(eig_ls.values); marker = :circle, label = L"$\bar{\mathbf{A}}$, least squares");
scatter!(p, func(eig_int.values); marker = :xcross, label = L"$\bar{\mathbf{A}}$, intrusive");
p

inds = 30
func = real
p = plot(; legend = :topleft)
# plot!(p, x, func.(eigᴹ₂.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₄.vectors[:, inds]))
# plot!(p, x, func.(eigᴹ₆.vectors[:, inds]))
plot!(p, x, func.(eigᴹ₈.vectors[:, inds]))
plot!(p, x, func.(eig_exp.vectors[:, inds]))
plot!(p, x, func.(eig_ls.vectors[:, inds]))
plot!(p, x, func.(eig_int.vectors[:, inds]))
p

svdᴹ₂ = svd(Matrix(Aᴹ₂))
svdᴹ₄ = svd(Matrix(Aᴹ₄))
svdᴹ₆ = svd(Matrix(Aᴹ₆))
svdᴹ₈ = svd(Matrix(Aᴹ₈))
svd_exp = svd(Ā_exp)
svd_ls =  svd(Ā_ls)
svd_int = svd(Ā_int)

p = plot(; legend = :topright)
scatter!(p, svdᴹ₂.S; marker = :diamond, label = L"\mathbf{A}_2^{(M)}")
scatter!(p, svdᴹ₄.S; marker = :diamond, label = L"\mathbf{A}_4^{(M)}")
scatter!(p, svdᴹ₆.S; marker = :diamond, label = L"\mathbf{A}_6^{(M)}")
scatter!(p, svdᴹ₈.S; marker = :diamond, label = L"\mathbf{A}_8^{(M)}")
scatter!(p, svd_exp.S; marker = :rect, label = L"$\bar{\mathbf{A}}$, explicit");
scatter!(p, svd_ls.S; marker = :circle, label = L"$\bar{\mathbf{A}}$, least squares");
scatter!(p, svd_int.S; marker = :xcross, label = L"$\bar{\mathbf{A}}$, intrusive");
p
