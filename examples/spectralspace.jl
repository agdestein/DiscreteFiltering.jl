using LaTeXStrings

gr()

plotmat(Ā_ls)

Û = fft(Ū, 1)
∂Û∂t = fft(∂Ū∂t, 1)

plotmat(abs.(fft(Ā_ls)))

λ = 1e-8 * size(Ū, 2)
Â_ls = ∂Û∂t * Û' / (Û * Û' + λ * I)
Âᴹ = 2π * im * blockdiag(-spdiagm(collect(0:M÷2-1)), spdiagm(collect(M÷2-1:-1:0)))
M̂ = Â_ls - Âᴹ

DFT = [exp(-2π * im * (k - 1) * (m - 1) / M) for k = 1:M, m = 1:M]
IDFT = [1 / M * exp(2π * im * (k - 1) * (m - 1) / M) for m = 1:M, k = 1:M]
plotmat(angle.(DFT))
plotmat(real.(DFT))
plotmat(imag.(DFT))
plotmat(abs.(DFT * IDFT))
plotmat(imag.(Âᴹ))
plotmat(imag.(DFT * Aᴹ * IDFT))
plotmat(imag.(DFT * Ā_ls * IDFT))

plotmat(real.(Aᴹ))
plotmat(real.(DFT * Âᴹ * IDFT))
(DFT * Âᴹ * IDFT)[50, :]

plotmat(real.(Â_ls))
plotmat(imag.(Â_ls))
plotmat(imag.(Âᴹ))
plotmat(imag.(M̂))
plotmat(abs.(Â_ls))
plotmat(angle.(Â_ls))

path = "convection/fourier/kmax50"

figsave(plotmat(real.(Âᴹ); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\mathbf{A}}$, real part"), "$path/Ahat_real"; suffices = ("png",))
figsave(plotmat(imag.(Âᴹ); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\mathbf{A}}$, imaginary part"), "$path/Ahat_imag"; suffices = ("png",))
figsave(plotmat(abs.(Âᴹ); aspect_ratio = :equal, size = (450, 400), title = L"|\hat{\mathbf{A}}|"), "$path/Ahat_abs"; suffices = ("png",))
figsave(plotmat(angle.(Âᴹ); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\mathbf{A}}$, phase shift"), "$path/Ahat_angle"; suffices = ("png",))

figsave(plotmat(real.(Â_ls); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\bar{\mathbf{A}}}$, real part"), "$path/Abarhat_real"; suffices = ("png",))
figsave(plotmat(imag.(Â_ls); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\bar{\mathbf{A}}}$, imaginary part"), "$path/Abarhat_imag"; suffices = ("png",))
figsave(plotmat(abs.(Â_ls); aspect_ratio = :equal, size = (450, 400), title = L"|\hat{\bar{\mathbf{A}}}|"), "$path/Abarhat_abs"; suffices = ("png",))
figsave(plotmat(angle.(Â_ls); aspect_ratio = :equal, size = (450, 400), title = L"$\hat{\bar{\mathbf{A}}}$, phase shift"), "$path/Abarhat_angle"; suffices = ("png",))

