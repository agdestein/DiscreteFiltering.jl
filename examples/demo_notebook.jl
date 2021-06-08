### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ cc1f58dc-b599-4bf4-a1da-bdd62d7b26d4
using PlutoUI

# ╔═╡ f6d1766a-ee6c-4f88-825c-903325adcbb7
using LinearAlgebra

# ╔═╡ 828c4651-1bac-4f10-bc9b-582cdb7f677d
using SparseArrays

# ╔═╡ f245bd45-3519-482b-948c-27b42f8bb5c7
using Polynomials

# ╔═╡ 5cb7375b-ee81-4243-81f8-c195f1335bda
using Intervals

# ╔═╡ db73cbe7-c658-4973-8da2-1dafe1353eeb
using Zygote

# ╔═╡ eb766d7f-9b10-4d16-8d4e-5dae1ac6d643
using DifferentialEquations

# ╔═╡ 77a01981-5513-49ba-8bb8-6972b2e22e14
using Plots

# ╔═╡ 0f52e8af-7413-4449-9b6a-53be3dc0acfc
md"# Discrete filtering of advection equation"

# ╔═╡ 3fb274d2-6911-4cec-b251-85140a249f6c
md"## Imports"

# ╔═╡ fdbc96a2-586f-488a-96f4-5acd923615d0
md"## Set plotting backend"

# ╔═╡ b0026f45-8281-47bd-ad7e-a96ced4d96b2
md"`GR` (enabled by default) is the fastest, while `Plotly` is more interactive."

# ╔═╡ e6a7eba4-01b9-45ca-824f-45f2643b89e0
# gr()
# plotly()
# pyplot()

# ╔═╡ 62031239-19c1-4aa4-a67c-f0e6f0093578
md"## Time"

# ╔═╡ acab5f55-7013-4932-9637-bd482f3ae49d
@bind T Slider(0.1:0.1:5.0, default = 1.0)

# ╔═╡ ac526d65-ab65-4336-818f-8e3eed50f813
T

# ╔═╡ 07717f08-8811-4308-a580-9add99cbca06
md"## Filter"

# ╔═╡ 896d4fe7-f66f-4998-bad1-3a499e354c37
md"""
The filtering operation on a function ``\varphi : [0, 2 \pi] \to \mathbb{R}`` is parameterized by a kernel ``G``:

$$\overline{\varphi}(x) = \int_0^{2 \pi} G(x, \xi) \varphi(\xi) \, \mathrm{d} \xi$$

Here we will consider a circular top-hat filter ``G_h``,  parameterized by a variable filter width ``2h``:

$$G_h(x, \xi) =
\begin{cases}
	\frac{1}{2 h(x)} & \text{if}\ |x - \xi - 2 \pi i| \leq h(x), \quad i \in \{-1, 0, 1\}\\
	0 & \text{otherwise}
\end{cases}$$
"""

# ╔═╡ 1b0e9fcc-0bbe-493a-93ec-a23f892684a2
@bind log_h₀ Slider(-3.0:0.1:0.0, default = -1.0)

# ╔═╡ 5b3006a0-2750-4876-9732-7f5bd1cca904
log_h₀

# ╔═╡ 5a5fbe7d-732b-4eee-ae27-fa98290163af
h₀ = 10.0 ^ log_h₀

# ╔═╡ 6f907412-7a84-4a9e-b247-75eb7cf08570
h(x) = h₀ * (1 - 1 / 2 * cos(x));

# ╔═╡ d9b23fba-0b5a-44bd-b6cd-652deaea0410
#dh(x) = h₀ / 2 * sin(x);

# ╔═╡ 5a5ed070-d0d3-4bdb-8ef7-fdfc6206637c
α(x) = 1 / 3 * h'(x) * h(x);

# ╔═╡ 1b17586c-dbcc-40fe-851d-8a60c4885cf1
plot(0:0.1:2π, h); ylims!((0, ylims()[2]))

# ╔═╡ 7759cc59-df9d-4735-a777-38a231503d7f
md"## Discretization"

# ╔═╡ 305c5464-0117-4752-a51a-3332e76e9f9b
n = 500

# ╔═╡ 95bdf52c-dba4-4176-8070-6039659db40c
x = LinRange(2π / n, 2π, n)

# ╔═╡ 79094bcc-233a-452d-8b4f-95f47452d168
Δx = x[2] - x[1]

# ╔═╡ dbf1ed2f-41eb-4e29-ad91-8843a5c9c01f
begin
	plot(x, abs.(α.(x)), label = "|α(x)|")
	plot!([x[1], x[end]], [Δx/2, Δx/2], label = "Δx/2")
end

# ╔═╡ a78e0d6c-b0a7-419e-96f9-afd611913457
md"## Advection matrix"

# ╔═╡ d267025d-19b4-4edb-9d28-80ae94d9b58f
md"""The matrix ``C`` approximates the advection operator ``\frac{\partial}{\partial x}`` using a central difference scheme with circular boundary condions:

$$C = \frac{1}{2 \Delta x}
\begin{pmatrix}
0  & 1      &        &        & -1 \\
-1 & 0      & 1      &        &    \\
   & \ddots & \ddots & \ddots &    \\
   &        & -1     & 0      & 1  \\
1  &        &        & -1     & 0
\end{pmatrix}$$
"""


# ╔═╡ dc992c13-6ba3-491d-818f-ebc45eb21643
begin
	C = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(1.0, n - 1))
	C[1, end] = -1.0
	C[end, 1] = 1.0
	C ./= 2Δx
end

# ╔═╡ 43eaa3ee-e66b-4afe-ba8f-b4b5ac75038c
md"## Diffusion matrix"

# ╔═╡ 20f3a297-4a7b-42ae-aa88-28330200a05b
md"""The matrix ``D`` approximates the diffusion operator ``\frac{\partial^2}{\partial x^2}`` using a central difference scheme with circular boundary condions:

$$D =
\frac{1}{\Delta x^2}
\begin{pmatrix}
-2 & 1      &        &        & 1  \\
1  & -2     & 1      &        &    \\
   & \ddots & \ddots & \ddots &    \\
   &        & 1      & -2     & 1  \\
1  &        &        & 1      & -2
\end{pmatrix}$$
"""

# ╔═╡ 0a677943-7f22-444a-9807-1c3f1b2f4f1b
begin
	D = spdiagm(-1 => fill(1.0, n - 1), 0 => fill(-2.0, n), 1 => fill(1.0, n - 1))
	D[1, end] = 1.0
	D[end, 1] = 1.0
	D ./= Δx^2
end

# ╔═╡ bac7ddbd-753a-4c0a-b219-208f7f2fbd93
md"We also define the diagonal operator ``A = \operatorname{diag}(\alpha(x_1), \dots, \alpha(x_n))``"

# ╔═╡ 039dfa5d-1ac9-481f-909f-71c5973d30c5
A = spdiagm(α.(x))

# ╔═╡ 84d04b1d-1ddf-4b1a-86e0-53080e6514c9
md"## Filtering matrix"

# ╔═╡ cbad822c-df13-4f47-a201-1dc122833bd0
md"### Chebyshev polynomials"

# ╔═╡ f74e93ac-66f0-4fce-85c5-a0fbfe670c18
p = ChebyshevT([1.8, 0, 3, 0, 0, 0.7])

# ╔═╡ 3266dc74-c83e-47f1-8005-7c4fa9c24ccd
p[0]

# ╔═╡ 711ffa2f-4459-42c8-98c1-073d6f3c6e53
p[2]

# ╔═╡ a0db65dd-27cf-4fb7-9a56-2ffb470d78fb
integrate(p)

# ╔═╡ 0f5893c5-96c4-40d3-8653-bf1ed478cd1d
integrate(p, -1, 1)

# ╔═╡ f8df05ca-0e13-4cc1-9571-e9033e4ba301
begin
	pl = plot()
	for i = 1:5
		coeffs = zeros(i)
		coeffs[i] = 1
		plot!(pl, ChebyshevT(coeffs))
	end
	pl
end

# ╔═╡ aaa8e1d5-8d72-4dac-a1d6-d5aab59141f2
van = vander(ChebyshevT, -1:0.2:1, 4)

# ╔═╡ f8b2926e-7bf6-44d9-a13c-6e1c0e53f2ce
begin
	pl2 = plot()
	for i = 1:5
		plot!(pl2, -1:0.2:1, van[:, i])
	end
	pl2
end

# ╔═╡ f6732413-b031-4391-9eb8-fe1f6da932bb
mod(-3.4, 3)

# ╔═╡ c5cfc1dc-16c4-49ec-af03-b570ff7dad2a
md"""
We construct a scaling operator ``\tau: 2 \pi \to [-1, 1]``
"""

# ╔═╡ 1267f4f0-bf89-4063-858f-fc545ea626ef
# τ(x) = (mod(x, 2π)  - π) / 2π;
τ(x) = (x - π) / 2π;

# ╔═╡ 15827d5b-a356-42c8-98b7-faac05bebf16
τ(x, a, b) = (x - (a + b) / 2) / (b - a);

# ╔═╡ 0ab43c8e-cfaf-44c4-b8e1-05141473b050
τ(π / 2)

# ╔═╡ 62e746e1-3eec-4051-bf27-870b8e1c1448
τ(π / 2, 0, 2π)

# ╔═╡ 6ae8033a-c4cd-46b5-8df6-539afb6380b3
τ(π / 2, π / 4, π)

# ╔═╡ 90b92d5d-603a-40ca-b3f1-bc562cb41704
#@bind degmax Slider(1:20, default = 20)
degmax = 1000

# ╔═╡ a9157042-7f60-4d0b-b35f-d12d51322110
degmax

# ╔═╡ 5f1650bf-c798-41ac-9a65-0b60da1a858e
ϕ = [ChebyshevT([fill(0, i); 1]) for i = 0:degmax]

# ╔═╡ 9183a6a9-bfc7-403e-99b5-72c56a396fb7
begin
	W = spzeros(n, n)
	for i = 1:n
		# Point
		xᵢ = x[i]
		
		# Filter width at point
		hᵢ = h(xᵢ)
		
		# Indices of integration points in circular reference
		I = Interval(xᵢ - hᵢ, xᵢ + hᵢ)
		inds_left = x .∈ I + 2π
		inds_mid = x .∈ I
		inds_right = x .∈ I - 2π
		inds = inds_left .| inds_mid .| inds_right
		deg = min(sum(inds) - 1, degmax)
		
		# Polynomials evaluated at integration points
		Vᵢ = vander(ChebyshevT, τ.(x[inds]), deg)'
		# Vᵢ = vander(ChebyshevT, τ.(x[inds], xᵢ - hᵢ, xᵢ + hᵢ), deg)'
		
		# Polynomial moments around point
		Domain = Interval(0, 2π)
		I_left = (I + 2π) ∩ Domain
		I_mid = I ∩ Domain
		I_right = (I - 2π) ∩ Domain
		μᵢ = integrate.(ϕ[1:deg+1], τ(I_left.first), τ(I_left.last))
		μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_mid.first), τ(I_mid.last))
		μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_right.first), τ(I_right.last))
		μᵢ = integrate.(ϕ[1:deg+1], τ(I.first), τ(I.last))
		# μᵢ = integrate.(ϕ[1:deg+1], -1, 1)
		μᵢ .*= 2π / 2hᵢ
		# μᵢ .*= 2hᵢ / 2π
		# μᵢ ./= 2

		# Fit weights
		npoint = size(Vᵢ, 2)
		wᵢ = fill(1 / npoint, npoint)
		try
			wᵢ = Vᵢ \ μᵢ
		catch e
			println(i)
		end
		
		# Store weights
		W[i, inds] .= wᵢ
	end
end
# begin
# 	W = spdiagm(-1 => fill(1.0, n - 1), 0 => fill(4.0, n), 1 => fill(1.0, n - 1))
# 	W[1, end] = 1
# 	W[end, 1] = 1
# 	W ./= 6
# end

# ╔═╡ 7b8aab02-0d7f-4f71-9fc8-49dbd9ae2064
W

# ╔═╡ 772cec35-2650-44da-b502-9352ed4f862e
spy(W)

# ╔═╡ 363f1fc6-0736-45bf-aabb-3e6757cad0df
row = n ÷ 2

# ╔═╡ a55d08ca-3b43-48bd-b749-a89178531a9a
filter(!isequal(0), W[row, :])

# ╔═╡ 661bdd7e-5050-4159-9aea-1c32fe51880b
sum(filter(!isequal(0), W[row, :]))

# ╔═╡ e8871c70-b93e-11eb-3d28-6b2401998411
md"## Exact solutions"

# ╔═╡ 5864dd46-bcc8-4e2b-94a1-e1ba1285fb71
u(x, t) = sin(x - t) + 0.6cos(5(x - t)) + 0.04sin(20(x - 1 - t));

# ╔═╡ 5bfd2fc7-1f35-4049-b211-31f888bc4b26
u_int(x, t) = -cos(x - t) + 0.6/5 * sin(5(x - t)) - 0.04/20 * cos(20(x - 1 - t));

# ╔═╡ 2767529e-c479-4075-b15b-27b5791894e8
ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t));

# ╔═╡ bc1a6782-81ac-4f4f-92f5-d36690223e16
md"## Discrete initial conditions"

# ╔═╡ 81705bc9-b42a-4276-bd1b-ee8e2aa21b1e
uₕ = u.(x, 0.0);

# ╔═╡ 5fdb8837-fe1c-4508-bb2c-fea65d1a081a
ūₕ = ū.(x, 0.0);

# ╔═╡ 0d9c11df-25d2-47da-8435-db053d453d2f
uₕ_allbar = W * uₕ;

# ╔═╡ 5d8918cc-9d3d-4eb9-88a5-a346155a31b1
begin
	plot(x, uₕ, label = "Discretized")
	plot!(x, ūₕ, label = "Filtered-then-discretized")
	plot!(x, uₕ_allbar, label = "Discretized-then-filtered")
end

# ╔═╡ 2a057ba9-f6e0-42a2-9254-95bfacbd2e22
md"## ODE right-hand side functions"

# ╔═╡ 9cb3eedd-7dd4-4a6b-b4e8-0315eaaee25c
md"""
The continuous equation is defined by

$$\frac{\partial u}{\partial{t}} + \frac{\partial u}{\partial x} = 0$$

with initial conditions ``u(x, 0) = u_0(x)`` and periodic boundary conditions ``u(0, t) = u(2 \pi, t)``. A possible discrete formulation is

$$\frac{\mathrm{d} u_h}{\mathrm{d} t} + C u_h = 0.$$

Filtering the discretized solution ``u_h`` gives ``\overline{u_h} = W u_h``. The discretized-then-filtered solution satisfies the following equation:

$$\frac{\mathrm{d} \overline{u_h}}{\mathrm{d} t} + W C W^{-1} \overline{u_h} = 0.$$


The filtered continuous equation is given by

$$\frac{\partial \overline{u}}{\partial{t}} + \frac{\partial \overline{u}}{\partial x} = \alpha(x) \frac{\partial^2 \overline{u}}{\partial x^2}$$

with initial conditions ``\overline{u}(x, 0) = \overline{u_0}(x)`` and periodic boundary conditions ``\overline{u}(0, t) = \overline{u}(2 \pi, t)``. Applying the discretization scheme after filtering gives

$$\frac{\mathrm{d} \overline{u}_h}{\mathrm{d} t} + C \overline{u}_h = A D \overline{u}.$$

"""

# ╔═╡ 60156fd6-468c-409a-b476-fd9c39b8d27a
∂uₕ∂t(uₕ, p, t) = -C * uₕ;

# ╔═╡ de0c8598-3683-426a-a2a6-ac90c660059f
∂ūₕ∂t(ūₕ, p, t) = (-C + A * D) * ūₕ;

# ╔═╡ 2ec9b609-9fc3-4cad-9e18-46ea8cc80f94
∂uₕ_allbar∂t(uₕ_allbar, p, t) = -W * (C * (W \ uₕ_allbar));

# ╔═╡ ede977d7-6d96-4136-92ec-cbd2941dd54b
md"## Solve discretized problem"

# ╔═╡ e2da6398-3532-4688-aedc-4d39f3cc8892
prob = ODEProblem(∂uₕ∂t, uₕ, (0, T));

# ╔═╡ a7425523-5cb7-4a63-8fc4-6e3ed8df0b74
sol = solve(prob, abstol = 1e-6, reltol = 1e-4);

# ╔═╡ 31d56864-e1ae-42d1-b9db-0ada73f939fd
md"### Plot solution"

# ╔═╡ ce8f746f-f0ac-4e27-a899-baa751a52982
@bind t₁ Slider(LinRange(0.0, T, 100), default = T)

# ╔═╡ 6eb69dd7-7657-41e0-87eb-23d6818b448e
t₁

# ╔═╡ 4c65f35f-49c1-439f-ba90-82582ff6728b
begin
	plot(x, uₕ, label = "Initial conditions")
	plot!(x, sol(t₁), label = "Discretized")
	plot!(x, u.(x, t₁), label = "Exact")
end

# ╔═╡ e19206b7-6eda-4864-b4cd-00d020ed8453
md"## Solve filtered-and-then-discretized problem"

# ╔═╡ 4bbb99d4-eeb9-4668-a205-465862747d25
prob_bar = ODEProblem(∂ūₕ∂t, ūₕ, (0, T));

# ╔═╡ e1d6d80f-52d3-4a80-81cb-dd8e6bfbcc12
sol_bar = solve(prob_bar, abstol = 1e-6, reltol = 1e-4);

# ╔═╡ f950a27f-c4ad-4c5c-9396-cf06d2b986f4
Δx^2 / maximum(abs.(α.(x)))

# ╔═╡ 01716cfe-2a2a-431a-9b8a-fe4f24850929
sol_bar

# ╔═╡ 17327732-f704-4a3d-84bf-4c00fc7022c4
md"### Plot solution"

# ╔═╡ f9a079f2-c1c4-4e49-9632-082d6a9892c0
@bind t₂ Slider(LinRange(0.0, T, 100), default = T)

# ╔═╡ fa0ab96d-70c8-4480-9deb-7ede4d1dee79
t₂

# ╔═╡ bf52219b-7e63-4f3d-a2f4-1e25d9580666
begin
	plot(x, uₕ, label = "Initial")
	plot!(x, ūₕ, label = "Initial filtered")
	plot!(x, sol_bar(t₂), label = "Filtered-discretized")
	plot!(x, ū.(x, t₂), label = "Filtered-exact")
	#plot!(x, 500*α.(x))
end

# ╔═╡ 2e8afcff-1de4-48df-92c9-f75944f9c125
md"## Solve discretized-and-then-filtered problem"

# ╔═╡ d3f2c2dc-cd06-4f97-8aae-884ff1bbb329
prob_allbar = ODEProblem(∂uₕ_allbar∂t, W * uₕ, (0, T));

# ╔═╡ 9d13a2a8-b2c4-4437-bffe-655f6c87cf41
sol_allbar = solve(prob_allbar, abstol = 1e-6, reltol = 1e-4);

# ╔═╡ 09898683-742d-48f1-b4e1-3d07ea8f52b8
md"### Plot solution"

# ╔═╡ f94099b1-b213-4c27-8bd0-77390fd9ad15
@bind t₃ Slider(LinRange(0.0, T, 100), default = T)

# ╔═╡ 9cdab2f4-1c1f-4594-a524-cfc85ac86c34
t₃

# ╔═╡ 1a1e2112-a2b0-4806-b522-c328943229fc
begin
	plot(x, uₕ, label = "Initial")
	plot!(x, uₕ_allbar, label = "Initial discretized-then-filtered")
	plot!(x, sol_allbar(t₃), label = "Discretized-then-filtered")
	plot!(x, W * u.(x, t₃), label = "Exact")
end

# ╔═╡ 3f0a5879-0d40-45c5-b657-d4d5e8f858a3
md"## Comparison"

# ╔═╡ defcf455-75d2-41fc-a1bf-4ce7f73bbe43
@bind t₄ Slider(LinRange(0.0, T, 100), default = T)

# ╔═╡ 1f80eb67-bb7d-4921-b933-6fad29ad3b23
t₄

# ╔═╡ a3d1ef1a-c991-4d4d-af34-cf4922cb1d92
md"### Solutions"

# ╔═╡ c720697f-3773-4c27-802c-ee596888073b
begin
	plot(x, uₕ, label = "Initial")
	plot!(x, ūₕ, label = "Initial filtered")
	plot!(x, sol(t₄), label = "Discretized")
	plot!(x, sol_bar(t₄), label = "Filtered-then-discretized")
	plot!(x, sol_allbar(t₄), label = "Discretized-then-filtered")
	# plot!(x, [u.(x, t₄), ū.(x, t₄)], label = "Exact")
	ylims!(minimum(uₕ), maximum(uₕ))
end

# ╔═╡ 3da2f0ec-fa87-4b69-88b9-fcc251a2b1b8
md"### Relative error"

# ╔═╡ b62dd663-9d31-4563-8ded-ab697e77a0bd
begin
	u_exact = u.(x, t₄)
	ū_exact = ū.(x, t₄)
	err = abs.(sol(t₄) - u_exact) ./ maximum(abs.(u_exact))
	err_bar = abs.(sol_bar(t₄) - u_exact) ./ maximum(abs.(u_exact))
	err_allbar = abs.(sol_allbar(t₄) - u_exact) ./ maximum(abs.(u_exact))
	plot()
	plot!(x, err, label = "Unfiltered discretized")
	plot!(x, err_bar, label = "Filtered-then-discretized")
	plot!(x, err_allbar, label = "Discretized-then-filtered")
end

# ╔═╡ d9ef66cb-8c6f-40ec-8abf-c57ba7404381
md"# Extension to a non-linear case: Burgers equation"

# ╔═╡ 8416c7c2-316a-41b0-a66e-0fc217358d6b
md"""
The Burgers equation is given by

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0.$$

A possible discrete formulation is given by

$$\frac{\mathrm{d} u_h}{\mathrm{d} t} + U_h C u_h - \nu D u_h = 0,$$

where ``U_h = \operatorname{diag}(u_h)``.

Applying a discrete filter after discretization results in the following equation (for the discretized-then-filtered solution ``\overline{u_h}``):

$$\frac{\mathrm{d} \overline{u_h}}{\mathrm{d} t} + W \operatorname{diag}(W^{-1} \overline{u_h}) C W^{-1} u_h - \nu W D W^{-1} u_h = 0.$$

In contrast, the filtered continuous equation reads

$$\frac{\partial \overline{u}}{\partial t} + \overline{u \frac{\partial u}{\partial x}} - \nu \overline{\frac{\partial^2 u}{\partial x^2}} = 0.$$
"""

# ╔═╡ e7d65165-bbf5-43f4-893d-11744572b96f
ν = 0.03

# ╔═╡ 186a39d7-bf10-4729-9181-dd2b6ccffbe8
burgers_∂uₕ∂t(uₕ, p, t) = -uₕ .* (C * uₕ) + ν * D * uₕ;

# ╔═╡ 5b4186f8-94f6-45ca-ab22-afc25bd6ac58
function burgers_∂uₕ_allbar∂t(uₕ_allbar, p, t)
	uₕ = W \ uₕ_allbar
	-W * (uₕ .* (C * uₕ)) + ν * W * D * uₕ
end;

# ╔═╡ 80582fde-553c-4df4-bceb-52aa48ae66eb
burgers_prob = ODEProblem(burgers_∂uₕ∂t, uₕ, (0, T));

# ╔═╡ e755a9c4-34a1-493e-899c-33707106ad46
burgers_sol = solve(burgers_prob, abstol = 1e-6, reltol = 1e-4);

# ╔═╡ 0db500cd-1a5f-4f91-9106-ed6daab35e2c
burgers_prob_allbar = ODEProblem(burgers_∂uₕ_allbar∂t, W * uₕ, (0, T));

# ╔═╡ 0cf705b6-f55d-42de-a025-502abd1fe863
burgers_sol_allbar = solve(burgers_prob_allbar, abstol = 1e-6, reltol = 1e-4);

# ╔═╡ 614724cb-2591-488b-9d84-0cd24bd7b591
@bind t₅ Slider(LinRange(0.0, T, 100), default = T)

# ╔═╡ a330084f-d64a-4b28-9cda-115e79d27596
t₅

# ╔═╡ e025b1a9-cf27-4179-9dd1-545459f83e0d
begin
	plot(x, uₕ, label = "Initial")
	plot!(x, W * uₕ, label = "Initial discretized-then-filtered")
	#plot!(x, burgers_sol(t₅), label = "Discretized")
	plot!(x, W * burgers_sol(t₅), label = "W * Discretized")
	plot!(x, burgers_sol_allbar(t₅), label = "Discretized-then-filtered")
	ylims!(minimum(uₕ), maximum(uₕ))
end

# ╔═╡ Cell order:
# ╟─0f52e8af-7413-4449-9b6a-53be3dc0acfc
# ╟─3fb274d2-6911-4cec-b251-85140a249f6c
# ╠═cc1f58dc-b599-4bf4-a1da-bdd62d7b26d4
# ╠═f6d1766a-ee6c-4f88-825c-903325adcbb7
# ╠═828c4651-1bac-4f10-bc9b-582cdb7f677d
# ╠═f245bd45-3519-482b-948c-27b42f8bb5c7
# ╠═5cb7375b-ee81-4243-81f8-c195f1335bda
# ╠═db73cbe7-c658-4973-8da2-1dafe1353eeb
# ╠═eb766d7f-9b10-4d16-8d4e-5dae1ac6d643
# ╠═77a01981-5513-49ba-8bb8-6972b2e22e14
# ╟─fdbc96a2-586f-488a-96f4-5acd923615d0
# ╟─b0026f45-8281-47bd-ad7e-a96ced4d96b2
# ╠═e6a7eba4-01b9-45ca-824f-45f2643b89e0
# ╟─62031239-19c1-4aa4-a67c-f0e6f0093578
# ╠═acab5f55-7013-4932-9637-bd482f3ae49d
# ╠═ac526d65-ab65-4336-818f-8e3eed50f813
# ╟─07717f08-8811-4308-a580-9add99cbca06
# ╟─896d4fe7-f66f-4998-bad1-3a499e354c37
# ╠═5b3006a0-2750-4876-9732-7f5bd1cca904
# ╠═1b0e9fcc-0bbe-493a-93ec-a23f892684a2
# ╠═5a5fbe7d-732b-4eee-ae27-fa98290163af
# ╠═6f907412-7a84-4a9e-b247-75eb7cf08570
# ╠═d9b23fba-0b5a-44bd-b6cd-652deaea0410
# ╠═5a5ed070-d0d3-4bdb-8ef7-fdfc6206637c
# ╠═1b17586c-dbcc-40fe-851d-8a60c4885cf1
# ╟─7759cc59-df9d-4735-a777-38a231503d7f
# ╠═305c5464-0117-4752-a51a-3332e76e9f9b
# ╠═95bdf52c-dba4-4176-8070-6039659db40c
# ╠═79094bcc-233a-452d-8b4f-95f47452d168
# ╠═dbf1ed2f-41eb-4e29-ad91-8843a5c9c01f
# ╟─a78e0d6c-b0a7-419e-96f9-afd611913457
# ╟─d267025d-19b4-4edb-9d28-80ae94d9b58f
# ╠═dc992c13-6ba3-491d-818f-ebc45eb21643
# ╟─43eaa3ee-e66b-4afe-ba8f-b4b5ac75038c
# ╟─20f3a297-4a7b-42ae-aa88-28330200a05b
# ╠═0a677943-7f22-444a-9807-1c3f1b2f4f1b
# ╟─bac7ddbd-753a-4c0a-b219-208f7f2fbd93
# ╠═039dfa5d-1ac9-481f-909f-71c5973d30c5
# ╟─84d04b1d-1ddf-4b1a-86e0-53080e6514c9
# ╟─cbad822c-df13-4f47-a201-1dc122833bd0
# ╠═f74e93ac-66f0-4fce-85c5-a0fbfe670c18
# ╠═3266dc74-c83e-47f1-8005-7c4fa9c24ccd
# ╠═711ffa2f-4459-42c8-98c1-073d6f3c6e53
# ╠═a0db65dd-27cf-4fb7-9a56-2ffb470d78fb
# ╠═0f5893c5-96c4-40d3-8653-bf1ed478cd1d
# ╠═f8df05ca-0e13-4cc1-9571-e9033e4ba301
# ╠═aaa8e1d5-8d72-4dac-a1d6-d5aab59141f2
# ╠═f8b2926e-7bf6-44d9-a13c-6e1c0e53f2ce
# ╠═f6732413-b031-4391-9eb8-fe1f6da932bb
# ╟─c5cfc1dc-16c4-49ec-af03-b570ff7dad2a
# ╠═1267f4f0-bf89-4063-858f-fc545ea626ef
# ╠═15827d5b-a356-42c8-98b7-faac05bebf16
# ╠═0ab43c8e-cfaf-44c4-b8e1-05141473b050
# ╠═62e746e1-3eec-4051-bf27-870b8e1c1448
# ╠═6ae8033a-c4cd-46b5-8df6-539afb6380b3
# ╠═90b92d5d-603a-40ca-b3f1-bc562cb41704
# ╠═a9157042-7f60-4d0b-b35f-d12d51322110
# ╠═5f1650bf-c798-41ac-9a65-0b60da1a858e
# ╠═9183a6a9-bfc7-403e-99b5-72c56a396fb7
# ╠═7b8aab02-0d7f-4f71-9fc8-49dbd9ae2064
# ╠═772cec35-2650-44da-b502-9352ed4f862e
# ╠═363f1fc6-0736-45bf-aabb-3e6757cad0df
# ╠═a55d08ca-3b43-48bd-b749-a89178531a9a
# ╠═661bdd7e-5050-4159-9aea-1c32fe51880b
# ╟─e8871c70-b93e-11eb-3d28-6b2401998411
# ╠═5864dd46-bcc8-4e2b-94a1-e1ba1285fb71
# ╠═5bfd2fc7-1f35-4049-b211-31f888bc4b26
# ╠═2767529e-c479-4075-b15b-27b5791894e8
# ╟─bc1a6782-81ac-4f4f-92f5-d36690223e16
# ╠═81705bc9-b42a-4276-bd1b-ee8e2aa21b1e
# ╠═5fdb8837-fe1c-4508-bb2c-fea65d1a081a
# ╠═0d9c11df-25d2-47da-8435-db053d453d2f
# ╠═5d8918cc-9d3d-4eb9-88a5-a346155a31b1
# ╟─2a057ba9-f6e0-42a2-9254-95bfacbd2e22
# ╟─9cb3eedd-7dd4-4a6b-b4e8-0315eaaee25c
# ╠═60156fd6-468c-409a-b476-fd9c39b8d27a
# ╠═de0c8598-3683-426a-a2a6-ac90c660059f
# ╠═2ec9b609-9fc3-4cad-9e18-46ea8cc80f94
# ╟─ede977d7-6d96-4136-92ec-cbd2941dd54b
# ╠═e2da6398-3532-4688-aedc-4d39f3cc8892
# ╠═a7425523-5cb7-4a63-8fc4-6e3ed8df0b74
# ╟─31d56864-e1ae-42d1-b9db-0ada73f939fd
# ╠═ce8f746f-f0ac-4e27-a899-baa751a52982
# ╠═6eb69dd7-7657-41e0-87eb-23d6818b448e
# ╠═4c65f35f-49c1-439f-ba90-82582ff6728b
# ╟─e19206b7-6eda-4864-b4cd-00d020ed8453
# ╠═4bbb99d4-eeb9-4668-a205-465862747d25
# ╠═e1d6d80f-52d3-4a80-81cb-dd8e6bfbcc12
# ╠═f950a27f-c4ad-4c5c-9396-cf06d2b986f4
# ╠═01716cfe-2a2a-431a-9b8a-fe4f24850929
# ╟─17327732-f704-4a3d-84bf-4c00fc7022c4
# ╠═f9a079f2-c1c4-4e49-9632-082d6a9892c0
# ╠═fa0ab96d-70c8-4480-9deb-7ede4d1dee79
# ╠═bf52219b-7e63-4f3d-a2f4-1e25d9580666
# ╟─2e8afcff-1de4-48df-92c9-f75944f9c125
# ╠═d3f2c2dc-cd06-4f97-8aae-884ff1bbb329
# ╠═9d13a2a8-b2c4-4437-bffe-655f6c87cf41
# ╟─09898683-742d-48f1-b4e1-3d07ea8f52b8
# ╠═f94099b1-b213-4c27-8bd0-77390fd9ad15
# ╠═9cdab2f4-1c1f-4594-a524-cfc85ac86c34
# ╠═1a1e2112-a2b0-4806-b522-c328943229fc
# ╟─3f0a5879-0d40-45c5-b657-d4d5e8f858a3
# ╠═defcf455-75d2-41fc-a1bf-4ce7f73bbe43
# ╠═1f80eb67-bb7d-4921-b933-6fad29ad3b23
# ╟─a3d1ef1a-c991-4d4d-af34-cf4922cb1d92
# ╠═c720697f-3773-4c27-802c-ee596888073b
# ╟─3da2f0ec-fa87-4b69-88b9-fcc251a2b1b8
# ╠═b62dd663-9d31-4563-8ded-ab697e77a0bd
# ╟─d9ef66cb-8c6f-40ec-8abf-c57ba7404381
# ╟─8416c7c2-316a-41b0-a66e-0fc217358d6b
# ╠═e7d65165-bbf5-43f4-893d-11744572b96f
# ╠═186a39d7-bf10-4729-9181-dd2b6ccffbe8
# ╠═5b4186f8-94f6-45ca-ab22-afc25bd6ac58
# ╠═80582fde-553c-4df4-bceb-52aa48ae66eb
# ╠═e755a9c4-34a1-493e-899c-33707106ad46
# ╠═0db500cd-1a5f-4f91-9106-ed6daab35e2c
# ╠═0cf705b6-f55d-42de-a025-502abd1fe863
# ╠═614724cb-2591-488b-9d84-0cd24bd7b591
# ╠═a330084f-d64a-4b28-9cda-115e79d27596
# ╠═e025b1a9-cf27-4179-9dd1-545459f83e0d
