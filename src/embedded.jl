"""
Create embedded loss function on dataset.
"""
function create_loss_fit(
    u,
    t;
    n_sample = size(u, 2),
    n_time = length(t) - 1,
    kwargs...,
)
    u₀ = u[:, :, 1]
    function loss(A)
        # randperm(n)
        iu = Zygote.@ignore sort(shuffle(1:size(u, 2))[1:n_sample])
        it = Zygote.@ignore sort(shuffle(2:length(t))[1:n_time])
        it = [1; it]
        uₜ = u[:, iu, it]
        sum(abs2, S(A, u₀[:, iu], t[it]; kwargs...) - uₜ) / prod(size(uₜ))
    end
    loss
end

create_loss_prior(A_ref) = A -> sum(abs2, A - A_ref) / prod(size(A_ref))

create_initial_state(A_ref) = (;
    A = Matrix(A_ref),
    A_min = Matrix(A_ref),
    m = zeros(size(A_ref)),
    v = zeros(size(A_ref)),
    hist_i = zeros(Int, 0),
    hist_r = zeros(0),
)

create_loss_mixed(losses, weights) = A -> sum(λ * loss(A) for (loss, λ) ∈ zip(losses, weights))

"""
    fit_embedded(
        state,
        loss;
        α = 0.001,
        β₁ = 0.9,
        β₂ = 0.999,
        ϵ = 1e-8,
        n_iter = 100,
        testloss,
        ntestloss = 10,
        doplot = true,
        kwargs...,
    )

Fit operator to data while embedded in ODE solver using the ADAM optimizer.
"""
function fit_embedded(
    state,
    loss;
    α = 0.001,
    β₁ = 0.9,
    β₂ = 0.999,
    ϵ = 1e-8,
    n_iter = 100,
    testloss,
    ntestloss = 10,
    doplot = true,
    kwargs...,
)
    doplot && gr()

    (; A, A_min, m, v, hist_i, hist_r) = state

    m̂ = copy(m)
    v̂ = copy(v)

    r = testloss(A)
    r_min = testloss(A_min)
    r_min < Inf || (r_min = Inf)
    isempty(hist_i) && push!(hist_i, 0)
    isempty(hist_r) && push!(hist_r, r)
    starttime = time()
    n_last = hist_i[end]
    for i = 1:n_iter
        itertime = time()
        print("Iteration $i \t")
        g = first(Zygote.gradient(loss, A))
        @. m = β₁ * m + (1 - β₁) * g
        @. v = β₂ * v + (1 - β₂) * g^2
        @. m̂ = m / (1 - β₁^i)
        @. v̂ = v / (1 - β₂^i)
        @. A = A - α * m̂ / (√v̂ + ϵ)
        if i % ntestloss == 0
            r = testloss(A)
            if r < r_min
                r_min = r
                A_min .= A
            end
            push!(hist_i, n_last+i)
            push!(hist_r, r)
            doplot && display(
                plot(
                    hist_i,
                    hist_r;
                    legend = false,
                    xlabel = "Iterations",
                    title = "Validation error",
                ),
            )
        end
        # println("batch: $(loss(A)) \t test: $(r) (-$(i % ntestloss))")
        println(
            join(
                [
                    @sprintf("valid: %.4g", r),
                    @sprintf("itertime: %.3f", time() - itertime),
                    @sprintf("totaltime: %.3f", time() - starttime),
                ],
                "\t",
            ),
        )
    end
    state
end
