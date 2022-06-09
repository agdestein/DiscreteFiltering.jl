# Loss function for training
loss_reg(A, A_ref) = sum(abs2, A - A_ref) / prod(size(A_ref))

"""
Create loss function on dataset.
"""
function create_loss(uₜ, t, A_ref; λ = 1e-10, loss_reg = loss_reg, kwargs...)
    u₀ = uₜ[:, :, 1]
    loss(A) =
        sum(abs2, S(A, u₀, t; kwargs...) - uₜ) / prod(size(uₜ)) + λ * loss_reg(A, A_ref)
    loss
end

# function fit_intrusive(
#     A_ref, uₜ, t;
#     opt = ADAM(0.01),
#     λ = 1e-10,
#     nbatch = 5,
#     niter = 20,
#     nepoch = 1,
#     initial = A_ref,
# )
#     A = Matrix(A_ref)
#     nsample = size(uₜ, 2)
#     batches = [1 + (nsample ÷ nbatch)*(i-1):(nsample ÷ nbatch)*i for i in 1:nbatch]
#     for epoch = 1:nepoch
#         # for batch = batches
#         for i = 1:5
#             batch = rand(batches)
#             loss = create_loss(uₜ[:, batch, :], t, A_ref; λ)
#             result_ode = DiffEqFlux.sciml_train(loss, A, opt; cb = (A, l) -> (println(l); false), maxiters = niter)
#             A = result_ode.u
#         end
#     end
#     A
# end

"""
    fit_intrusive(
        A_ref,
        uₜ,
        t;
        α = 0.001,
        β₁ = 0.9,
        β₂ = 0.999,
        ϵ = 1e-8,
        λ = 1e-10,
        nbatch = 10,
        niter = 100,
        initial = nothing,
        testloss = A -> nothing,
        ntestloss = 10,
        ntime = 10,
        ntimebatch = 10,
        doplot = true,
        kwargs...,
    )

Fit operator to data intrusively (trough the ODE solver) using the ADAM optimizer.
"""
function fit_intrusive(
    A_ref,
    uₜ,
    t;
    α = 0.001,
    β₁ = 0.9,
    β₂ = 0.999,
    ϵ = 1e-8,
    λ = 1e-10,
    nbatch = 10,
    niter = 100,
    initial = nothing,
    testloss = A -> nothing,
    ntestloss = 10,
    ntime = 10,
    ntimebatch = 10,
    doplot = true,
    kwargs...,
)
    gr()

    if isnothing(initial)
        state = (;
            A = Matrix(A_ref),
            A_min = Matrix(A_ref),
            m = zeros(size(A_ref)),
            v = zeros(size(A_ref)),
            hist = zeros(0),
        )
    else
        state = initial
    end

    (; A, A_min, m, v, hist) = state

    m̂ = copy(m)
    v̂ = copy(v)

    r = testloss(A)
    r_min = testloss(A_min)
    r_min < Inf || (r_min = Inf)
    nsample = size(uₜ, 2)
    samples = [(1+(nsample÷nbatch)*(i-1)):((nsample÷nbatch)*i) for i = 1:nbatch]
    # times = [rand(1:length(t), ntime) for _ ∈ ntimebatch]
    times = [sort(shuffle(1:length(t))[1:ntime]) for _ ∈ ntimebatch]
    losses =
        [create_loss(uₜ[:, s, iₜ], t[iₜ], A_ref; λ, kwargs...) for s ∈ samples, iₜ ∈ times]
    isempty(hist) && push!(hist, r)
    starttime = time()
    for i = 1:niter
        itertime = time()
        print("Iteration $i \t")
        loss = rand(losses)
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
            push!(hist, r)
            doplot && display(
                plot(
                    hist;
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
                    "test: $r",
                    @sprintf("itertime: %.3f", time() - itertime),
                    @sprintf("totaltime: %.3f", time() - starttime),
                ],
                "\t",
            ),
        )
    end
    state
end
