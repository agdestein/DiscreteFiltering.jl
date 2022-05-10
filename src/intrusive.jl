# Loss function for training
loss_data_mse(A, u₀, uₜ, t) = sum(abs2, S(A, u₀, t) - uₜ) / sum(abs2, uₜ)
loss_reg(A, A_ref) = sum(abs2, A - A_ref) / sum(abs2, A_ref)

"""Create loss function on dataset."""
function create_loss(
    u₀, uₜ, t, A_ref;
    λ = 1e-10,
    loss_data = loss_data_mse,
    loss_reg = loss_reg,
)
    loss(A) = loss_data(A, u₀, uₜ, t) + λ * loss_reg(A, A_ref)
    loss
end

# function fit_intrusive(
#     A_ref, u₀, uₜ, t;
#     opt = ADAM(0.01),
#     λ = 1e-10,
#     nbatch = 5,
#     niter = 20,
#     nepoch = 1,
#     initial = A_ref,
# )
#     A = Matrix(A_ref)
#     nsample = size(u₀, 2)
#     batches = [1 + (nsample ÷ nbatch)*(i-1):(nsample ÷ nbatch)*i for i in 1:nbatch]
#     for epoch = 1:nepoch
#         # for batch = batches
#         for i = 1:5
#             batch = rand(batches)
#             loss = create_loss(u₀[:, batch], uₜ[:, batch, :], t, A_ref; λ)
#             result_ode = DiffEqFlux.sciml_train(loss, A, opt; cb = (A, l) -> (println(l); false), maxiters = niter)
#             A = result_ode.u
#         end
#     end
#     A
# end

"""Fit operator to data intrusively (trough the ODE solver) using the ADAM optimizer."""
function fit_intrusive(
    A_ref, u₀, uₜ, t;
    α = 0.001, β₁ = 0.9, β₂ = 0.999, ϵ = 1e-8, λ = 1e-10,
    nbatch = 10,
    niter = 100,
    nepoch = 1,
    initial = A_ref,
    testloss = A -> nothing,
    ntestloss = 10,
)
    testresult = "nothing"
    A = Matrix(initial)
    m = zeros(size(A))
    m̂ = zeros(size(A))
    v = zeros(size(A))
    v̂ = zeros(size(A))
    nsample = size(u₀, 2)
    batches = [1+(nsample ÷ nbatch)*(i-1):(nsample ÷ nbatch)*i for i = 1:nbatch]
    losses = [create_loss(u₀[:, batch], uₜ[:, batch, :], t, A_ref; λ) for batch ∈ batches]
    for epoch = 1:nepoch
        # for batch ∈ batches
        for i = 1:niter
            print("Iteration $i \t")
            loss = rand(losses)
            g = first(Zygote.gradient(loss, A))
            @. m = β₁ * m + (1 - β₁) * g
            @. v = β₂ * v + (1 - β₂) * g ^ 2
            @. m̂ = m / (1 - β₁ ^ i)
            @. v̂ = v / (1 - β₂ ^ i)
            @. A = A - α * m̂ / (√v̂ + ϵ)
            if i % ntestloss == 0
                testresult = "$(testloss(A))"
            end
            println("batch: $(loss(A)) \t test: $(testresult) (-$(i % ntestloss))")
        end
    end
    A
end
