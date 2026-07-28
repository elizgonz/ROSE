using CSV
using DataFrames
using JLD2
using Statistics
using Turing
using DynamicPPL
using Distributions
using LinearAlgebra
using Random
using Glob
using FITSIO
using CUDA
using Serialization
using MCMCChains

CUDA.allowscalar(false)

orders = [20,22,23,24,25,26,27,28,29,30,31,32,33,36,37,38,39,41,42,43,44,45,46,48,49,50,54,55,56,57,58,59,62,63,64,72,73,74,91,94]
data_path = "../data/simulated_allocs"

# -----------------------------
# PRELOAD ALL DATA
# -----------------------------
println("Preloading data into memory...")

function preload_data(orders, data_path)

    all_data = Dict()

    for (i, ord) in enumerate(orders)

        order_data = []

        epoch = 1

        while true

            file = joinpath(
                data_path,
                "order_$(ord)",
                "epoch_$(epoch).jld2"
            )

            if !isfile(file)
                break
            end

            ld          = load(file, "ld")
            dA          = load(file, "dA")
            phase       = load(file, "phase")
            projected_v = load(file, "projected_v")

            tan_half_phase = CuArray(tan.(mod.(phase ./ 2, pi)))

            push!(order_data, (
                ld              = CuArray(ld),
                dA              = CuArray(dA),
                tan_half_phase  = tan_half_phase,
                projected_v     = CuArray(projected_v),
            ))

            epoch += 1
        end

        all_data[i] = order_data

        println("Loaded order $ord with $(length(order_data)) epochs")
    end

    return all_data
end

all_data = preload_data(orders, data_path)

# -----------------------------
# LOAD OBSERVATIONS
# -----------------------------
line_data = CSV.read(
    "../data/NEID_order_rv.csv",
    DataFrame
)

line_data_err = CSV.read(
    "../data/NEID_order_rv_err.csv",
    DataFrame
)

function build_dataset(orders, line_data, line_data_err)

    dataset = []

    for ord in orders

        rv  = line_data[!, "order_$(ord)"][24:121]
        err = line_data_err[!, "order_$(ord)"][24:121]

        push!(dataset, (
            rv  = rv,
            err = err
        ))
    end

    return dataset
end

dataset = build_dataset(orders, line_data, line_data_err)

# -----------------------------
# FORWARD MODEL
# -----------------------------
function gpu_epoch_rv(
    d,
    B_SH,
    inv_h_SH
)

    tanh = d.tan_half_phase

    # ---- SH contribution
    denom_sh = 1 .+ inv_h_SH .* tanh
    sh = B_SH ./ denom_sh


    # ---- weights
    b = d.ld .* d.dA .* (1 .+ sh) 

    num = sum(d.projected_v .* b)
    den = sum(b)

    rv = num / den

    if !isfinite(rv)
        println("Warning: RV not finite")
        return NaN
    end

    return rv
end

function model_all_orders(
    B_SH,
    h_SH,
    all_data,
    orders
)

    T = typeof(B_SH)
    results = Vector{Vector{T}}(undef, length(orders))

    inv_h_SH = 1.0 / h_SH

    for (i, ord) in enumerate(orders)

        order_data = all_data[i]

        rv_model_arr = Vector{T}(undef, length(order_data))

        for t in eachindex(order_data)

            d = order_data[t]

            rv_model_arr[t] = gpu_epoch_rv(
                d,
                B_SH,
                inv_h_SH
            )
        end

        results[i] = rv_model_arr
    end

    return results
end

# -----------------------------
# TURING MODEL
# -----------------------------
@model function rv_model(
    obs_mat,
    err_mat,
    all_data,
    orders
)

    # ---- shared parameters
    B_SH ~ Uniform(0, 1)

    log_h_SH ~ Normal(log(1e-2), 1)

    h_SH = exp(log_h_SH)

    model_rv = model_all_orders(
        B_SH,
        h_SH,
        all_data,
        orders
    )

    for i in eachindex(obs_mat)

        if any(isnan, model_rv[i])

            println("Warning: RV not finite")

            Turing.@addlogprob!(-Inf)

            return
        end

        for t in eachindex(obs_mat[i])

            obs_mat[i][t] ~ Normal(
                model_rv[i][t],
                err_mat[i][t]
            )
        end
    end
end

# -----------------------------
# RUN MCMC
# -----------------------------
Random.seed!()

obs_mat = [dataset[i].rv for i in eachindex(dataset)]
err_mat = [dataset[i].err for i in eachindex(dataset)]

model = rv_model(
    obs_mat,
    err_mat,
    all_data,
    orders
)

total_samples = 5000
chunk_size    = 100

n_chunks = total_samples ÷ chunk_size

all_chains = Chains[]

# -----------------------------
# INITIAL PARAMETERS
# -----------------------------

# After defining model, create the LogDensityFunction once before the loop
ldf = DynamicPPL.LogDensityFunction(model)

current_init = DynamicPPL.InitFromVector([0.5, -8.0], ldf)

# -----------------------------
# SAMPLING LOOP
# -----------------------------
for i in 1:n_chunks

    global current_init

    println("Running chunk $i / $n_chunks")

    chain = Chains(sample(
        model,
        NUTS(),
        chunk_size,
        progress = true,
        initial_params = current_init
    ))

    push!(all_chains, chain)

    combined_chain = reduce(chainscat, all_chains)

    serialize(
        "chain_checkpoint_SH.jls",
        combined_chain
    )

    println("Saved checkpoint after chunk $i")

    # initialize next chunk from last sample
    param_vals   = vec(Array(chain[end, :, 1]))
    current_init = DynamicPPL.InitFromVector(param_vals, ldf)
end

final_chain = reduce(chainscat, all_chains)

println(describe(final_chain))

df = DataFrame(final_chain)

CSV.write(
    "mcmc_chain_final_SH.csv",
    df
)