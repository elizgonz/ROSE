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

orders = [20,22,23,24,25,26,27,28,29,30,31,32,33,36,37,38,39,41,42,43,44,45,46,48,49,50,54,55,56,57,58,59,62,63,64,72,73,74,91,94]
wavelengths = [4012.338861810625, 4065.4845352620687, 4092.5889458867427, 4120.056656758336, 4147.896340182007, 4176.114670163549, 4204.719253588902, 4233.718408904378, 4263.118683657377, 4292.933326587802, 4323.166218217185, 4353.82827059154, 4384.928162394974, 4480.951359801634, 4513.900515086146, 4547.337859551167, 4581.274233569078, 4650.689434914498, 4686.19211540847, 4722.240524196751, 4758.848222153147, 4796.027842766245, 4833.7925364793955, 4911.135240469596, 4950.741881972264, 4990.992794881698, 5158.761280269197, 5202.480473364398, 5246.947034189537, 5292.180241532383, 5338.200128507048, 5385.027399821762, 5530.57141372899, 5580.850191469802, 5632.051637984013, 6078.161902948444, 6138.944361995399, 6200.954822586068, 7486.53469912381, 7770.835960089047]
wave_ref = median(wavelengths) 
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
    B_CB,
    h_cb_eff
)

    tanh = d.tan_half_phase

    # ---- CB contribution
    x = tanh ./ h_cb_eff

    term = (1 .- exp.(-x)) ./ x
    denom_cb = (1 .+ x).^2

    cb = (B_CB / 2) .* ((1 .+ term) ./ denom_cb)

    # ---- weights
    b = d.ld .* d.dA .* (1 .+ cb) 

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
    B_CB,
    h_CB_orders,
    dhdw,
    all_data,
    orders
)

    T = typeof(B_CB)
    results = Vector{Vector{T}}(undef, length(orders))

    for (i, ord) in enumerate(orders)

        order_data = all_data[i]

        h_cb_eff = h_CB_orders + dhdw * (wavelengths[i] - wave_ref)

        rv_model_arr = Vector{T}(undef, length(order_data))

        for t in eachindex(order_data)

            d = order_data[t]

            rv_model_arr[t] = gpu_epoch_rv(
                d,
                B_CB,
                h_cb_eff
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
    B_CB ~ Uniform(0, 1)

    log_h_CB ~ Normal(log(1e-2), 1)

    h_CB = exp(log_h_CB)

    dhdw ~ Normal(0, 1.0)

    model_rv = model_all_orders(
        B_CB,
        h_CB,
        dhdw,
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

# Fresh start
current_init = DynamicPPL.InitFromVector([0.5, 0.2, 0.1], ldf)

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
        "chain_checkpoint_CB_model.jls",
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
    "mcmc_chain_final_CB_model.csv",
    df
)