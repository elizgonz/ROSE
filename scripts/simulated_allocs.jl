using JLD2
using SPICE
using Revise
using CSV
using DataFrames
using Statistics
using CUDA
using Dates 
using NaNMath; nm=NaNMath

include("../src/get_kernels.jl")
include("../src/gpu_physics.jl")
include("../src/gpu_precomps_europa.jl")
include("../src/GPUAllocs.jl")

get_kernels()

function save_gpu_allocs(file, gpu_allocs)
    idx_grid = Array(gpu_allocs.ld[:, :, 1]) .> 0.0
    jldopen(file, "w") do f
        f["ld"] = Array(view(gpu_allocs.ld, idx_grid))
        f["dA"] = Array(view(gpu_allocs.dA, idx_grid))
        f["phase"] = Array(view(gpu_allocs.phase, idx_grid))
        f["projected_v"] = Array(view(gpu_allocs.projected_v, idx_grid))
    end
end

wavelengths = [4012.338861810625, 4065.4845352620687, 4092.5889458867427, 4120.056656758336, 4147.896340182007, 4176.114670163549, 4204.719253588902, 4233.718408904378, 4263.118683657377, 4292.933326587802, 4323.166218217185, 4353.82827059154, 4384.928162394974, 4480.951359801634, 4513.900515086146, 4547.337859551167, 4581.274233569078, 4650.689434914498, 4686.19211540847, 4722.240524196751, 4758.848222153147, 4796.027842766245, 4833.7925364793955, 4911.135240469596, 4950.741881972264, 4990.992794881698, 5158.761280269197, 5202.480473364398, 5246.947034189537, 5292.180241532383, 5338.200128507048, 5385.027399821762, 5530.57141372899, 5580.850191469802, 5632.051637984013, 6078.161902948444, 6138.944361995399, 6200.954822586068, 7486.53469912381, 7770.835960089047]
orders = [20,22,23,24,25,26,27,28,29,30,31,32,33,36,37,38,39,41,42,43,44,45,46,48,49,50,54,55,56,57,58,59,62,63,64,72,73,74,91,94]

obs_lat = 31.9583 
obs_long = -111.5967  
alt = 2.090

function projected_RV_gpu(time_stamps)

    # set up paramaters for disk
    N = 197
    Nt = length(time_stamps)
    Nsubgrid=40

    # get latitude grid edges and centers
    ϕe = range(deg2rad(-90.0), deg2rad(90.0), length=N+1)
    ϕc = get_grid_centers(ϕe)
    # number of longitudes in each latitude slice
    Nθ = get_Nθ.(ϕc, step(ϕe)) 

    RV_list_no_cb_all = Vector{Vector{Float64}}(undef,length(wavelengths)...)
    # loop over lines
    for i in 1:length(wavelengths)
        println(i)

        RV_list_no_cb = Vector{Float64}(undef,length(time_stamps)...)
        mkpath("../data/simulated_allocs/order_$(orders[i])")
        for t in 1:Nt
            gpu_allocs = GPUAllocs(N, Nt, Nθ, precision=Float64)

            calc_europa_quantities_gpu!(time_stamps[t], obs_long, obs_lat, alt, wavelengths[i], N, Nθ, Nsubgrid, gpu_allocs) 
            save_gpu_allocs("../data/simulated_allocs/order_$(orders[i])/epoch_$t.jld2", gpu_allocs)

        end
    end

end

line_data = CSV.read("../data/NEID_rv_unbin.csv", DataFrame)
# convert from utc to et as needed by SPICE
dts = string.(DateTime.(
    getfield.(match.(r"\d{8}T\d{6}", line_data[!, "filename"][24:121]), :match),
    dateformat"yyyymmddTHHMMSS"
))
time_stamps = utc2et.(dts)

projected_RV_gpu(time_stamps)