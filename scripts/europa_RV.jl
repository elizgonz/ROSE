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

wavelengths = [4012.338861810625, 4065.4845352620687, 4092.5889458867427, 4120.056656758336, 4147.896340182007, 4176.114670163549, 4204.719253588902, 4233.718408904378, 4263.118683657377, 4292.933326587802, 4323.166218217185, 4353.82827059154, 4384.928162394974, 4480.951359801634, 4513.900515086146, 4547.337859551167, 4581.274233569078, 4650.689434914498, 4686.19211540847, 4722.240524196751, 4758.848222153147, 4796.027842766245, 4833.7925364793955, 4911.135240469596, 4950.741881972264, 4990.992794881698, 5158.761280269197, 5202.480473364398, 5246.947034189537, 5292.180241532383, 5338.200128507048, 5385.027399821762, 5530.57141372899, 5580.850191469802, 5632.051637984013, 6078.161902948444, 6138.944361995399, 6200.954822586068, 7486.53469912381, 7770.835960089047]

obs_lat = 31.9583 
obs_long = -111.5967  
alt = 2.090

function projected_RV_gpu(time_stamps, OP_mech)

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
        if OP_mech == 0
            B_OPE = 0.358
            h_OPE = exp(-6.7)
        elseif OP_mech == 1
            B_OPE = 0.456
            h_OPE = exp(-8.21)
        end

        RV_list_no_cb = Vector{Float64}(undef,length(time_stamps)...)
        for t in 1:Nt
            gpu_allocs = GPUAllocs(N, Nt, Nθ, precision=Float64)

            calc_europa_quantities_gpu!(time_stamps[t], obs_long, obs_lat, alt, wavelengths[i], OP_mech, N, Nθ, Nsubgrid, gpu_allocs, B_OPE, h_OPE)
            
            idx_grid = Array(gpu_allocs.ld[:, :, 1]) .> 0.0

            brightness = Array(gpu_allocs.ld[:, :, 1]) .* Array(gpu_allocs.dA)

            cheapflux = sum(view(brightness, idx_grid))

            # determine final mean weighted velocity for disk grid
            final_weight_v_no_cb = sum(view(Array(gpu_allocs.projected_v) .* brightness, idx_grid)) / cheapflux 

            RV_list_no_cb[t] = deepcopy(final_weight_v_no_cb)
        end
    RV_list_no_cb_all[i] = RV_list_no_cb
    end

    if OP_mech == 0
        @save "projected_gpu_CB_fit.jld2"
        jldopen("projected_gpu_CB_fit.jld2", "a+") do file
            file["RV_list_no_cb"] = deepcopy(RV_list_no_cb_all) 
            file["time"] = deepcopy(SPICE.et2utc.(time_stamps, "ISOC", 3))
        end
    elseif OP_mech == 1
        @save "projected_gpu_SH_fit.jld2"
        jldopen("projected_gpu_SH_fit.jld2", "a+") do file
            file["RV_list_no_cb"] = deepcopy(RV_list_no_cb_all) 
            file["time"] = deepcopy(SPICE.et2utc.(time_stamps, "ISOC", 3))
        end
    end
end

time_stamps = range(utc2et("2026-01-05T00:00:00.0"), utc2et.("2026-01-27T00:00:00.0"), step = 1800.0)

projected_RV_gpu(time_stamps, 1)