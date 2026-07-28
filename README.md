# ROSE - Radial velocity Opposition Surge Emulator 

ROSE is designed to produce a time series of radial velocities with the opposition surge of Europa as the Earth transits the Sun as seen from Europa. 

ROSE is described in Gonzalez et al. 2026b, which develops the first radial velocity model that is informed by the opposition surge effect in order to model the observed Europa radial velocities during the Earth transit. The results of this paper can be reproduced using this repo. 

## Installation

ROSE is written entirely in Julia and requires Julia v1.12 or greater. Installation instructions for Julia are available from [julialang.org](https://julialang.org/downloads/). ROSE requires the CUDA Toolkit (You may also need to install the proper version NVIDIA driver, for more information on installation, please see: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 

```bash
git clone git@github.com:elizgonz/ROSE.git
cd ROSE
julia
```

## Example

```julia
using SPICE
using CUDA
using CSV
using DataFrames

include("src/get_kernels.jl")
include("src/gpu_physics.jl")
include("src/gpu_precomps_europa.jl")
include("src/GPUAllocs.jl")

get_kernels()

# Observer Coordinates (NEID)
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

        calc_europa_quantities_gpu!(time_stamps[t], obs_long, obs_lat, alt, 5530.57141372899, OP_mech, N, Nθ, 
                                    Nsubgrid, gpu_allocs, B_OPE, h_OPE)
            
        idx_grid = Array(gpu_allocs.ld[:, :, 1]) .> 0.0

        brightness = Array(gpu_allocs.ld[:, :, 1]) .* Array(gpu_allocs.dA)

        cheapflux = sum(view(brightness, idx_grid))

        # determine final mean weighted velocity for disk grid
        final_weight_v_no_cb = sum(view(Array(gpu_allocs.projected_v) .* brightness, idx_grid)) / cheapflux 

        RV_list_no_cb[t] = deepcopy(final_weight_v_no_cb)
    end
end

time_stamps = range(utc2et("2026-01-05T00:00:00.0"), utc2et.("2026-01-27T00:00:00.0"), step = 1800.0)

projected_RV_gpu(time_stamps, 1)
```
![RVcurve](./figures/OSfigure.pdf)

## Included Information

The Gonzalez et al. 2026b results from hpfserval and neidserval, along with the NEID order-by-order results, are included under data. The script to generate the figures from the manuscript is found under figures/RVs, which includes the simulated RVs when using various opposition surge parameters. The example above to generate the radial velocities is expanded in scripts/europa_RV.jl for the opposition surge parameter fits from MCMC runs. The MCMC scripts are included under scripts. 

