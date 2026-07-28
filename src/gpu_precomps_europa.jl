import Base: AbstractArray as AA
import Base: AbstractFloat as AF
using LinearAlgebra

include(joinpath(@__DIR__, "gpu_physics.jl"))
include(joinpath(@__DIR__, "get_kernels.jl"))

quad_ld_coeff_SSD_Europa = CSV.read(joinpath(@__DIR__, "..", "data", "LD_coeff_SSD_Europa.csv"), DataFrame)

#set required body paramters as global variables (units:km)
earth_radius = bodvrd("EARTH", "RADII")[1]	
earth_radius_pole = bodvrd("EARTH", "RADII")[3]	
sun_radius = bodvrd("SUN","RADII")[1]
moon_radius = bodvrd("MOON", "RADII")[1] 

function calc_europa_quantities_gpu!(epoch::T1, obs_long::T1, obs_lat::T1, alt::T1, wavelength, N, Nθ, Nsubgrid, 
                                        gpu_allocs) where {T1<:AF}

    phase = gpu_allocs.phase
    ld = gpu_allocs.ld
    projected_v = gpu_allocs.projected_v
    dA = gpu_allocs.dA

    # re-zero everything
    CUDA.@sync begin
        phase .= 0.0
        ld .= 0.0
        projected_v .= 0.0
        dA .= 0.0
    end

    # convert scalars from disk params to desired precision
    A = convert(T1, 14.713)
    B = convert(T1, -2.396)
    C = convert(T1, -1.787)

    filtered_df = quad_ld_coeff_SSD_Europa[quad_ld_coeff_SSD_Europa.wavelength .== wavelength, :]
    u1 = convert(T1, filtered_df.law_4u1[1])
    u2 = convert(T1, filtered_df.law_4u2[1])
    u3 = convert(T1, filtered_df.law_4u3[1])
    u4 = convert(T1, filtered_df.law_4u4[1])

    # geometry from disk
    Nϕ = N
    Nθ_max = maximum(Nθ)

    # query SPICE for E, S, M position (km) and velocities (km/s)
    BE_bary = spkssb(399,epoch,"J2000")

    # determine xyz earth coordinates for lat/long of observatory
    flat_coeff = (earth_radius - earth_radius_pole) / earth_radius
    EO_earth_pos = georec(deg2rad(obs_long), deg2rad(obs_lat), alt, earth_radius, flat_coeff)
    # set earth velocity vectors
    EO_earth = vcat(EO_earth_pos, [0.0, 0.0, 0.0])
    # transform into ICRF frame
    EO_bary = sxform("IAU_EARTH", "J2000", epoch) * EO_earth
    CUDA.@sync EO_bary_gpu = CuArray(EO_bary)

    # get vector from barycenter to observatory on Earth's surface
    BO_bary = BE_bary .+ EO_bary

    # set string for ltt and abberation
    lt_flag = "CN+S"

    #Europa position vectors:
    EuE_bary = spkezp(399, epoch, "J2000", lt_flag, 502)[1]
    EuS_bary = spkezp(10, epoch, "J2000", lt_flag, 502)[1]
    phase_angle = acos(calc_mu(EuE_bary, EuS_bary))

    CUDA.@sync EuO_bary_gpu = CuArray(EuE_bary[1:3] + EO_bary[1:3])
    CUDA.@sync EuS_bary_gpu = CuArray(EuS_bary)
    CUDA.@sync moon_radius_gpu = CuArray([moon_radius])
    CUDA.@sync earth_radius_gpu = CuArray([earth_radius])
    CUDA.@sync sun_radius_gpu = CuArray([sun_radius])

    # get light travel time corrected OS vector
    OS_bary, OS_lt, OS_dlt = spkltc(10, epoch, "J2000", lt_flag, BO_bary)
    CUDA.@sync OS_bary_gpu = CuArray(OS_bary)

    # get vector from observatory on earth's surface to moon center
    OM_bary, OM_lt, OM_dlt = spkltc(301, epoch, "J2000", lt_flag, BO_bary)
    CUDA.@sync OM_bary_gpu = CuArray(OM_bary)

    # get modified epch
    epoch_lt = epoch - OS_lt

    # get rotation matrix for sun
    sun_rot_mat = pxform("IAU_SUN", "J2000", epoch_lt)
    CUDA.@sync sun_rot_mat_gpu = CuArray(sun_rot_mat)

    CUDA.@sync begin
        Nθ = CuArray{Float64}(Nθ)
    end

    # compute geometric parameters, average over subtiles
    threads1 = 256
    blocks1 = cld(prod(CUDA.size(phase)), prod(threads1))
    CUDA.@sync @cuda threads=threads1 blocks=blocks1 calc_europa_quantities_gpu!(phase, Nϕ, Nθ, Nsubgrid, Nθ_max, ld, projected_v,
                                                                            dA, moon_radius_gpu, OS_bary_gpu, OM_bary_gpu, EO_bary_gpu,
                                                                            sun_rot_mat_gpu, sun_radius_gpu, A, B, C, u1, u2, u3, u4, 
                                                                            earth_radius_gpu, EuO_bary_gpu, EuS_bary_gpu)
    return
end                               

function calc_europa_quantities_gpu!(phase, Nϕ, Nθ, Nsubgrid, Nθ_max, ld, projected_v, 
                                      dA, moon_radius, OS_bary, OM_bary, EO_bary,
                                      sun_rot_mat, sun_radius, A, B, C, u1, u2, u3, u4, 
                                      earth_radius, EuO_bary_gpu, EuS_bary_gpu) 
    sun_radius = sun_radius[1] 
    # get indices from GPU blocks + threads
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # get number of elements along tile side
    k = Nsubgrid

    # total number of elements output array
    num_tiles = Nϕ * Nθ_max

    # get latitude subtile step size
    N_ϕ_edges = Nϕ * Nsubgrid
    dϕ = π / (N_ϕ_edges)

    for t in idx:sdx:num_tiles
        # get index for output array 
        row = (t - 1) ÷ Nθ_max
        col = (t - 1) % Nθ_max
        m = row + 1
        n = col + 1

        # get indices for input array
        i = row * k + 1
        j = col * k + 1

        # get number of longitude tiles in course latitude slice
        N_θ_edges = Nθ[m] * Nsubgrid

        # set up sum holders for scalars
        phase_sum = CUDA.zero(CUDA.eltype(phase))
        dA_sum = CUDA.zero(CUDA.eltype(phase))
        ld_sum = CUDA.zero(CUDA.eltype(phase))
        projected_v_sum = CUDA.zero(CUDA.eltype(phase))

        # initiate counter
        μ_count = 0

        # loop over latitude sub tiles
        for ti in i:i+k-1
            # get coordinates of latitude subtile center
            ϕc_sub = -π/2 + (dϕ/2.0) + (ti - 1) * dϕ
    
            # loop over longitude subtiles
            for tj in j:j+k-1
                # move on if we've looped past last longitude
                if tj > N_θ_edges
                    continue
                end
    
                # get longitude subtile step size
                dθ = 2π / (N_θ_edges)
    
                # get longitude
                θc_sub = (dθ/2.0) + (tj - 1) * dθ  

                # get cartesian coords of patch
                x, y, z = sphere_to_cart_gpu(sun_radius, ϕc_sub, θc_sub)

                # get vector from spherical circle center to surface patch
                a = x
                b = y
                c = CUDA.zero(CUDA.eltype(phase))

                # take cross product to get vector in direction of rotation
                d = - sun_radius * b
                e = sun_radius * a
                f = CUDA.zero(CUDA.eltype(phase))

                # make it a unit vector
                def_norm = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
                d /= def_norm
                e /= def_norm
                f /= def_norm

                # set magnitude by differential rotation
                rp = 2π * sun_radius * CUDA.cos(ϕc_sub) / rotation_period_gpu(ϕc_sub, A, B, C)

                # get in units of c
                rp /= 86400.0

                # set magnitude of vector
                d *= rp
                e *= rp
                f *= rp

                # xyz rotated for SP bary
                x_new = sun_rot_mat[1] * x + sun_rot_mat[4] * y + sun_rot_mat[7] * z
                y_new = sun_rot_mat[2] * x + sun_rot_mat[5] * y + sun_rot_mat[8] * z
                z_new = sun_rot_mat[3] * x + sun_rot_mat[6] * y + sun_rot_mat[9] * z

                # vel component rotated for SP bary
                vx = sun_rot_mat[1] * d + sun_rot_mat[4] * e + sun_rot_mat[7] * f
                vy = sun_rot_mat[2] * d + sun_rot_mat[5] * e + sun_rot_mat[8] * f
                vz = sun_rot_mat[3] * d + sun_rot_mat[6] * e + sun_rot_mat[9] * f

                # OP_bary state vector
                OP_bary_x = OS_bary[1] + x_new
                OP_bary_y = OS_bary[2] + y_new
                OP_bary_z = OS_bary[3] + z_new

                # Europa to position state vector
                EuP_bary_x = EuS_bary_gpu[1] + x_new
                EuP_bary_y = EuS_bary_gpu[2] + y_new
                EuP_bary_z = EuS_bary_gpu[3] + z_new

                # calculate mu
                μ_sub = calc_mu_gpu(x_new, y_new, z_new, OP_bary_x, OP_bary_y, OP_bary_z) 
                if μ_sub <= 0.0
                    continue
                end
                μ_count += 1

                # get OP_bary and SP_bary between them and find projected_velocities_no_cb
                n1 = CUDA.sqrt(OP_bary_x^2.0 + OP_bary_y^2.0 + OP_bary_z^2.0)
                n2 = CUDA.sqrt(vx^2.0 + vy^2.0 + vz^2.0)
                angle = (OP_bary_x * vx + OP_bary_y * vy + OP_bary_z * vz) / (n1 * n2)
                v_rot_sub = (n2 * angle)
                v_rot_sub *= 1000.0
                projected_v_sum += v_rot_sub

                # get projected area element
                dA_sub = calc_dA_gpu(sun_radius, ϕc_sub, dϕ, dθ)
                dA_sub *= μ_sub
                dA_sum += dA_sub

                # calculate distance (Europa)
                n1 = CUDA.sqrt(EuP_bary_x^2.0 + EuP_bary_y^2.0 + EuP_bary_z^2.0)  
                n2 = CUDA.sqrt(EuO_bary_gpu[1]^2.0 + EuO_bary_gpu[2]^2.0 + EuO_bary_gpu[3]^2.0)  
                phase_sub = acos((EuO_bary_gpu[1] * EuP_bary_x + EuO_bary_gpu[2] * EuP_bary_y + EuO_bary_gpu[3] * EuP_bary_z) / (n2 * n1))
                phase_sum += phase_sub

                # get limb darkening
                ld_sub = quad_limb_darkening_gpu(μ_sub, u1, u2, u3, u4)
                ld_sum += ld_sub

            end
        end
        # take averages
        @inbounds phase[m,n] = phase_sum / μ_count
        @inbounds dA[m,n] = dA_sum 

        @inbounds projected_v[m,n] = projected_v_sum / μ_count
        @inbounds ld[m,n,1] = ld_sum / μ_count
    end

    return nothing
end

function calc_europa_quantities_gpu!(epoch::T1, obs_long::T1, obs_lat::T1, alt::T1, wavelength, OP_mech, N, Nθ, Nsubgrid, 
                                        gpu_allocs, B_OPE, h_OPE) where {T1<:AF}

    phase = gpu_allocs.phase
    ld = gpu_allocs.ld
    projected_v = gpu_allocs.projected_v
    dA = gpu_allocs.dA

    # re-zero everything
    CUDA.@sync begin
        phase .= 0.0
        ld .= 0.0
        projected_v .= 0.0
        dA .= 0.0
    end

    # convert scalars from disk params to desired precision
    A = convert(T1, 14.713)
    B = convert(T1, -2.396)
    C = convert(T1, -1.787)
    B_OPE = convert(T1, B_OPE)
    h_OPE = convert(T1, h_OPE)

    filtered_df = quad_ld_coeff_SSD_Europa[quad_ld_coeff_SSD_Europa.wavelength .== wavelength, :]
    u1 = convert(T1, filtered_df.law_4u1[1])
    u2 = convert(T1, filtered_df.law_4u2[1])
    u3 = convert(T1, filtered_df.law_4u3[1])
    u4 = convert(T1, filtered_df.law_4u4[1])

    # geometry from disk
    Nϕ = N
    Nθ_max = maximum(Nθ)

    # query SPICE for E, S, M position (km) and velocities (km/s)
    BE_bary = spkssb(399,epoch,"J2000")

    # determine xyz earth coordinates for lat/long of observatory
    flat_coeff = (earth_radius - earth_radius_pole) / earth_radius
    EO_earth_pos = georec(deg2rad(obs_long), deg2rad(obs_lat), alt, earth_radius, flat_coeff)
    # set earth velocity vectors
    EO_earth = vcat(EO_earth_pos, [0.0, 0.0, 0.0])
    # transform into ICRF frame
    EO_bary = sxform("IAU_EARTH", "J2000", epoch) * EO_earth
    CUDA.@sync EO_bary_gpu = CuArray(EO_bary)

    # get vector from barycenter to observatory on Earth's surface
    BO_bary = BE_bary .+ EO_bary

    # set string for ltt and abberation
    lt_flag = "CN+S"

    #Europa position vectors:
    EuE_bary = spkezp(399, epoch, "J2000", lt_flag, 502)[1]
    EuS_bary = spkezp(10, epoch, "J2000", lt_flag, 502)[1]
    phase_angle = acos(calc_mu(EuE_bary, EuS_bary))

    CUDA.@sync EuO_bary_gpu = CuArray(EuE_bary[1:3] + EO_bary[1:3])
    CUDA.@sync EuS_bary_gpu = CuArray(EuS_bary)
    CUDA.@sync moon_radius_gpu = CuArray([moon_radius])
    CUDA.@sync earth_radius_gpu = CuArray([earth_radius])
    CUDA.@sync sun_radius_gpu = CuArray([sun_radius])

    # get light travel time corrected OS vector
    OS_bary, OS_lt, OS_dlt = spkltc(10, epoch, "J2000", lt_flag, BO_bary)
    CUDA.@sync OS_bary_gpu = CuArray(OS_bary)

    # get vector from observatory on earth's surface to moon center
    OM_bary, OM_lt, OM_dlt = spkltc(301, epoch, "J2000", lt_flag, BO_bary)
    CUDA.@sync OM_bary_gpu = CuArray(OM_bary)

    # get modified epch
    epoch_lt = epoch - OS_lt

    # get rotation matrix for sun
    sun_rot_mat = pxform("IAU_SUN", "J2000", epoch_lt)
    CUDA.@sync sun_rot_mat_gpu = CuArray(sun_rot_mat)

    CUDA.@sync begin
        Nθ = CuArray{Float64}(Nθ)
    end

    # compute geometric parameters, average over subtiles
    threads1 = 256
    blocks1 = cld(prod(CUDA.size(phase)), prod(threads1))
    CUDA.@sync @cuda threads=threads1 blocks=blocks1 calc_europa_quantities_gpu!(phase, Nϕ, Nθ, Nsubgrid, Nθ_max, ld, projected_v,
                                                                            dA, moon_radius_gpu, OS_bary_gpu, OM_bary_gpu, EO_bary_gpu,
                                                                            sun_rot_mat_gpu, sun_radius_gpu, A, B, C, u1, u2, u3, u4, 
                                                                            earth_radius_gpu, EuO_bary_gpu, EuS_bary_gpu, OP_mech, B_OPE, h_OPE)
    return
end                               

function calc_europa_quantities_gpu!(phase, Nϕ, Nθ, Nsubgrid, Nθ_max, ld, projected_v, 
                                      dA, moon_radius, OS_bary, OM_bary, EO_bary,
                                      sun_rot_mat, sun_radius, A, B, C, u1, u2, u3, u4, 
                                      earth_radius, EuO_bary_gpu, EuS_bary_gpu, OP_mech, B_OPE, h_OPE) 
    sun_radius = sun_radius[1] 
    # get indices from GPU blocks + threads
    idx = threadIdx().x + blockDim().x * (blockIdx().x-1)
    sdx = gridDim().x * blockDim().x

    # get number of elements along tile side
    k = Nsubgrid

    # total number of elements output array
    num_tiles = Nϕ * Nθ_max

    # get latitude subtile step size
    N_ϕ_edges = Nϕ * Nsubgrid
    dϕ = π / (N_ϕ_edges)

    for t in idx:sdx:num_tiles
        # get index for output array 
        row = (t - 1) ÷ Nθ_max
        col = (t - 1) % Nθ_max
        m = row + 1
        n = col + 1

        # get indices for input array
        i = row * k + 1
        j = col * k + 1

        # get number of longitude tiles in course latitude slice
        N_θ_edges = Nθ[m] * Nsubgrid

        # set up sum holders for scalars
        phase_sum = CUDA.zero(CUDA.eltype(phase))
        dA_sum = CUDA.zero(CUDA.eltype(phase))
        ld_sum = CUDA.zero(CUDA.eltype(phase))
        projected_v_sum = CUDA.zero(CUDA.eltype(phase))

        # initiate counter
        μ_count = 0

        # loop over latitude sub tiles
        for ti in i:i+k-1
            # get coordinates of latitude subtile center
            ϕc_sub = -π/2 + (dϕ/2.0) + (ti - 1) * dϕ
    
            # loop over longitude subtiles
            for tj in j:j+k-1
                # move on if we've looped past last longitude
                if tj > N_θ_edges
                    continue
                end
    
                # get longitude subtile step size
                dθ = 2π / (N_θ_edges)
    
                # get longitude
                θc_sub = (dθ/2.0) + (tj - 1) * dθ  

                # get cartesian coords of patch
                x, y, z = sphere_to_cart_gpu(sun_radius, ϕc_sub, θc_sub)

                # get vector from spherical circle center to surface patch
                a = x
                b = y
                c = CUDA.zero(CUDA.eltype(phase))

                # take cross product to get vector in direction of rotation
                d = - sun_radius * b
                e = sun_radius * a
                f = CUDA.zero(CUDA.eltype(phase))

                # make it a unit vector
                def_norm = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
                d /= def_norm
                e /= def_norm
                f /= def_norm

                # set magnitude by differential rotation
                rp = 2π * sun_radius * CUDA.cos(ϕc_sub) / rotation_period_gpu(ϕc_sub, A, B, C)

                # get in units of c
                rp /= 86400.0

                # set magnitude of vector
                d *= rp
                e *= rp
                f *= rp

                # xyz rotated for SP bary
                x_new = sun_rot_mat[1] * x + sun_rot_mat[4] * y + sun_rot_mat[7] * z
                y_new = sun_rot_mat[2] * x + sun_rot_mat[5] * y + sun_rot_mat[8] * z
                z_new = sun_rot_mat[3] * x + sun_rot_mat[6] * y + sun_rot_mat[9] * z

                # vel component rotated for SP bary
                vx = sun_rot_mat[1] * d + sun_rot_mat[4] * e + sun_rot_mat[7] * f
                vy = sun_rot_mat[2] * d + sun_rot_mat[5] * e + sun_rot_mat[8] * f
                vz = sun_rot_mat[3] * d + sun_rot_mat[6] * e + sun_rot_mat[9] * f

                # OP_bary state vector
                OP_bary_x = OS_bary[1] + x_new
                OP_bary_y = OS_bary[2] + y_new
                OP_bary_z = OS_bary[3] + z_new

                # Europa to position state vector
                EuP_bary_x = EuS_bary_gpu[1] + x_new
                EuP_bary_y = EuS_bary_gpu[2] + y_new
                EuP_bary_z = EuS_bary_gpu[3] + z_new

                # calculate mu
                μ_sub = calc_mu_gpu(x_new, y_new, z_new, OP_bary_x, OP_bary_y, OP_bary_z) 
                if μ_sub <= 0.0
                    continue
                end
                μ_count += 1

                # get OP_bary and SP_bary between them and find projected_velocities_no_cb
                n1 = CUDA.sqrt(OP_bary_x^2.0 + OP_bary_y^2.0 + OP_bary_z^2.0)
                n2 = CUDA.sqrt(vx^2.0 + vy^2.0 + vz^2.0)
                angle = (OP_bary_x * vx + OP_bary_y * vy + OP_bary_z * vz) / (n1 * n2)
                v_rot_sub = (n2 * angle)
                v_rot_sub *= 1000.0
                projected_v_sum += v_rot_sub

                # get projected area element
                dA_sub = calc_dA_gpu(sun_radius, ϕc_sub, dϕ, dθ)
                dA_sub *= μ_sub
                dA_sum += dA_sub

                # calculate distance (Europa)
                n1 = CUDA.sqrt(EuP_bary_x^2.0 + EuP_bary_y^2.0 + EuP_bary_z^2.0)  
                n2 = CUDA.sqrt(EuO_bary_gpu[1]^2.0 + EuO_bary_gpu[2]^2.0 + EuO_bary_gpu[3]^2.0)  
                phase_sub = acos((EuO_bary_gpu[1] * EuP_bary_x + EuO_bary_gpu[2] * EuP_bary_y + EuO_bary_gpu[3] * EuP_bary_z) / (n2 * n1))
                phase_sum += phase_sub

                # get limb darkening
                ld_sub = quad_limb_darkening_gpu(μ_sub, u1, u2, u3, u4)

                if OP_mech == 0
                    ld_sum += ld_sub*cb_phase_curve(phase_sub, B_OPE, h_OPE)
                elseif OP_mech == 1
                    ld_sum += ld_sub*sh_phase_curve(phase_sub, B_OPE, h_OPE)
                end
            end
        end
        # take averages
        @inbounds phase[m,n] = phase_sum / μ_count
        @inbounds dA[m,n] = dA_sum 

        @inbounds projected_v[m,n] = projected_v_sum / μ_count
        @inbounds ld[m,n,1] = ld_sum / μ_count
    end

    return nothing
end