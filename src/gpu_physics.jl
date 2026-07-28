function calc_mu_gpu(x, y, z, Ox, Oy, Oz)
    dp = x * Ox + y * Oy + z * Oz
    n1 = CUDA.sqrt(Ox^2.0 + Oy^2.0 + Oz^2.0)
    n2 = CUDA.sqrt(x^2.0 + y^2.0 + z^2.0)
    return dp / (n1 * n2)
end

function calc_mu(xyz, O⃗) 
    return dot(O⃗, xyz) / (norm(O⃗) * norm(xyz))
end

function sphere_to_cart_gpu(ρ, ϕ, θ)
    # compute trig quantities
    sinϕ = CUDA.sin(ϕ)
    sinθ = CUDA.sin(θ)
    cosϕ = CUDA.cos(ϕ)
    cosθ = CUDA.cos(θ)

    # now get cartesian coords
    x = ρ * cosϕ * cosθ
    y = ρ * cosϕ * sinθ
    z = ρ * sinϕ
    return x, y, z
end

function rotation_period_gpu(ϕ, A, B, C)
    sinϕ = sin(ϕ)
    return 360.0/(A + B * sinϕ^2.0 + C * sinϕ^4.0) 
end

function calc_dA_gpu(ρs, ϕc, dϕ, dθ)
    return ρs^2.0 * CUDA.sin(π/2.0 - ϕc) * dϕ * dθ
end

function quad_limb_darkening_gpu(μ, u1, u2, u3, u4)
    return 1.0 - u1 * (1.0 - μ^0.5) - u2 * (1.0 - μ) - u3 * (1.0 - μ^1.5) - u4 * (1.0 - μ^2.0)
end

function cb_phase_curve(theta, B_CB, h_CB)
    x = tan(theta / 2) / h_CB
    delta_CB = (B_CB/2) * ((1 + ((1- exp(-x)) / (x))) / ((1 + x)^2)) 
    return (1 + delta_CB)
end

function sh_phase_curve(theta, B_SH, h_SH)
    delta_SH = B_SH / (1 + (1/h_SH) * tan(theta / 2)) 
    return (1 + delta_SH)
end

function get_Nθ(ϕc, dϕ)
    return ceil(Int, 2π * cos(ϕc) / dϕ)
end

function get_grid_centers(grid::StepRangeLen)
    start = first(grid) + 0.5 * step(grid)
    stop = last(grid) - 0.5 * step(grid)
    return range(start, stop, length=length(grid)-1)
end