struct GPUAllocs{T1<:AF}
    phase::CuArray{T1,2}
    dA::CuArray{T1,2}
    ld::CuArray{T1,3}
    projected_v::CuArray{T1,2}
end

function GPUAllocs(N, Nt, Nθ; precision::DataType=Float64)
    # allocate memory for precomputation
    Nϕ = N
    Nθ_max = maximum(Nθ)

    # allocate memory for pre-computations
    CUDA.@sync  begin
        phase = CUDA.zeros(precision, Nϕ, Nθ_max)
        dA = CUDA.zeros(precision, Nϕ, Nθ_max)
        ld = CUDA.zeros(precision, Nϕ, Nθ_max, 1)
        projected_v = CUDA.zeros(precision, Nϕ, Nθ_max)
    end

    return GPUAllocs(phase, dA, ld, projected_v)
end