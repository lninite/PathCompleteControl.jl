module MPC

using LinearAlgebra
using StaticArrays
using JuMP
using MosekTools

export robust_mpc, robust_mpc_terminal_cost

function robust_mpc(
    x0::SVector{Nx,T},
    A::AbstractVector{<:SMatrix{Nx,Nx,T,Nxx}},
    B::AbstractVector{<:SMatrix{Nx,Nu,T,Nxu}},
    Q::SMatrix{Nx,Nx,T,Nxx},
    R::SMatrix{Nu,Nu,T,Nuu};
    horizon::Int = 4
) where {Nx,Nu,T,Nxx,Nxu,Nuu}

    num_modes = length(A)
    nu = size(B[1], 2)

    switching_sequences = collect(
        Iterators.product(
            [1:num_modes for _ in 1:horizon]...
        )
    )

    model = Model(Mosek.Optimizer)
    set_silent(model)

    U = Dict{Tuple{Vararg{Int}}, Vector{VariableRef}}()

    for t in 0:horizon-1
        prefixes = t == 0 ? [()] : collect(Iterators.product([1:num_modes for _ in 1:t]...))
        for prefix in prefixes
            U[prefix] = [@variable(model) for _ in 1:nu]
        end
    end

    @variable(model, γ)

    for σ in switching_sequences
        x = x0
        cost = @expression(model, 0.0)

        for t in 0:horizon-1
            mode = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end

        @constraint(model, cost <= γ)
    end

    @objective(model, Min, γ)
    optimize!(model)

    return collect(Float64.(value.(U[()])))
end

function robust_mpc_terminal_cost(
    x0::SVector{Nx,T},
    A::AbstractVector{<:SMatrix{Nx,Nx,T,Nxx}},
    B::AbstractVector{<:SMatrix{Nx,Nu,T,Nxu}},
    Q::SMatrix{Nx,Nx,T,Nxx},
    R::SMatrix{Nu,Nu,T,Nuu},
    P::AbstractVector{<:AbstractMatrix{T}};
    horizon::Int = 4
) where {Nx,Nu,T,Nxx,Nxu,Nuu}

    num_modes = length(A)
    nu = size(B[1], 2)

    switching_sequences = collect(
        Iterators.product(
            [1:num_modes for _ in 1:horizon]...
        )
    )

    model = Model(Mosek.Optimizer)
    set_silent(model)

    U = Dict{Tuple{Vararg{Int}}, Vector{VariableRef}}()
    t_var = Dict{Tuple{Vararg{Int}}, VariableRef}()

    for t in 0:horizon-1
        prefixes = t == 0 ? [()] : collect(Iterators.product([1:num_modes for _ in 1:t]...))
        for prefix in prefixes
            U[prefix] = [@variable(model) for _ in 1:nu]
        end
    end

    for σ in switching_sequences
        t_var[Tuple(σ)] = @variable(model)
    end

    @variable(model, γ)

    for σ in switching_sequences
        x = x0
        cost = @expression(model, 0.0)

        for t in 0:horizon-1
            mode = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end

        cost += t_var[Tuple(σ)]
        @constraint(model, cost <= γ)

        for i in 1:length(P)
            @constraint(model, [t_var[Tuple(σ)] x'; x inv(P[i])] in PSDCone())
        end
    end

    @objective(model, Min, γ)
    optimize!(model)

    return collect(Float64.(value.(U[()])))
end

end # module
