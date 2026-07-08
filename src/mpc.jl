module MPC

using LinearAlgebra
using StaticArrays
using JuMP
using MosekTools

export robust_mpc, robust_mpc_terminal_cost

function robust_mpc(
    x0::SVector{D,T},
    Σ_basis::AbstractVector{<:SMatrix{D,D,T,N1}},
    B::AbstractVector{<:SMatrix{D,Nu,T,N2}},
    Q::SMatrix{D,D,T,N3},
    R::SMatrix{Nu,Nu,T,N4};
    horizon::Int = 4
) where {D,T,Nu,N1,N2,N3,N4}

    num_modes::Int = length(Σ_basis)
    nu::Int = size(B[1], 2)

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
            mode::Int = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = Σ_basis[mode] * x + B[mode] * u
        end

        @constraint(model, cost <= γ)
    end

    @objective(model, Min, γ)
    optimize!(model)

    return collect(Float64.(value.(U[()])))
end

function robust_mpc_terminal_cost(
    x0::SVector{D,T},
    Σ_basis::AbstractVector{<:SMatrix{D,D,T,N1}},
    B::AbstractVector{<:SMatrix{D,Nu,T,N2}},
    Q::SMatrix{D,D,T,N3},
    R::SMatrix{Nu,Nu,T,N4},
    P::AbstractVector{<:AbstractMatrix{T}};
    horizon::Int = 4
) where {D,T,Nu,N1,N2,N3,N4}

    num_modes::Int = length(Σ_basis)
    nu::Int = size(B[1], 2)

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
            mode::Int = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = Σ_basis[mode] * x + B[mode] * u
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
