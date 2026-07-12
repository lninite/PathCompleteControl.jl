module AlternateOpt

using StaticArrays
using HybridSystems
using JuMP
using MosekTools
using LinearAlgebra
import MathOptInterface as MOI

export alternate_optimization

function optimize_K_stab(
    P::AbstractVector{<:AbstractMatrix{T}},
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    dim = size(A[1], 1)
    udim = size(B[1], 2)

    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)

    @variable(model, K[1:udim, 1:dim])
    @variable(model, γ >= 0)

    for trans in transitions(pc)
        α = trans.edge.src
        β = trans.edge.dst
        i = pc.Σ[trans.edge][trans.id]

        @constraint(model, [γ*P[α] - Q K' (A[i] + B[i]*K)';
                            K inv(R) zeros(udim, dim);
                            A[i] + B[i]*K zeros(dim, udim) inv(P[β])] in PSDCone())
    end

    @objective(model, Min, γ)
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        println("Status: ", termination_status(model))
    end

    return value.(K), objective_value(model)
end

function optimize_K_fixed_gamma(
    γfix::Real,
    P::AbstractVector{<:AbstractMatrix{T}},
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    dim = size(A[1], 1)
    udim = size(B[1], 2)

    model = Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true))

    @variable(model, K[1:udim, 1:dim])
    @variable(model, γp >= 0)

    for trans in transitions(pc)
        α = trans.edge.src
        β = trans.edge.dst
        i = pc.Σ[trans.edge][trans.id]

        @constraint(model,
            [γp*P[α] - Q K' (A[i] + B[i]*K)';
             K inv(R) zeros(udim, dim);
             A[i]+B[i]*K zeros(dim, udim) (1/γfix)*inv(P[β])]
            in PSDCone()
        )
    end

    @objective(model, Min, γp)
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        println(termination_status(model))
        return nothing, Inf, false
    end

    γpstar = value(γp)
    return value.(K), γpstar, γpstar <= γfix
end

function optimize_K(
    P::AbstractVector{<:AbstractMatrix{T}},
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton;
    γ_low::Real = 0.0,
    γ_high::Real = 100.0,
    tol::Real = 1e-3,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    bestK = nothing
    bestγ = γ_high

    while γ_high - γ_low > tol
        γmid = (γ_low + γ_high) / 2

        K, γp, feasible = optimize_K_fixed_gamma(γmid, P, A, B, Q, R, pc)

        if feasible
            bestK = K
            bestγ = γmid
            γ_high = γmid
        else
            γ_low = γmid
        end
    end

    return bestK, bestγ
end

function feasible_P(
    K::AbstractMatrix{T},
    γ::Real,
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton;
    verbose::Bool = false,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    dim = size(A[1], 1)

    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)

    P = [@variable(model, [1:dim, 1:dim], Symmetric) for _ in 1:nstates(pc)]

    for α in 1:nstates(pc)
        @constraint(model, P[α] in PSDCone())
        @constraint(model, P[α] >= 1e-6*I, PSDCone())
    end

    for trans in transitions(pc)
        α = trans.edge.src
        β = trans.edge.dst
        i = pc.Σ[trans.edge][trans.id]
        Acl = A[i] + B[i]*K

        @constraint(
            model,
            γ*P[α] - Q - K' * R * K - Acl' * P[β] * Acl >= 0,
            PSDCone()
        )
    end

    @objective(model, Min, sum(tr(P[α]) for α in 1:nstates(pc)))

    optimize!(model)

    feasible = true

    if termination_status(model) == MOI.OPTIMAL
        return feasible, [value.(Pα) for Pα in P]
    else
        for trans in transitions(pc)
            α = trans.edge.src
            β = trans.edge.dst
            i = pc.Σ[trans.edge][trans.id]
            Acl = A[i] + B[i]*K
            Pα = value.(P[α])
            Pβ = value.(P[β])
            M = γ*Pα - Q - K' * R * K - Acl' * Pβ * Acl
            if minimum(eigvals(Array(M))) < -1e-4
                if verbose
                    println("Constraint violated for transition $trans: minimum eigenvalue = $(minimum(eigvals(Array(M))))")
                end
                feasible = false
                break
            end
        end
        for α in 1:nstates(pc)
            Pα = value.(P[α])
            if minimum(eigvals(Array(Pα))) < -1e-4
                if verbose
                    println("P[$α] is not positive definite: minimum eigenvalue = $(minimum(eigvals(Array(Pα))))")
                end
                feasible = false
                break
            end
        end
        if !feasible
            return false, nothing
        else
            return true, [value.(Pα) for Pα in P]
        end
    end
end

function optimize_P(
    K::AbstractMatrix{T},
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton;
    γ_low::Real = 0.0,
    γ_high::Real = 100.0,
    tol::Real = 1e-4,
    verbose::Bool = false,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    Pbest = nothing

    while γ_high - γ_low > tol
        γ = (γ_low + γ_high) / 2

        feasible, P = feasible_P(K, γ, A, B, Q, R, pc, verbose = verbose)

        if feasible
            γ_high = γ
            Pbest = P
        else
            γ_low = γ
        end
    end

    return Pbest, γ_high
end

# Initialization phase for the alternating optimization algorithm
function initialize_controller(
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    pc::GraphAutomaton;
    N::Int = 50,
    verbose::Bool = false,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}

    dim = size(A[1], 1)

    P = [Matrix{Float64}(I, dim, dim) for _ in 1:nstates(pc)]
    K = zeros(size(B[1], 2), dim)

    # Small weights used only during the stabilization phase
    Qstab = SMatrix{Nx,Nx,Float64}(I / 100)
    Rstab = SMatrix{Nu,Nu,Float64}(I / 100)

    for iter in 1:N
        if verbose
            println("\n================================================")
            println("Stabilization iteration $iter")
            println("================================================")
        end

        K, γK = optimize_K_stab(P, A, B, Qstab, Rstab, pc)

        if verbose
            println("γ(K-step) = $γK")
        end

        P, γP = optimize_P(
            K,
            A,
            B,
            Qstab,
            Rstab,
            pc;
            verbose = verbose,
        )

        if verbose
            println("γ(P-step) = $γP")
        end

        if γP < 1
            if verbose
                println("Stabilization phase converged: γ < 1 at iteration $iter")
            end
            return K, P
        end
    end

    error(
        "Failed to find a stabilizing controller after $N iterations. γ = $γP"
    )
end

# Main function : compute a stabilizing K and P, then alternate between optimizing K and P until convergence 
function alternate_optimization(
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}},
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}},
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton;
    N::Int = 50,
    verbose::Bool = false,
    tol::Real = 1e-3,
) where {Nx, Nu, T, Nxx, Nxu, Nuu}

    K, P = initialize_controller(
    A,
    B,
    pc;
    N = N,
    verbose = verbose,
    )
    
    K_old = copy(K)
    P_old = copy(P)

    for iter in 1:N
        if verbose
            println("\n================================================")
            println("Iteration $iter")
            println("================================================")
        end

        feasible, P = feasible_P(K, 1, A, B, Q, R, pc, verbose = verbose)

        if !feasible
            error("P-step failed")
        end

        K, γK = optimize_K(P, A, B, Q, R, pc)

        if verbose
            println("γ(K-step) = $γK")
            println("K = ")
            println(K)
        end

        if iter > 1
            K_norm_diff_opt = norm(K - K_old)
            P_norm_diffs_opt = [norm(P[i] - P_old[i]) for i in 1:length(P)]
            max_P_diff_opt = maximum(P_norm_diffs_opt)

            if verbose
                println("  norm(K - K_old) = $K_norm_diff_opt")
                println("  max norm(P[i] - P_old[i]) = $max_P_diff_opt")
            end

            if K_norm_diff_opt < tol && max_P_diff_opt < tol
                if verbose
                    println("Converged at iteration $iter: K and P changes below threshold")
                end
                break
            end
        end

        K_old = copy(K)
        P_old = copy(P)
    end

    return K, P
end

end # module