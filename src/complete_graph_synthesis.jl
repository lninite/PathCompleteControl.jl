module CompleteGraphSynthesis

using StaticArrays
using HybridSystems
using JuMP
using MosekTools
using LinearAlgebra
import MathOptInterface as MOI

export pclqr_control

function pclqr_control(
    A::AbstractVector{<:SMatrix{Nx, Nx, T, Nxx}}, 
    B::AbstractVector{<:SMatrix{Nx, Nu, T, Nxu}}, 
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton 
) where {Nx, Nu, T, Nxx, Nxu, Nuu}

    psd_margin = 1e-4
    n_nodes = nstates(pc)
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    dim, dim_in = size(B[1])
    lower_triangular(P) = [P[i, j] for i = 1:size(P, 1) for j = 1:i]
    psd_eye(n) = psd_margin * Matrix{Float64}(I, n, n)
    
    model = Model(solver)
    S = [@variable(model, [1:dim, 1:dim], Symmetric) for _ in 1:n_nodes]
    @constraint(model, [α = 1:n_nodes], S[α] >= psd_eye(dim), PSDCone())
    Y = [@variable(model, [1:dim_in, 1:dim]) for _ in 1:n_nodes]
    @variable(model, t[1:n_nodes])
    @constraint(model, [α = 1:n_nodes], [t[α]; 1; lower_triangular(S[α])] in MOI.LogDetConeTriangle(dim))
    @objective(model, Max, sum([t[α] for α ∈ 1:n_nodes])) 
   
    for trans ∈ transitions(pc)
        α, β = trans.edge.src, trans.edge.dst
        i = pc.Σ[trans.edge][trans.id]
        X11 = S[α]
        X12 = S[α] * A[i]' + (B[i] * Y[α])'
        X13 = S[α]
        X14 = Y[α]'
        X22 = S[β]
        X23 = zeros(dim, dim)
        X24 = zeros(dim, dim_in)
        X33 = inv(Q)
        X34 = zeros(dim, dim_in)
        X44 = inv(R)
        @constraint(
            model, 
            [
                X11 X12 X13 X14;
                X12' X22 X23 X24;
                X13' X23' X33 X34;
                X14' X24' X34' X44
            ] 
            >= psd_eye(3*dim + dim_in), 
            PSDCone()
        )
    end
    
    JuMP.optimize!(model)
    if termination_status(model) ∈ [MOI.OPTIMAL, MOI.SLOW_PROGRESS]
        P = [inv(value.(S[α])) for α ∈ 1:n_nodes]
        K = [value.(Y[α]) * P[α] for α ∈ 1:n_nodes]
        
        # Feasibility check
        for trans ∈ transitions(pc)
            α, β = trans.edge.src, trans.edge.dst   
            i = pc.Σ[trans.edge][trans.id]
            LHS = P[α] - Q - K[α]' * R * K[α] - (A[i] + B[i] * K[α])' * P[β] * (A[i] + B[i] * K[α])
            λ_min = eigmin(Symmetric(LHS))
            if λ_min < -1e-6*max(1, opnorm(LHS))
                println("LMI violated at transition (α=$α, β=$β, i=$i): minimum eigenvalue = ", λ_min)
            end
        end
        return P, K
    else
        @error "The LQR problem is infeasible!"
    end
end

end # module
