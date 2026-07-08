module Bounds

using StaticArrays
using HybridSystems
using JuMP
using MosekTools
using LinearAlgebra
import MathOptInterface as MOI

export pclqr_control

function pclqr_control(
    A::Vector{SMatrix{Nx, Nx, T, Nxx}}, 
    B::Vector{SMatrix{Nx, Nu, T, Nxu}}, 
    Q::SMatrix{Nx, Nx, T, Nxx},
    R::SMatrix{Nu, Nu, T, Nuu},
    pc::GraphAutomaton 
) where {Nx, Nu, T, Nxx, Nxu, Nuu}
    """
    Compute optimal Lyapunov matrices P and gains K for path-complete LQR.
    
    ∀(i, j, σ) ∈ PC : Pi ⪰ (R + Ki^⊤QKi) + (Aσ + BσKi)^⊤ Pj (Aσ + BσKi)
    """
    psd_margin = 1e-4
    V = nstates(pc)
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    dim, dim_in = size(B[1])
    lower_triangular(P) = [P[i, j] for i = 1:size(P, 1) for j = 1:i]
    psd_eye(n) = psd_margin * Matrix{Float64}(I, n, n)
    
    model = Model(solver)
    S = [@variable(model, [1:dim, 1:dim], Symmetric) for s in 1:V]
    @constraint(model, [s = 1:V], S[s] >= psd_eye(dim), PSDCone())
    Y = [@variable(model, [1:dim_in, 1:dim]) for s in 1:V]
    @variable(model, t[1:V])
    @constraint(model, [s = 1:V], [t[s]; 1; lower_triangular(S[s])] in MOI.LogDetConeTriangle(dim))
    @objective(model, Max, sum([t[s] for s ∈ 1:V])) 
   
    for trans ∈ transitions(pc)
        i, j = trans.edge.src, trans.edge.dst
        σ = pc.Σ[trans.edge][trans.id]
        X11 = S[i]
        X12 = S[i] * A[σ]' + (B[σ] * Y[i])'
        X13 = S[i]
        X14 = Y[i]'
        X22 = S[j]
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
        P = [inv(value.(S[i])) for i ∈ 1:V]
        K = [value.(Y[i]) * P[i] for i ∈ 1:V]
        
        # Feasibility check
        for trans ∈ transitions(pc)
            i, j = trans.edge.src, trans.edge.dst   
            σ = pc.Σ[trans.edge][trans.id]
            LHS = P[i] - Q - K[i]' * R * K[i] - (A[σ] + B[σ] * K[i])' * P[j] * (A[σ] + B[σ] * K[i])
            λ_min = eigmin(Symmetric(LHS))
            if λ_min < -1e-6*max(1, opnorm(LHS))
                println("LMI violated at transition (i=$i, j=$j, σ=$σ): minimum eigenvalue = ", λ_min)
            end
        end
        return P, K
    else
        @error "The LQR problem is infeasible!"
    end
end

end # module
