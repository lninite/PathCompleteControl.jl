module Utils

using SwitchOnSafety
using HybridSystems
using MosekTools
using StaticArrays
using JuMP
using LinearAlgebra

export de_bruijn, feedback_tree, white_box_JSR, common_Lyap, largest_gamma_bisection

function de_bruijn(order::Int, n_modes::Int; mode::Symbol=:complete)
    tuples = Iterators.product([1:n_modes for _ ∈ 1:order]...)
    tuples_to_id = Dict(zip(tuples, 1:length(tuples)))
    automaton = GraphAutomaton(length(tuples))

    for (i, tuple) ∈ enumerate(tuples)
        for j ∈ 1:n_modes 
            tuple_to = (j, tuple[1:end-1]...)
            if mode == :complete
                add_transition!(automaton, i, tuples_to_id[tuple_to], j)
            elseif mode == :co_complete 
                add_transition!(automaton, tuples_to_id[tuple_to], i, j)
            end
        end
    end
    return automaton
end

function common_Lyap(A::Vector{<:Any}, gamma::Float64 = 1.0)
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    dim, _ = size(A[1])
    model = Model(solver)
    @variable(model, P[1:dim, 1:dim])
    @constraint(model, P in PSDCone())
    @objective(model, Min, 0) 
    for σ in 1:length(A)
        @constraint(
            model, 
            P - (gamma*A[σ]') * P * (gamma*A[σ]) in PSDCone()
        )
    end
    @constraint(model, P - I in PSDCone())
    @constraint(model, 100*I - P in PSDCone())

    JuMP.optimize!(model)
    if termination_status(model) ∈ [MOI.OPTIMAL, MOI.SLOW_PROGRESS]
        return value.(P)
    else
        error("The common Lyapunov problem is infeasible!")
    end
end

function largest_gamma_bisection(
    A::Vector{<:Any}, 
    gamma_lower::Float64 = 0.0,
    gamma_upper::Float64 = 100.0,
    tol::Float64 = 1e-4,
    max_iter::Int = 50
)
    for iter in 1:max_iter
        gamma_mid = (gamma_lower + gamma_upper) / 2.0
        try
            common_Lyap(A, gamma_mid)
            gamma_lower = gamma_mid
        catch
            gamma_upper = gamma_mid
        end
        if gamma_upper - gamma_lower < tol
            break
        end
    end
    
    return (gamma_lower + gamma_upper) / 2.0
end

end
