# TreeMPC must be the same controller as robust_mpc and robust_mpc_terminal_cost (src/mpc.jl).

using Test
using LinearAlgebra
using StaticArrays
using JuMP
using HybridSystems
using PathCompleteControl

# System and terminal costs of experiments/temperature_regulation.jl

c = 1375.0
tau = 20.0

A = [
    @SMatrix [
        1 - tau/c*(1/1.5 + 1/R13 + 1/3)   tau/c/1.5                         tau/c/R13
        tau/c/1.5                         1 - tau/c*(1/1.5 + 1/R23 + 1/3)   tau/c/R23
        tau/c/R13                         tau/c/R23                         1 - tau/c*(1/R13 + 1/R23 + 1/2.7)
    ]
    for R13 in (1.2, 0.8) for R23 in (1.2, 0.8)
]
B = [SMatrix{3,3,Float64}(tau/c * I) for _ in eachindex(A)]
Q = SMatrix{3,3,Float64}(I)
R = SMatrix{3,3,Float64}(I)

automaton = GraphAutomaton(1)
for σ in eachindex(A)
    add_transition!(automaton, 1, 1, σ)
end
P_common, _ = single_sdp(A, B, Q, R, automaton)
_, P_dual = alternate_optimization(A, B, Q, R, de_bruijn(1, length(A), mode = :co_complete); N = 10)

terminal_costs = [("no terminal cost", nothing), ("common quadratic", P_common), ("dual De Bruijn", P_dual)]
states = [SVector(5.0, -5.0, 5.0), SVector(2.0, 1.0, -1.5), SVector(-0.3, 0.2, 0.4)]

original(x, P, N) = P === nothing ? robust_mpc(x, A, B, Q, R; horizon = N) :
                                    robust_mpc_terminal_cost(x, A, B, Q, R, P; horizon = N)

# Cost that robust_mpc and robust_mpc_terminal_cost minimise, computed without any solver:
# every switching sequence σ is simulated, with u[d][:, k] the control of node k of depth d.
function worst_case_cost(u, x0, P, N)
    M = length(A)
    worst = -Inf
    for σ in Iterators.product(fill(1:M, N)...)
        x, k, cost = x0, 1, 0.0
        for d in 0:N-1
            cost += x' * Q * x + u[d][:, k]' * R * u[d][:, k]
            x = A[σ[d+1]] * x + B[σ[d+1]] * u[d][:, k]
            k = (k - 1) * M + σ[d+1]
        end
        if P !== nothing
            cost += maximum(x' * Pi * x for Pi in P)
        end
        worst = max(worst, cost)
    end
    return worst
end

# Optimal value of the TreeMPC model at x, plus the constant x'Qx that the model leaves out
function tree_value(mpc, x)
    mpc(x)
    return objective_value(mpc.model) + x' * Q * x
end

@testset "TreeMPC" begin
@testset "N = $N, $name" for N in 1:3, (name, P) in terminal_costs

    x = states[1]

    # 1. Same cost function. With all the controls fixed to arbitrary values, the value of
    #    the TreeMPC model is the worst-case cost over all switching sequences.
    mpc = TreeMPC(A, B, Q, R; horizon = N, terminal_cost = P, normalize = false)
    u = Dict(d => [sin(i + 2k + 3d) for i in axes(v, 1), k in axes(v, 2)] for (d, v) in mpc.u)
    for (d, v) in mpc.u
        fix.(v, u[d])
    end
    @test tree_value(mpc, x) ≈ worst_case_cost(u, x, P, N) rtol = 1e-9

    # 2. Same minimiser. The control returned by the original function is optimal for the
    #    TreeMPC model: imposing it as first control does not increase the optimal value.
    mpc = TreeMPC(A, B, Q, R; horizon = N, terminal_cost = P, normalize = false)
    optimal_value = tree_value(mpc, x)
    fix.(mpc.u[0][:, 1], original(x, P, N))
    @test tree_value(mpc, x) ≈ optimal_value rtol = 1e-6

    # 3. Same control, up to the accuracy of the solver. The cost is flat around the
    #    minimiser, so the controls agree to fewer digits than the costs.
    mpc = TreeMPC(A, B, Q, R; horizon = N, terminal_cost = P)
    for x in states
        @test norm(mpc(x) - original(x, P, N)) <= 1e-2 * norm(x)
    end

    # 4. Same closed-loop cost, up to the accuracy of the solver
    sequence = repeat([1, 3, 2, 4, 4, 2, 1, 3, 3, 1], 2)
    J, _, _ = simulate_system(mpc, x, A, B, Q, R, sequence)
    J_original, _, _ = simulate_system(x -> original(x, P, N), x, A, B, Q, R, sequence)
    @test J ≈ J_original rtol = 1e-3

end
end
