"""
Code for generating Table 1 of the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"
"""

using LinearAlgebra
using StaticArrays
using PathCompleteControl
using HybridSystems
using Random
using Statistics
using Serialization
using Dates

Random.seed!(0)

# ============================================================
# System
# ============================================================

dim = 3
dim_in = 3

c = 1375.0
tau = 20.0

A = Vector{SMatrix{dim,dim,Float64,dim*dim}}()

for R13 in (1.2, 0.8), R23 in (1.2, 0.8)

    push!(A,
        @SMatrix [
            1 - tau/c*(1/1.5 + 1/R13 + 1/3)   tau/c/1.5                     tau/c/R13
            tau/c/1.5                         1 - tau/c*(1/1.5 + 1/R23 + 1/3) tau/c/R23
            tau/c/R13                         tau/c/R23                     1 - tau/c*(1/R13 + 1/R23 + 1/2.7)
        ]
    )

end

Bmat = tau/c * Matrix(I, dim_in, dim_in)

B = [
    SMatrix{dim,dim_in,Float64,dim*dim_in}(Bmat)
    for _ in eachindex(A)
]

Q = SMatrix{dim,dim,Float64}(I)
R = SMatrix{dim_in,dim_in,Float64}(I)

x0 = @SVector [5.0, -5.0, 5.0]

# ============================================================
# Parameters
# ============================================================

Tsim = 300 # 300 
num_trials = 50 # 50

order_max = 4
mpc_horizons = [2,3,4,5] # [2,3,4,5]

# ============================================================
# Storage for results
# ============================================================

# storage for complete graph results (SDP)
pc_results = Dict{Int, Dict{Symbol, Any}}()

# storage for co-complete graph results (alternate optimization)
pc_co_results = Dict{Int, Dict{Symbol, Any}}()

# storage for MPC results
# without terminal cost
mpc_results = Dict{Any, Any}()
# with terminal cost (common P)
mpc_terminal_cost_common_results = Dict{Any, Any}()
# with terminal cost (order-1 dual De Bruijn graph)
mpc_terminal_cost_results = Dict{Any, Any}()

for N in mpc_horizons
    mpc_results[N] = Dict(
        :cost => Float64[],
        :t_on => 0.0
    )
    mpc_terminal_cost_common_results[N] = Dict(
        :cost => Float64[],
        :t_on => 0.0,
        :t_off => 0.0
    )
    mpc_terminal_cost_results[N] = Dict(
        :cost => Float64[], 
        :t_on => 0.0, 
        :t_off => 0.0)
end

trajectories = Dict(
    :complete => Vector{Vector{SVector{dim,Float64}}}(),
    :cocomplete => Vector{Vector{SVector{dim,Float64}}}(),
    :mpc => Vector{Vector{SVector{dim,Float64}}}(),
    :mpc_terminal_common => Vector{Vector{SVector{dim,Float64}}}(),
    :mpc_terminal_dualDb => Vector{Vector{SVector{dim,Float64}}}()
)

println("\n================ OFFLINE COMPUTATION =================")

# Complete graphs (policy and upper bound computed through SDP)
for ℓ in 1:order_max

    pc = de_bruijn(ℓ, length(A), mode = :complete)

    t_off = @elapsed begin
        P, K = pclqr_control(
            A,
            B,
            Q,
            R,
            pc
        )
    end

    pc_results[ℓ] = Dict(
        :P => P,
        :K => K,
        :V => minimum(x0' * Pi * x0 for Pi in P),
        :t_off => t_off,
        :t_on => 0.0,
        :cost => Float64[]
    )

end

# Co-complete graphs (policy and upper bound computed through alternate optimization)
for ℓ in 1:order_max

    pc = de_bruijn(
        ℓ,
        length(A),
        mode = :co_complete
    )

    t_off = @elapsed begin

        K, P =
            alternate_optimization(
                A,
                B,
                Q,
                R,
                pc;
                N = 50
            )

    end

    pc_co_results[ℓ] = Dict(
        :P => P,
        :K => K,
        :V => maximum(x0' * Pi * x0 for Pi in P),
        :t_off => t_off,
        :t_on => 0.0,
        :cost => Float64[]
    )

end

# Terminal cost function for MPC (common Lyapunov function)
automaton = GraphAutomaton(1)

for σ in 1:length(A)
    add_transition!(automaton, 1, 1, σ)
end

t_off = @elapsed begin

    P_common, K_common =
        pclqr_control(
            A,
            B,
            Q,
            R,
            automaton
        )

end

mpc_terminal_cost_common_results[:P] = P_common
mpc_terminal_cost_common_results[:K] = K_common
mpc_terminal_cost_common_results[:t_off] = t_off

# Terminal cost function for MPC (dual De Bruijn graph)
pc = de_bruijn(
    1,
    length(A),
    mode = :co_complete
)

t_off = @elapsed begin

    K_terminal,
    P_terminal =
        alternate_optimization(
            A,
            B,
            Q,
            R,
            pc;
            N = 10
        )

end

N = mpc_horizons[end]

mpc_terminal_cost_results[N][:P] = P_terminal
mpc_terminal_cost_results[N][:K] = K_terminal
mpc_terminal_cost_results[N][:t_off] = t_off

println("\n================ MONTE CARLO =================")

all_switching_sequences = Vector{Vector{Int}}()

for trial in 1:num_trials

    switching_sequence = rand(1:length(A), Tsim)
    push!(all_switching_sequences, switching_sequence)

    # ========================================================
    # MPC
    # ========================================================

    for N in mpc_horizons

        # ---------------- MPC ----------------

        controller = x -> robust_mpc(
            x,
            A,
            B,
            Q,
            R;
            horizon = N
        )

        J, t_on, traj = simulate_system(
            controller,
            x0,
            A,
            B,
            Q,
            R,
            switching_sequence;
            save_trajectory = (N == mpc_horizons[end] && trial == 1)
        )

        if N == mpc_horizons[end] && trial == 1
            push!(trajectories[:mpc], traj)
        end

        push!(mpc_results[N][:cost], J)
        mpc_results[N][:t_on] += t_on / num_trials


        # ---------------- MPC + common terminal ----------------

        controller = x -> robust_mpc_terminal_cost(
            x,
            A,
            B,
            Q,
            R,
            P_common;
            horizon = N
        )

        J, t_on, traj = simulate_system(
            controller,
            x0,
            A,
            B,
            Q,
            R,
            switching_sequence;
            save_trajectory = (N == mpc_horizons[end] && trial == 1)
        )

        if N == mpc_horizons[end] && trial == 1
            push!(trajectories[:mpc_terminal_common], traj)
        end

        push!(mpc_terminal_cost_common_results[N][:cost], J)
        mpc_terminal_cost_common_results[N][:t_on] += t_on / num_trials


        # ---------------- MPC + dual De Bruijn terminal ----------------

        if N == mpc_horizons[end]

            controller = x -> robust_mpc_terminal_cost(
                x,
                A,
                B,
                Q,
                R,
                mpc_terminal_cost_results[N][:P];
                horizon = N
            )

            J, t_on, traj = simulate_system(
                controller,
                x0,
                A,
                B,
                Q,
                R,
                switching_sequence;
                save_trajectory = (trial == 1)
            )

            if trial == 1
                push!(trajectories[:mpc_terminal_dualDb], traj)
            end

            push!(mpc_terminal_cost_results[N][:cost], J)
            mpc_terminal_cost_results[N][:t_on] += t_on / num_trials

        end

    end

    # ========================================================
    # Path-complete
    # ========================================================

    for ℓ in 1:order_max

        P = pc_results[ℓ][:P]
        K = pc_results[ℓ][:K]

        controller = x -> begin
            idx = argmin(x' * P[i] * x for i in eachindex(P))
            K[idx] * x
        end

        J, t_on, traj = simulate_system(
            controller,
            x0,
            A,
            B,
            Q,
            R,
            switching_sequence;
            save_trajectory = (ℓ == 1 && trial == 1)
        )

        if ℓ == 1 && trial == 1
            push!(trajectories[:complete], traj)
        end

        push!(pc_results[ℓ][:cost], J)
        pc_results[ℓ][:t_on] += t_on / num_trials

    end

    # ========================================================
    # Co-complete
    # ========================================================

    for ℓ in 1:order_max

        K = pc_co_results[ℓ][:K]

        controller = x -> K*x

        J, t_on, traj = simulate_system(
            controller,
            x0,
            A,
            B,
            Q,
            R,
            switching_sequence;
            save_trajectory = (ℓ == 1 && trial == 1)
        )

        if ℓ == 1 && trial == 1
            push!(trajectories[:cocomplete], traj)
        end

        push!(pc_co_results[ℓ][:cost], J)
        pc_co_results[ℓ][:t_on] += t_on / num_trials

    end

    println("Trial $trial completed.")

end

save_data(all_switching_sequences, "switching_sequences")
save_data(trajectories, "trajectories")

print_path_complete_results(pc_results, "Path-complete (complete graph) results (SDP procedure)")
print_path_complete_results(pc_co_results, "Path-complete (co-complete graph) results (alternate optimization)")
print_mpc_results(
    mpc_results,
    mpc_terminal_cost_common_results,
    mpc_terminal_cost_results,
    mpc_horizons
)