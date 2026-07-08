"""
    figure_5_complete_vs_cocomplete.jl

Experiment code for Figure 5 of the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"

This script compares the value functions computed via:
- co-complete path-complete LQR with alternate optimization
- complete path-complete LQR (from bounds.jl)

The results are plotted as value functions V^ν on the unit circle.
"""

using PathCompleteControl
using StaticArrays
using HybridSystems
using LinearAlgebra
using Statistics
using Plots

################################################################################
# System Definition
################################################################################

A1 = SMatrix{2,2}([0.0 1.0;
                   -1.0 0.0])

A2 = SMatrix{2,2}([-0.1 0.0;
                    0.0 -0.95])

B1 = SMatrix{2,1}([1.0;
                   0.0])

A = [A1, A2]
B = [B1, B1]

Q = SMatrix{2, 2}([1. 0; 0 1.])
R = SMatrix{1, 1}([1.])

################################################################################
# Parameter Setup
################################################################################

theta = LinRange(0, π, 500)
orders = 1:4

################################################################################
# Compute Value Functions: Co-Complete Mode
################################################################################

Vvals_co_complete = Dict()
P_list_co_complete = []
K_list_co_complete = []

for order in orders
    println("\n================= CO-COMPLETE ORDER $order =================")

    pc = de_bruijn(
        order,
        length(A),
        mode = :co_complete
    )

    Kopt, Popt, γopt = alternate_optimization(
        A,
        B,
        Q,
        R,
        pc;
        N = 50
    )

    V(x) = maximum(x' * P * x for P in Popt)
    Vvals = [V([cos(t), sin(t)]) for t in theta]
    Vvals_co_complete[order] = Vvals

    push!(P_list_co_complete, Popt)
    push!(K_list_co_complete, Kopt)
end

################################################################################
# Compute Value Functions: Complete Mode
################################################################################

Vvals_complete = Dict()
P_list_complete = []
K_list_complete = []

for order in orders
    println("\n================= COMPLETE ORDER $order =================")
    
    pc = de_bruijn(
        order,
        length(A),
        mode = :complete
    )

    P, K = pclqr_control(
        A,
        B,
        Q,
        R,
        pc
    )

    V(x) = minimum(x' * P * x for P in P)
    Vvals = [V([cos(t), sin(t)]) for t in theta]
    Vvals_complete[order] = Vvals

    push!(P_list_complete, P)
    push!(K_list_complete, K)
end

################################################################################
# Trajectory Simulation and Finite Horizon Cost Computation
################################################################################

H = 500
n_traj = 1
switching_sequences = [rand(1:length(A), H) for _ in 1:n_traj]

costs_co_complete = [Float64[] for _ in orders]
x0 = [cos(1.5), sin(1.5)]

for seq in switching_sequences
    for order in orders
        K = K_list_co_complete[order]
        cost = 0.0
        x = x0
        for t in 1:H
            mode = seq[t]
            u = K * x
            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end
        push!(costs_co_complete[order], cost)
    end
end

costs_complete = [Float64[] for _ in orders]

for seq in switching_sequences
    for order in orders
        K = K_list_complete[order]
        cost = 0.0
        x = x0
        for t in 1:H
            mode = seq[t]
            idx_control = argmin(x' * P_list_complete[order][i] * x for i in 1:length(P_list_complete[order]))
            u = K_list_complete[order][idx_control] * x
            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end
        push!(costs_complete[order], cost)
    end
end

for order in orders
    println("Order $order: Average cost (co-complete) = ", mean(costs_co_complete[order]))
    println("Order $order: Average cost (complete) = ", mean(costs_complete[order]))
end

################################################################################
# Plotting
################################################################################

colors = [:skyblue, :salmon, :mediumseagreen, :plum]

# Main value function plot
p_main = plot(
    xlabel = "\$\\theta\$ [rad]",
    ylabel = "\$V(\\cos\\theta,\\sin\\theta)\$",
    legend = false,
    labelfontsize = 16,
    legendfontsize = 16,
    xguidefont = Plots.font(16),
    yguidefont = Plots.font(16),
    tickfont = Plots.font(14),
    foreground_color_legend = :grey50,
    background_color_legend = :white,
    left_margin = 18Plots.mm,
    right_margin = 12Plots.mm,
    bottom_margin = 8Plots.mm,
    top_margin = 6Plots.mm
)

for order in orders
    plot!(p_main, theta, Vvals_co_complete[order], label = "", color = colors[order], linewidth = 2.5)
    plot!(p_main, theta, Vvals_complete[order], label = "", linestyle = :dash, color = colors[order], linewidth = 1.8)
end

# Create legend
legend_plot = plot(
    foreground_color_legend = :grey50, 
    background_color_legend = :white, 
    legendfontsize = 16, 
    labelfontsize = 16
)
for order in orders
    Plots.plot!(legend_plot, [NaN], [NaN], color = colors[order], linewidth = 1.6, 
                label = "\$V^{" * string(order) * "}_{cc}\$")
    Plots.plot!(legend_plot, [NaN], [NaN], color = colors[order], linewidth = 1.2, linestyle = :dash, 
                label = "\$V^{" * string(order) * "}_{c}\$")
end

Plots.plot!(legend_plot, framestyle = :none, legend = :top, legendcolumns = length(orders)*2, 
            grid = false, xticks = false, yticks = false)

# Combine plots
final_plot = plot(legend_plot, p_main, 
                  layout = @layout([A{0.16h}; B]), 
                  size = (920, 620), 
                  left_margin = 18Plots.mm, 
                  right_margin = 12Plots.mm)

mkpath("../figures")
savefig(final_plot, joinpath("../figures", "figure_5_complete_vs_cocomplete.pdf"))
display(final_plot)
