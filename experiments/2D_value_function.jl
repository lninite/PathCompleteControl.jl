"""
Code for generating Figure 5 of the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"

This script compares upper bounds on the value function computed via:
- co-complete graphs (alternating optimization procedure)
- complete graphs (SDP)

The upper bounds are plotted on the unit circle for different orders of the (dual) De Bruijn graph.

The generated figure is saved in the `figures/` directory.
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
# Compute Value Functions Approximations: Co-Complete Graph
################################################################################

Vvals_co_complete = Dict()
P_list_co_complete = []
K_list_co_complete = []

for order in orders
    pc = de_bruijn(
        order,
        length(A),
        mode = :co_complete
    )

    Kopt, Popt = alternate_optimization(
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
# Compute Value Functions Approximations: Complete Graph
################################################################################

Vvals_complete = Dict()
P_list_complete = []
K_list_complete = []

for order in orders
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
# Plotting
################################################################################

colors = [:skyblue, :salmon, :mediumseagreen, :plum]

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

final_plot = plot(legend_plot, p_main, 
                  layout = @layout([A{0.16h}; B]), 
                  size = (920, 620), 
                  left_margin = 18Plots.mm, 
                  right_margin = 12Plots.mm)
display(final_plot)

savefig(final_plot, "figures/UB_on_value_function.pdf")