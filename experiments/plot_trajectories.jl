"""
Code for generating Figure 7 of the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"

This script loads the most recent trajectory file saved in the `data/`
directory and reproduces the comparison of closed-loop trajectories between:

- Path-complete control with a complete graph 
- Path-complete control with a co-complete graph
- Robust MPC without terminal cost
- Robust MPC with terminal cost

The generated figure is saved in the `figures/` directory.

The trajectory file must have been generated beforehand by running
`temperature_regulation.jl`.
"""

using Serialization
using Dates
using Plots

function plot_saved_trajectories(;
    trial::Int = 1,
    datapath::AbstractString = "data",
    savepath::AbstractString = joinpath("figures", "comparison_trajectories.pdf"),
    tau::Real = 20.0
)

    mkpath(dirname(savepath))

    # ------------------------------------------------------------
    # Load latest trajectory file
    # ------------------------------------------------------------

    files = filter(f ->
        startswith(f, "trajectories_") &&
        endswith(f, ".jls"),
        readdir(datapath)
    )

    @assert !isempty(files) "No trajectories_*.jls file found in $datapath."

    sort!(files)

    filename = joinpath(datapath, files[end])

    println("Loading trajectories from:")
    println(filename)

    trajectories = open(filename) do io
        deserialize(io)
    end

    # ------------------------------------------------------------
    # Convert Vector{SVector} -> Matrix
    # ------------------------------------------------------------

    traj2mat(traj) = reduce(hcat, traj)

    trajs = Dict{String, Matrix{Float64}}()

    trajs["Complete"] =
        traj2mat(trajectories[:complete][trial])

    trajs["CoComplete"] =
        traj2mat(trajectories[:cocomplete][trial])

    trajs["MPC"] =
        traj2mat(trajectories[:mpc][trial])

    trajs["MPC_terminal"] =
        traj2mat(trajectories[:mpc_terminal_dualDb][trial])

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------

    colors = [
        colorant"midnightblue",
        colorant"seagreen",
        colorant"deepskyblue3",
        colorant"darkorange2"
    ]

    # ------------------------------------------------------------
    # Subplots
    # ------------------------------------------------------------

    p1 = Plots.plot(
        title = "Zone 1",
        xlabel = "",
        ylabel = "Temperature [°C]",
        legend = false,
        titlefontsize = 16,
        xaxisfontsize = 16,
        yaxisfontsize = 16,
        xtickfont = Plots.font(12),
        ytickfont = Plots.font(12),
        labelfontsize = 16
    )

    p2 = Plots.plot(
        title = "Zone 2",
        xlabel = "",
        ylabel = "Temperature [°C]",
        legend = false,
        titlefontsize = 16,
        xaxisfontsize = 16,
        yaxisfontsize = 16,
        xtickfont = Plots.font(12),
        ytickfont = Plots.font(12),
        labelfontsize = 16
    )

    p3 = Plots.plot(
        title = "Zone 3",
        xlabel = "Time [min]",
        ylabel = "Temperature [°C]",
        legend = false,
        titlefontsize = 16,
        xaxisfontsize = 16,
        yaxisfontsize = 16,
        xtickfont = Plots.font(12),
        ytickfont = Plots.font(12),
        labelfontsize = 16
    )

    names = [
        "Complete",
        "CoComplete",
        "MPC",
        "MPC_terminal"
    ]

    # ------------------------------------------------------------
    # Plot trajectories
    # ------------------------------------------------------------

    for (i, name) in enumerate(names)

        series = trajs[name]

        zone_1 = series[1, :] .+ 24
        zone_2 = series[2, :] .+ 24
        zone_3 = series[3, :] .+ 24

        t = (0:size(series, 2)-1) .* tau ./ 60

        Plots.plot!(
            p1,
            t,
            zone_1,
            color = colors[i],
            linewidth = i == 3 ? 2.5 : 2,
            label = ""
        )

        Plots.plot!(
            p2,
            t,
            zone_2,
            color = colors[i],
            linewidth = i == 3 ? 2.5 : 2,
            label = ""
        )

        Plots.plot!(
            p3,
            t,
            zone_3,
            color = colors[i],
            linewidth = i == 3 ? 2.5 : 2,
            label = ""
        )

    end

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    for p in (p1, p2, p3)

        Plots.hline!(
            p,
            [24],
            color = :black,
            linestyle = :dash,
            linewidth = 1.5,
            label = ""
        )

        Plots.ylims!(p, (18, 30))

    end

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    legend_plot = Plots.plot(
        foreground_color_legend = :grey50,
        background_color_legend = :white
    )

    labels = [
        "Complete",
        "Co-complete",
        "MPC",
        "MPC + terminal cost"
    ]

    widths = [
        2.0,
        2.0,
        2.5,
        2.0
    ]

    for (lbl, col, lw) in zip(labels, colors, widths)

        Plots.plot!(
            legend_plot,
            [NaN],
            [NaN],
            color = col,
            linewidth = lw,
            label = lbl,
            legendfontsize = 14
        )

    end

    Plots.plot!(
        legend_plot,
        [NaN],
        [NaN],
        color = :black,
        linestyle = :dash,
        linewidth = 1,
        label = "Reference",
        legendfontsize = 14
    )

    Plots.plot!(
        legend_plot,
        framestyle = :none,
        legend = :top,
        legendcolumns = 3,
        grid = false,
        xticks = false,
        yticks = false
    )

    # ------------------------------------------------------------
    # Final figure
    # ------------------------------------------------------------

    final_plot = Plots.plot(
        legend_plot,
        p1,
        p2,
        p3,
        layout = @layout([A{0.12h}; B; C; D]),
        size = (850, 1200),
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm,
        top_margin = 5Plots.mm
    )

    Plots.savefig(final_plot, savepath)

    display(final_plot)

    println("Saved figure to $savepath")

    return savepath

end

plot_saved_trajectories()