module Utils

using SwitchOnSafety
using HybridSystems
using MosekTools
using StaticArrays
using JuMP
using LinearAlgebra
using Serialization
using Dates
using Statistics

export de_bruijn, save_data, print_path_complete_results, print_mpc_results

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

function save_data(data, prefix; folder="data")

    mkpath(folder)

    filename = joinpath(
        folder,
        prefix * "_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS") * ".jls"
    )

    open(filename, "w") do io
        serialize(io, data)
    end

    println("Saved $(prefix) to $filename")

end

function print_path_complete_results(results, title)

    println("\n$title")

    for ℓ in sort(collect(keys(results)))

        J = mean(results[ℓ][:cost])
        σ = std(results[ℓ][:cost])

        println(
            "  - Order = $ℓ: mean_cost = ",
            round(J,digits=6),
            " ± ",
            round(σ,digits=6),
            ", V(x0) = ",
            round(results[ℓ][:V],digits=6),
            ", offline = ",
            round(results[ℓ][:t_off],digits=6),
            " s, online = ",
            round(results[ℓ][:t_on],digits=6),
            " s, total = ",
            round(results[ℓ][:t_off] + results[ℓ][:t_on],digits=6),
            " s"
        )

    end

end

function print_mpc_results(
    mpc_results,
    mpc_terminal_cost_common_results,
    mpc_terminal_cost_results,
    mpc_horizons
)

    println("\n================ MPC RESULTS =================")

    for N in mpc_horizons

        println("\nHorizon = $N")

        # ----------------------------------------------------
        # MPC
        # ----------------------------------------------------

        J = mean(mpc_results[N][:cost])
        σ = std(mpc_results[N][:cost])
        t = mpc_results[N][:t_on]

        println(
            "  - MPC (no terminal cost):           cost = ",
            round(J, digits = 6),
            " ± ",
            round(σ, digits = 6),
            ", online = ",
            round(t, digits = 6),
            " s"
        )

        # ----------------------------------------------------
        # MPC + common terminal cost
        # ----------------------------------------------------

        J = mean(mpc_terminal_cost_common_results[N][:cost])
        σ = std(mpc_terminal_cost_common_results[N][:cost])

        t_on = mpc_terminal_cost_common_results[N][:t_on]
        t_off = mpc_terminal_cost_common_results[:t_off]

        println(
            "  - MPC + terminal cost (common P):   cost = ",
            round(J, digits = 6),
            " ± ",
            round(σ, digits = 6),
            ", offline = ",
            round(t_off, digits = 6),
            " s, online = ",
            round(t_on, digits = 6),
            " s, total = ",
            round(t_off + t_on, digits = 6),
            " s"
        )

        # ----------------------------------------------------
        # MPC + dual De Bruijn terminal cost
        # ----------------------------------------------------

        if haskey(mpc_terminal_cost_results, N)

            if !isempty(mpc_terminal_cost_results[N][:cost])

                J = mean(mpc_terminal_cost_results[N][:cost])
                σ = std(mpc_terminal_cost_results[N][:cost])

                t_on = mpc_terminal_cost_results[N][:t_on]
                t_off = mpc_terminal_cost_results[N][:t_off]

                println(
                    "  - MPC + terminal cost (order 1 dual De Bruijn):    cost = ",
                    round(J, digits = 6),
                    " ± ",
                    round(σ, digits = 6),
                    ", offline = ",
                    round(t_off, digits = 6),
                    " s, online = ",
                    round(t_on, digits = 6),
                    " s, total = ",
                    round(t_off + t_on, digits = 6),
                    " s"
                )

            end

        end

    end

end

end
