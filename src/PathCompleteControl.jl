module PathCompleteControl

"""
PathCompleteControl

Code accompanying the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"

Authors: Léa Ninite, Adrien Banse, Guillaume O. Berger, Raphaël M. Jungers
"""

include("utils.jl")
include("complete_graph_policy_synthesis.jl")
include("general_graph_policy_synthesis.jl")
include("mpc.jl")
include("simulate.jl")

using .Utils
using .CompleteGraphPolicySynthesis
using .GeneralGraphPolicySynthesis
using .MPC
using .Simulate

export de_bruijn, save_data, print_path_complete_results, print_mpc_results, pclqr_control, alternate_optimization, robust_mpc, robust_mpc_terminal_cost, simulate_system

const VERSION = v"0.1.0"

version() = VERSION

end # module