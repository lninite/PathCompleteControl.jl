module PathCompleteControl

"""
    PathCompleteControl

Code accompanying the paper:
"Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework"

Authors: Léa Ninite, Adrien Banse, Guillaume O. Berger, Raphaël M. Jungers
"""

include("utils.jl")
include("bounds.jl")
include("mpc.jl")
include("alternate_opt.jl")

using .Utils
using .Bounds
using .MPC
using .AlternateOpt

export de_bruijn, feedback_tree, white_box_JSR, common_Lyap, largest_gamma_bisection
export pclqr_control
export robust_mpc, robust_mpc_terminal_cost
export alternate_optimization

const VERSION = v"0.1.0"

version() = VERSION

end # module
