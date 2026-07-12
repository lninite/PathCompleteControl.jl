module Simulate

export simulate_system

function simulate_system(
    controller,
    x0,
    A,
    B,
    Q,
    R,
    switching_sequence;
    save_trajectory = false
)

    x = x0
    J = 0.0

    traj = save_trajectory ? [x] : nothing

    t_on = @elapsed begin

        for mode in switching_sequence

            u = controller(x)

            J += x'Q*x + u'R*u

            x = A[mode] * x + B[mode] * u

            if save_trajectory
                push!(traj, x)
            end

        end

    end

    return J, t_on, traj

end

end # module