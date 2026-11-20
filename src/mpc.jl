module MPC

using LinearAlgebra
using StaticArrays
using JuMP
using MosekTools

export robust_mpc, robust_mpc_terminal_cost, TreeMPC

function robust_mpc(
    x0::SVector{Nx,T},
    A::AbstractVector{<:SMatrix{Nx,Nx,T,Nxx}},
    B::AbstractVector{<:SMatrix{Nx,Nu,T,Nxu}},
    Q::SMatrix{Nx,Nx,T,Nxx},
    R::SMatrix{Nu,Nu,T,Nuu};
    horizon::Int = 4
) where {Nx,Nu,T,Nxx,Nxu,Nuu}

    num_modes = length(A)
    nu = size(B[1], 2)

    switching_sequences = collect(
        Iterators.product(
            [1:num_modes for _ in 1:horizon]...
        )
    )

    model = Model(Mosek.Optimizer)
    set_silent(model)

    U = Dict{Tuple{Vararg{Int}}, Vector{VariableRef}}()

    for t in 0:horizon-1
        prefixes = t == 0 ? [()] : collect(Iterators.product([1:num_modes for _ in 1:t]...))
        for prefix in prefixes
            U[prefix] = [@variable(model) for _ in 1:nu]
        end
    end

    @variable(model, γ)

    for σ in switching_sequences
        x = x0
        cost = @expression(model, 0.0)

        for t in 0:horizon-1
            mode = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end

        @constraint(model, cost <= γ)
    end

    @objective(model, Min, γ)
    optimize!(model)

    return collect(Float64.(value.(U[()])))
end

function robust_mpc_terminal_cost(
    x0::SVector{Nx,T},
    A::AbstractVector{<:SMatrix{Nx,Nx,T,Nxx}},
    B::AbstractVector{<:SMatrix{Nx,Nu,T,Nxu}},
    Q::SMatrix{Nx,Nx,T,Nxx},
    R::SMatrix{Nu,Nu,T,Nuu},
    P::AbstractVector{<:AbstractMatrix{T}};
    horizon::Int = 4
) where {Nx,Nu,T,Nxx,Nxu,Nuu}

    num_modes = length(A)
    nu = size(B[1], 2)

    switching_sequences = collect(
        Iterators.product(
            [1:num_modes for _ in 1:horizon]...
        )
    )

    model = Model(Mosek.Optimizer)
    set_silent(model)

    U = Dict{Tuple{Vararg{Int}}, Vector{VariableRef}}()
    t_var = Dict{Tuple{Vararg{Int}}, VariableRef}()

    for t in 0:horizon-1
        prefixes = t == 0 ? [()] : collect(Iterators.product([1:num_modes for _ in 1:t]...))
        for prefix in prefixes
            U[prefix] = [@variable(model) for _ in 1:nu]
        end
    end

    for σ in switching_sequences
        t_var[Tuple(σ)] = @variable(model)
    end

    @variable(model, γ)

    for σ in switching_sequences
        x = x0
        cost = @expression(model, 0.0)

        for t in 0:horizon-1
            mode = σ[t+1]
            prefix = t == 0 ? () : Tuple(σ[1:t])
            u = U[prefix]

            cost += x' * Q * x + u' * R * u
            x = A[mode] * x + B[mode] * u
        end

        cost += t_var[Tuple(σ)]
        @constraint(model, cost <= γ)

        for i in 1:length(P)
            @constraint(model, [t_var[Tuple(σ)] x'; x inv(P[i])] in PSDCone())
        end
    end

    @objective(model, Min, γ)
    optimize!(model)

    return collect(Float64.(value.(U[()])))
end

"""
    TreeMPC(A, B, Q, R; horizon, terminal_cost = nothing, normalize = true)

Faster implementation of the two controllers above: same control as `robust_mpc`
(`terminal_cost = nothing`) and as `robust_mpc_terminal_cost` (`terminal_cost = P`).
The controller is built once and then called at each step:

    mpc = TreeMPC(A, B, Q, R; horizon = 5, terminal_cost = P)
    u = mpc(x)

Both functions above choose one control per node of the scenario tree (a node is
a sequence of past modes) and minimise the worst-case cost over the `M^N`
switching sequences `σ`:

    min_u max_σ  Σ_{t<N} (x_t'Qx_t + u_t'Ru_t) + max_i x_N'P_i x_N.

They write one constraint per sequence. `TreeMPC` writes the same problem node by
node, where `c(ν, j)` is the child of node `ν` reached with mode `j`:

    min  w_root
    s.t. x_c(ν,j) = A_j x_ν + B_j u_ν               the node states are variables
         s_ν ≥ x_ν'Qx_ν + u_ν'Ru_ν                  stage cost (rotated second-order cone)
         w_ν ≥ s_ν + w_c(ν,j)       for all j       worst-case cost-to-go (linear)
         w_leaf ≥ x_leaf'P_i x_leaf for all i       terminal cost (rotated second-order cone)

For given controls, the smallest feasible `w_root` is the worst-case cost over
all sequences, minus the constant `x0'Qx0`. The two problems therefore have the
same minimisers, and the first control is unique because `R ≻ 0`.

Only the right-hand side `A_j x0` of the first-level dynamics depends on the
current state, so a call updates it and re-solves. With `normalize = true` the
problem is solved for `x/‖x‖` and the control is scaled back by `‖x‖`. This gives
the same control, since `u*(λx) = λ u*(x)` for `λ > 0`, and keeps the solver
tolerances relative to `‖x‖`.

`Q`, `R` and the `P_i` must be positive definite.
"""
struct TreeMPC
    model::Model
    u::Dict{Int,Matrix{VariableRef}}       # u[d][:, k] is the control of node k of depth d
    root_dynamics::Matrix{ConstraintRef}   # x_c(root,j) - B_j u_root == A_j x0, one column per mode j
    A::Vector{Matrix{Float64}}
    normalize::Bool
end

function TreeMPC(A, B, Q, R; horizon::Int, terminal_cost = nothing, normalize::Bool = true)

    N = horizon
    M = length(A)
    n, m = size(B[1])
    A = [Matrix{Float64}(a) for a in A]
    B = [Matrix{Float64}(b) for b in B]

    # S = L'L, so x'Sx = ‖Lx‖²
    factor(S) = Matrix(cholesky(Symmetric(Matrix{Float64}(S))).U)
    Lq = factor(Q)
    Lr = factor(R)
    has_terminal = terminal_cost !== nothing
    Lp = has_terminal ? [factor(P) for P in terminal_cost] : nothing

    model = direct_model(Mosek.Optimizer())
    set_silent(model)
    set_attribute(model, "MSK_IPAR_NUM_THREADS", 1)

    # Node k = 1, ..., M^d of depth d has children (k - 1)M + 1, ..., kM at depth d + 1.
    # Without terminal cost the states of depth N have no cost and are left out.
    child(k, j) = (k - 1) * M + j
    deepest = has_terminal ? N : N - 1

    x = Dict(d => @variable(model, [1:n, 1:M^d]) for d in 1:deepest)  # states (the root state x0 is data)
    u = Dict(d => @variable(model, [1:m, 1:M^d]) for d in 0:N-1)      # controls
    s = Dict(d => @variable(model, [1:M^d]) for d in 0:N-1)           # stage costs
    w = Dict(d => @variable(model, [1:M^d]) for d in 0:deepest)       # worst-case costs-to-go

    # Stage costs: s ≥ x'Qx + u'Ru  ⟺  2 * s * 1/2 ≥ ‖Lq x‖² + ‖Lr u‖².
    # At the root, x0'Qx0 is a constant and is left out.
    @constraint(model, [s[0][1]; 0.5; Lr * u[0][:, 1]] in RotatedSecondOrderCone())
    for d in 1:N-1, k in 1:M^d
        @constraint(model, [s[d][k]; 0.5; Lq * x[d][:, k]; Lr * u[d][:, k]] in RotatedSecondOrderCone())
    end

    # Dynamics. At the root the state is data: x_c(root,j) - B_j u_root == A_j x0,
    # and the right-hand side is set at each call.
    root_dynamics = Matrix{ConstraintRef}(undef, n, deepest >= 1 ? M : 0)
    for j in axes(root_dynamics, 2)
        root_dynamics[:, j] = @constraint(model, x[1][:, j] .- B[j] * u[0][:, 1] .== zeros(n))
    end
    for d in 1:deepest-1, k in 1:M^d, j in 1:M
        @constraint(model, x[d+1][:, child(k, j)] .== A[j] * x[d][:, k] .+ B[j] * u[d][:, k])
    end

    # Worst-case cost-to-go: w ≥ s + w_child for every child.
    for d in 0:N-1, k in 1:M^d
        if d == deepest
            @constraint(model, w[d][k] >= s[d][k])
        else
            for j in 1:M
                @constraint(model, w[d][k] >= s[d][k] + w[d+1][child(k, j)])
            end
        end
    end

    # Terminal cost: w_leaf ≥ x'P_i x for every i.
    if has_terminal
        for k in 1:M^N, L in Lp
            @constraint(model, [w[N][k]; 0.5; L * x[N][:, k]] in RotatedSecondOrderCone())
        end
    end

    @objective(model, Min, 1.0 * w[0][1])

    return TreeMPC(model, u, root_dynamics, A, normalize)
end

function (mpc::TreeMPC)(x0)

    scale = mpc.normalize ? norm(x0) : 1.0
    scale == 0 && return zeros(size(mpc.u[0], 1))

    for j in axes(mpc.root_dynamics, 2)
        set_normalized_rhs.(mpc.root_dynamics[:, j], mpc.A[j] * (x0 / scale))
    end

    optimize!(mpc.model)

    return scale .* value.(mpc.u[0][:, 1])
end

end # module
