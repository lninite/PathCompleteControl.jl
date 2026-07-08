using Test
using PathCompleteControl
using LinearAlgebra
using StaticArrays

@testset "PathCompleteControl.jl" begin

    @testset "de_bruijn" begin
        automaton = de_bruijn(2, 3)
        @test !isnothing(automaton)
        println("✓ de_bruijn(2, 3) creates automaton")
    end

    @testset "feedback_tree" begin
        automaton = feedback_tree(2, 3)
        @test !isnothing(automaton)
        println("✓ feedback_tree(2, 3) creates automaton")
    end

    @testset "common_Lyap" begin
        # Simple stable system
        A = [SMatrix{2,2}([0.5 0.0; 0.0 0.5]), SMatrix{2,2}([0.3 0.1; 0.1 0.3])]
        P = common_Lyap(A, 1.0)
        @test !isnothing(P)
        @test size(P) == (2, 2)
        @test all(eigvals(P) .> -1e-6)  # P should be PSD
        println("✓ common_Lyap returns PSD matrix")
    end

    @testset "largest_gamma_bisection" begin
        A = [SMatrix{2,2}([0.5 0.0; 0.0 0.5]), SMatrix{2,2}([0.3 0.1; 0.1 0.3])]
        gamma = largest_gamma_bisection(A, gamma_lower=0.0, gamma_upper=2.0, tol=1e-3)
        @test !isnothing(gamma)
        @test gamma > 0.0
        println("✓ largest_gamma_bisection returns positive gamma")
    end

end
