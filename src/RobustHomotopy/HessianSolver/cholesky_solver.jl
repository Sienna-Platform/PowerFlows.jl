struct CholeskyHessianSolver <: HessianSolver
    F::SparseArrays.CHOLMOD.Factor{Float64, J_INDEX_TYPE}
    mat::FixedStructureCHOLMOD{Float64, J_INDEX_TYPE}
    buff::Vector{Float64} # buffer for solving
end

function CholeskyHessianSolver(H::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    mat = FixedStructureCHOLMOD(H)
    # I need to create a CHOLMOD factorization object, so I also symbolic factor it here.
    n = size(H, 1)
    return CholeskyHessianSolver(symbolic_factor(mat), mat, zeros(n))
end

function symbolic_factor!(::CholeskyHessianSolver, ::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    # hSolver.F = symbolic_factor(hSolver.mat) # if I make FixedStructureCHOLMOD mutable.
    return
end

function modify_and_numeric_factor!(
    hSolver::CholeskyHessianSolver,
    H::SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    minDiagElem = minimum(H[i, i] for i in axes(H, 1))
    τ_old = 0.0
    τ = minDiagElem > 0.0 ? 0.0 : -minDiagElem + β
    @debug "initial τ = $τ"
    nonsingular = false
    while !nonsingular
        for i in axes(H, 1)
            H[i, i] += τ - τ_old # now try H + τ*I
        end
        set_values!(hSolver.mat, SparseArrays.nonzeros(H))
        # `issuccess` checks PD directly, replacing the throwaway `F \ ones` solve.
        # `numeric_factor!` can still throw PosDefException, so keep the catch.
        ok = try
            numeric_factor!(hSolver.F, hSolver.mat)
            LinearAlgebra.issuccess(hSolver.F)
        catch e
            e isa SparseArrays.CHOLMOD.PosDefException ? false : rethrow(e)
        end
        if ok
            nonsingular = true
            @debug "nonsingular with τ = $τ"
        else
            τ_old = τ
            τ *= 2.0
            τ = max(τ, β) # ensure τ is at least β
        end
    end
    return
end

function solve!(solver::CholeskyHessianSolver, b::Vector{Float64})
    copyto!(solver.buff, b)
    b .= solver.F \ solver.buff
end
