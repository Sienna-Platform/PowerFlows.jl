
struct KLUHessianSolver <: HessianSolver
    linearSolver::PNM.KLULinSolveCache{Float64, J_INDEX_TYPE}
end

function KLUHessianSolver(H::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    linearSolver = PNM.KLULinSolveCache(H)
    return KLUHessianSolver(linearSolver)
end

function symbolic_factor!(
    hSolver::KLUHessianSolver,
    H::SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    symbolic_factor!(hSolver.linearSolver, H)
    return
end

function modify_and_numeric_factor!(
    hSolver::KLUHessianSolver,
    H::SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    minDiagElem = minimum(H[i, i] for i in axes(H, 1))
    τ_old = 0.0
    if minDiagElem > 0.0
        τ = 0.0
    else
        τ = -minDiagElem + β
    end
    @debug "initial τ = $τ"
    nonsingular = false
    while !nonsingular
        for i in axes(H, 1)
            H[i, i] += τ - τ_old # now try H + τ*I
        end
        try
            numeric_refactor!(hSolver.linearSolver, H)
            nonsingular = true
            @debug "nonsingular with τ = $τ"
        catch e
            if e isa LinearAlgebra.SingularException
                τ_old = τ
                τ = max(2 * τ, β)
                nonsingular = false
            else
                rethrow(e)
            end
        end
    end
    return
end

solve!(hSolver::KLUHessianSolver, b::Vector{Float64}) = solve!(hSolver.linearSolver, b)
