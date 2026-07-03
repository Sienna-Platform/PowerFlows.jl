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
        # Force an LL′ final factor so `issuccess` is a true positive-definiteness check:
        # CHOLMOD's simplicial LDL′ (its default at power-grid sparsity) "succeeds" on
        # indefinite matrices with negative D entries, which would let this loop exit
        # with τ too small and hand the line search a non-descent direction.
        # `final_ll = true` matches stdlib `cholesky!`; with it, minor < n on any non-PD
        # input and `issuccess` returns false. The catch stays as defensive cover for
        # factorization-time throws.
        ok = try
            SparseArrays.CHOLMOD.@cholmod_param final_ll = true begin
                numeric_factor!(hSolver.F, hSolver.mat)
            end
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
