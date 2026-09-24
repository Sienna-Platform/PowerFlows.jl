"""In order to in-place modify the numeric values of a CHOLMOD matrix, we need to 
write our own wrapper around `CHOLMOD.Sparse`. """
struct FixedStructureCHOLMOD{T, I} <: SparseArrays.AbstractSparseMatrix{T, I}
    _mat::SparseArrays.CHOLMOD.Sparse{T, I}
    _values::Vector{T}
end

Base.show(mat::FixedStructureCHOLMOD) =
    print("FixedStructureCHOLMOD with $(size(mat)) and $(length(mat._values)) values")

function FixedStructureCHOLMOD(A::SparseMatrixCSC{T, I}) where {T <: VTypes, I <: ITypes}
    cholmod_mat = SparseArrays.CHOLMOD.Sparse(LinearAlgebra.Symmetric(A))
    SparseArrays.CHOLMOD.check_sparse(cholmod_mat) ||
        throw(ArgumentError("Invalid CHOLMOD sparse matrix"))
    chol = unsafe_load(SparseArrays.CHOLMOD.typedpointer(cholmod_mat))
    Bool(chol.packed) || throw(ArgumentError("Unpacked CHOLMOD matrices not supported"))
    # max capacity may not be the same as the number of nonzeros:
    # instead of chol.nzmax, we want chol.p[chol.ncol]
    nonzero_count = unsafe_load(Ptr{I}(chol.p), chol.ncol + 1)
    # know chol.x != C_NULL because of check_sparse above.
    values = unsafe_wrap(Vector{T}, Ptr{T}(chol.x), nonzero_count; own = false)
    return FixedStructureCHOLMOD(cholmod_mat, values)
end

"""
    set_values!(mat::FixedStructureCHOLMOD, new_vals::AbstractVector{Float64})
In-place update of the numeric values in the CHOLMOD matrix.
"""
function set_values!(
    mat::FixedStructureCHOLMOD{T, I},
    new_vals::AbstractVector{T},
) where {T <: VTypes, I <: ITypes}
    length(new_vals) == length(mat._values) ||
        throw(ArgumentError("Allocated storage of size "))
    copyto!(mat._values, new_vals)
    return mat
end

function symbolic_factor(mat::FixedStructureCHOLMOD{T, I}) where {T <: VTypes, I <: ITypes}
    # I'm really feeding in a CHOLMOD_PATTERN object here, ie xtype = 0, but Julia's
    # interface doesn't seem to support that.
    return SparseArrays.CHOLMOD.symbolic(mat._mat)
end

function numeric_factor!(
    F::SparseArrays.CHOLMOD.Factor{T, I},
    mat::FixedStructureCHOLMOD{T, I},
) where {T <: VTypes, I <: ITypes}
    SparseArrays.CHOLMOD.factorize!(mat._mat, F)
    return
end

Base.size(mat::FixedStructureCHOLMOD) = size(mat._mat)
Base.getindex(mat::FixedStructureCHOLMOD, I...) = getindex(mat._mat, I...)
Base.setindex!(mat::FixedStructureCHOLMOD, v, I...) = setindex!(mat._mat, v, I...)
