module operators

import Base: trace, ==, +, -, *, /, ishermitian
import ..bases: tensor, ptrace, permutesystems
import ..states: dagger, normalize, normalize!

using ..sortedindices, ..bases, ..states

export Operator,
       dagger, identityoperator,
       trace, ptrace, normalize, normalize!, expect, variance,
       tensor, permutesystems, embed


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
basis_l and basis_r defining the left and right side bases.

For fast time evolution also at least the function
gemv!(alpha, op::Operator, x::Ket, beta, result::Ket) should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract Operator

# Common error messages
arithmetic_unary_error(funcname, x::Operator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. full() or sparse()."))
arithmetic_binary_error(funcname, a::Operator, b::Operator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. full() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))


# Arithmetic operations
*(a::Operator, b::Operator) = arithmetic_binary_error("Multiplication", a, b)

+(a::Operator, b::Operator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::Operator) = addnumbererror()
+(a::Operator, b::Number) = addnumbererror()

-(a::Operator) = arithmetic_unary_error("Negation", a)
-(a::Operator, b::Operator) = arithmetic_binary_error("Subtraction", a, b)
-(a::Number, b::Operator) = addnumbererror()
-(a::Operator, b::Number) = addnumbererror()

bases.basis(a::Operator) = (check_samebases(a); a.basis_l)

"""
Hermitian conjugate of the given operator.
"""
dagger(a::Operator) = arithmetic_unary_error("Hermitian conjugate", a)

"""
Identity operator.
"""
identityoperator{T<:Operator}(::Type{T}, b1::Basis, b2::Basis) = throw(ArgumentError("Identity operator not defined for operator type $T."))
identityoperator{T<:Operator}(::Type{T}, b::Basis) = identityoperator(T, b, b)
identityoperator{T<:Operator}(op::T) = identityoperator(T, op.basis_l, op.basis_r)

Base.one(b::Basis) = identityoperator(b)
Base.one(op::Operator) = identityoperator(op)

"""
Trace of given operator.
"""
trace(x::Operator) = arithmetic_unary_error("Trace", x)

"""
Partial trace of the given operator over the specified indices.
"""
ptrace(a::Operator, index::Vector{Int}) = arithmetic_unary_error("Partial trace", a)

"""
Normalized copy of given operator (trace is 1.).
"""
normalize(op::Operator) = op/trace(op)

"""
Normalize the given operator in-place (trace is 1.).
"""
normalize!(op::Operator) = throw(ArgumentError("normalize! is not defined for this type of operator: $(typeof(op)).\n You may have to fall back to the non-inplace version 'normalize()'."))

"""
Expectation value of the given operator for the specified state(s).
"""
expect(op::Operator, state::Ket) = dagger(state)*(op*state)
expect(op::Operator, state::Operator) = trace(op*state)
function expect(indices::Vector{Int}, op::Operator, state::Operator)
    N = length(state.basis_l.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
function expect(indices::Vector{Int}, op::Operator, state::Ket)
    N = length(state.basis.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
expect(index::Int, op::Operator, state) = expect([index], op, state)
expect(op::Operator, states::Vector) = [expect(op, state) for state=states]
expect(indices::Vector{Int}, op::Operator, states::Vector) = [expect(indices, op, state) for state=states]

"""
Variance of the given operator for the specified state(s).
"""
function variance(op::Operator, state::Ket)
    x = op*state
    stateT = dagger(state)
    stateT*op*x - (stateT*x)^2
end
function variance(op::Operator, state::Operator)
    expect(op*op, state) - expect(op, state)^2
end
variance(op::Operator, states::Vector) = [variance(op, state) for state=states]

"""
Tensor product of operators.
"""
tensor(a::Operator, b::Operator) = arithmetic_binary_error("Tensor product", a, b)

"""
Change the ordering of the subsystems of the given operator.

Arguments
---------
a
    An operator represented in a composite basis.
perm
    Vector defining the new ordering of the subsystems.
"""
permutesystems(a::Operator, perm::Vector{Int}) = arithmetic_unary_error("Permutations of subsystems", a)


"""
Vector of indices that are not in the given vector.
"""
complement(N::Int, indices) = Int[i for i=1:N if i ∉ indices]

"""
Tensor product of operators where missing indices are filled up with identity operators.

Arguments
---------
basis_l
    Left hand side basis of the resulting operator.
basis_r
    Right hand side basis of the resulting operator.
indices
    Indices of the subsystems in which the given operators live.
operators
    Operators defined in the subsystems.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            indices::Vector{Int}, operators::Vector{T})
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N
    @assert length(indices) == length(operators)
    sortedindices.check_indices(N, indices)
    tensor([i ∈ indices ? operators[findfirst(indices, i)] : identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
end
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Int, op::Operator) = embed(basis_l, basis_r, Int[index], [op])
embed(basis::CompositeBasis, index::Int, op::Operator) = embed(basis, basis, Int[index], [op])
embed{T<:Operator}(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{T}) = embed(basis, basis, indices, operators)

"""
Tensor product of operators where all missing indices are identity operators.

Arguments
---------
basis_l
    Left hand side basis of the resulting operator.
basis_r
    Right hand side basis of the resulting operator.
operators
    Dictionary specifying to which subsystems the corresponding operator
    belongs.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            operators::Dict{Vector{Int}, T})
    @assert length(basis_l.bases) == length(basis_r.bases)
    N = length(basis_l.bases)
    if length(operators) == 0
        return identityoperator(T, basis_l, basis_r)
    end
    indices, operator_list = zip(operators...)
    operator_list = [operator_list...;]
    indices_flat = [indices...;]
    start_indices_flat = [i[1] for i in indices]
    complement_indices_flat = complement(N, indices_flat)
    operators_flat = T[]
    if all([minimum(I):maximum(I);]==I for I in indices)
        for i in 1:N
            if i in complement_indices_flat
                push!(operators_flat, identityoperator(T, basis_l.bases[i], basis_r.bases[i]))
            elseif i in start_indices_flat
                push!(operators_flat, operator_list[findfirst(start_indices_flat, i)])
            end
        end
        return tensor(operators_flat...)
    else
        complement_operators = [identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i in complement_indices_flat]
        op = tensor([operator_list; complement_operators]...)
        perm = sortperm([indices_flat; complement_indices_flat])
        return permutesystems(op, perm)
    end
end
embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis_l, basis_r, Dict([i]=>op_i for (i, op_i) in operators); kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis, basis, operators; kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Vector{Int}, T}; kwargs...) = embed(basis, basis, operators; kwargs...)

"""
Fast multiplication of operators with states.
"""
gemv!() = error("Not Implemented.")

"""
Fast multiplication of operators with DenseOperators.
"""
gemm!() = error("Not Implemented.")


# Helper functions to check validity of arguments
function check_ptrace_arguments(a::Operator, indices::Vector{Int})
    @assert length(a.basis_l.shape) == length(a.basis_r.shape)
    sortedindices.check_indices(length(a.basis_l.shape), indices)
end

bases.samebases(a::Operator) = samebases(a.basis_l, a.basis_r)
bases.samebases(a::Operator, b::Operator) = samebases(a.basis_l, b.basis_l) && samebases(a.basis_r, b.basis_r)
bases.check_samebases(a::Operator) = check_samebases(a.basis_l, a.basis_r)

bases.multiplicable(a::Operator, b::Ket) = multiplicable(a.basis_r, b.basis)
bases.multiplicable(a::Bra, b::Operator) = multiplicable(a.basis, b.basis_l)
bases.multiplicable(a::Operator, b::Operator) = multiplicable(a.basis_r, b.basis_l)

"""
Check if an operator is Hermitian.
"""
ishermitian(A::Operator) = arithmetic_unary_error(ishermitian, A)

"""
Operator exponential.
"""
Base.expm(op::Operator) = throw(ArgumentError("expm() is not defined for this type of operator: $(typeof(op)).\nTry to convert to dense operator first with full()."))

end # module
