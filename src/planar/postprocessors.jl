# Additional postprocessors for @planar and @plansor

# Temporaries were explicitly created by _decompose_planar_contractions and were thus
# instantiated as if they were new output tensors rather than temporary tensors; we need
# to correct for this by adding the `istemp = true` flag.
function _annotate_temporaries(ex, temporaries)
    if isexpr(ex, :(=)) && isexpr(ex.args[2], :call) &&
            ex.args[2].args[1] ∈
            (GlobalRef(TO, :tensoralloc_add), GlobalRef(TO, :tensoralloc_contract))
        lhs = ex.args[1]
        i = findfirst(==(lhs), temporaries)
        if i !== nothing
            rhs = ex.args[2]
            # add `istemp = true` flag
            newrhs = Expr(:call, rhs.args[1:(end - 1)]..., Val(true))
            return Expr(:(=), lhs, newrhs)
        end
    elseif ex isa Expr
        return Expr(ex.head, [_annotate_temporaries(a, temporaries) for a in ex.args]...)
    end
    return ex
end

function _contains_arg(ex, arg)
    if ex isa Expr
        return any(_contains_arg(a, arg) for a in ex.args)
    else
        return ex == arg
    end
end

function _free_temporaries(ex, temporaries)
    if isexpr(ex, :block)
        newargs = copy(ex.args)
        for t in temporaries
            i = findlast(e -> _contains_arg(e, t), newargs)
            @assert !isnothing(i)
            if i == length(newargs)
                @assert isexpr(newargs[i], :(=))
                lhs = newargs[i].args[1]
                push!(newargs, Expr(:call, :tensorfree!, t))
                push!(newargs, lhs)
            else
                newargs = insert!(newargs, i + 1, Expr(:call, :tensorfree!, t))
            end
        end
        return Expr(:block, newargs...)
    end
end

# Replace the tensor operations created by `TO.instantiate` with the corresponding
# planar operations, immediately inserting them with `GlobalRef`.

# TODO: replace _planarmethod with planarmethod in everything below
const _PLANAR_OPERATIONS = (:planaradd!, :planartrace!, :planarcontract!)

function _insert_planar_operations(ex)
    if isexpr(ex, :call)
        if ex.args[1] == GlobalRef(TensorOperations, :tensoradd!)
            conjA = popat!(ex.args, 5)
            @assert !conjA "conj flag should be disabled"
            return Expr(
                ex.head, GlobalRef(TensorKit, Symbol(:planaradd!)),
                map(_insert_planar_operations, ex.args[2:end])...
            )
        elseif ex.args[1] == GlobalRef(TensorOperations, :tensorcontract!)
            conjB = popat!(ex.args, 8)
            conjA = popat!(ex.args, 5)
            @assert !conjA && !conjB "conj flags should be disabled ($conjA), ($conjB)"
            return Expr(
                ex.head, GlobalRef(TensorKit, Symbol(:planarcontract!)),
                map(_insert_planar_operations, ex.args[2:end])...
            )
        elseif ex.args[1] == GlobalRef(TensorOperations, :tensortrace!)
            conjA = popat!(ex.args, 6)
            @assert !conjA "conj flag should be disabled"
            return Expr(
                ex.head, GlobalRef(TensorKit, Symbol(:planartrace!)),
                map(_insert_planar_operations, ex.args[2:end])...
            )
        elseif ex.args[1] in TensorOperations.tensoroperationsfunctions
            return Expr(
                ex.head, GlobalRef(TensorOperations, ex.args[1]),
                map(_insert_planar_operations, ex.args[2:end])...
            )
        end
    elseif isa(ex, Expr)
        return Expr(ex.head, (_insert_planar_operations(e) for e in ex.args)...)
    end
    return ex
end

"""
    insertplanarbackend(ex, backend)

Insert the backend argument into the tensor operation methods `planaradd!`, `planartrace!`, and `planarcontract!`.

See also: [`TensorOperations.insertbackend`](@ref).
"""
function insertplanarbackend(ex, backend)
    return TO.insertargument(ex, backend, (:planaradd!, :planartrace!, :planarcontract!))
end

"""
    insertplanarallocator(ex, allocator)

Insert the allocator argument into the tensor operation methods `planaradd!`, `planartrace!`, and `planarcontract!`.

See also: [`TensorOperations.insertallocator`](@ref).
"""
function insertplanarallocator(ex, allocator)
    return TO.insertargument(ex, allocator, (:planaradd!, :planartrace!, :planarcontract!))
end
