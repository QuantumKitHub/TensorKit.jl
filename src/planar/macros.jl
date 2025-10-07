# NEW MACROS: @planar and @plansor
"""
    @planar [kwargs...] expr

Perform a planar tensor contraction, i.e., a tensor contraction that respects the 
non-trivial braiding of fermionic or anyonic tensor indices. This macro is the 
planar counterpart of the `@tensor` macro from TensorOperations.jl.

The `@planar` macro should be used when working with tensors that have fermionic or 
anyonic symmetries (i.e., `BraidingStyle(sectortype(tensor)) isa Fermionic` or 
`BraidingStyle(sectortype(tensor)) isa Anyonic`). For such tensors, the order of 
tensor contractions and index permutations matters due to non-trivial braiding, and 
the `@planar` macro correctly handles the braiding tensors `τ` that encode these 
braidings.

# Syntax

The syntax is identical to `@tensor`, with the addition that braiding tensors can be 
explicitly specified using `τ[i j; k l]` to denote a braiding between indices.

```julia
@planar C[i; j] := A[i; k] * B[k; j]
@planar C[i; j] := A[i; k] * τ[k l; m n] * B[m; n] * τ[n o; j l]
```

# Keyword arguments

The same keyword arguments as `@tensor` are supported:
- `opt=true` or `opt=optexpr`: optimize the contraction order
- `backend=backend`: specify the backend for the contraction
- `allocator=allocator`: specify the allocator for temporary tensors
- `order=(...)`: manually specify the contraction order
- `contractcheck=true`: add runtime checks for contraction validity

# Key differences from `@tensor`

While `@tensor` assumes trivial braiding (appropriate for bosonic tensors), `@planar` 
correctly handles non-trivial braiding statistics. Specifically:

1. `@tensor` ignores the order of tensor factors in a product (they commute)
2. `@planar` respects the order and handles explicit braiding tensors `τ`
3. `@planar` uses planar operations (`planaradd!`, `planarcontract!`, etc.) that 
   account for braiding when permuting indices

For bosonic tensors (trivial braiding), `@planar` reduces to the same operations as 
`@tensor`, but with potential performance overhead. Use `@plansor` to automatically 
dispatch to the appropriate implementation based on the braiding style.

# Example

```julia
# Fermionic tensors
V = Vect[FermionParity](0 => 2, 1 => 2)
A = randn(V ⊗ V ← V)
B = randn(V ← V ⊗ V)

# Correct handling of fermionic statistics
@planar C[i; j] := A[i k; l] * B[l; k j]

# With explicit braiding tensor
@planar C[i; j] := A[i k; l] * τ[k m; n l] * B[n; m j]
```

See also: [`@plansor`](@ref), [`@tensor`](@ref)
"""
macro planar(args::Vararg{Expr})
    isempty(args) && throw(ArgumentError("No arguments passed to `planar`"))

    planarexpr = args[end]
    kwargs = TO.parse_tensor_kwargs(args[1:(end - 1)])
    parser = planarparser(planarexpr, kwargs...)

    return esc(parser(planarexpr))
end

function planarparser(planarexpr, kwargs...)
    parser = TO.TensorParser()

    pop!(parser.preprocessors) # remove TO.extracttensorobjects
    push!(parser.preprocessors, _conj_to_adjoint)
    push!(parser.preprocessors, _extract_tensormap_objects)

    temporaries = Vector{Symbol}()
    push!(parser.postprocessors, ex -> _annotate_temporaries(ex, temporaries))
    push!(parser.postprocessors, ex -> _free_temporaries(ex, temporaries))
    push!(parser.postprocessors, _insert_planar_operations)

    # braiding tensors need to be instantiated before kwargs are processed
    push!(parser.preprocessors, _construct_braidingtensors)

    # the order of backend and allocator postprocessors are important so let's find them first
    hasbackend = false
    for (name, val) in kwargs
        if name == :backend
            hasbackend = true
            backend = val
            push!(parser.postprocessors, ex -> TO.insertbackend(ex, backend))
            break
        end
    end
    for (name, val) in kwargs
        if name == :allocator
            allocator = val
            if !hasbackend
                backend = Expr(:call, GlobalRef(TensorOperations, :DefaultBackend))
                push!(parser.postprocessors, ex -> TO.insertbackend(ex, backend))
            end
            push!(parser.postprocessors, ex -> TO.insertallocator(ex, allocator))
            break
        end
    end
    for (name, val) in kwargs
        if name == :order
            isexpr(val, :tuple) ||
                throw(ArgumentError("Invalid use of `order`, should be `order=(...,)`"))
            indexorder = map(normalizeindex, val.args)
            parser.contractiontreebuilder = network -> TO.indexordertree(
                network, indexorder
            )

        elseif name == :contractcheck
            val isa Bool ||
                throw(ArgumentError("Invalid use of `contractcheck`, should be `contractcheck=bool`."))
            val && push!(parser.preprocessors, ex -> TO.insertcontractionchecks(ex))

        elseif name == :costcheck
            val in (:warn, :cache) ||
                throw(ArgumentError("Invalid use of `costcheck`, should be `costcheck=warn` or `costcheck=cache`"))
            parser.contractioncostcheck = val
        elseif name == :opt
            if val isa Bool && val
                optdict = TO.optdata(planarexpr)
            elseif val isa Expr
                optdict = TO.optdata(val, planarexpr)
            else
                throw(ArgumentError("Invalid use of `opt`, should be `opt=true` or `opt=OptExpr`"))
            end
            parser.contractiontreebuilder = network -> TO.optimaltree(network, optdict)[1]
        elseif !(name == :allocator || name == :backend) # already processed
            throw(ArgumentError("Unknown keyword argument `$name`."))
        end
    end

    treebuilder = parser.contractiontreebuilder
    treesorter = parser.contractiontreesorter
    costcheck = parser.contractioncostcheck
    push!(
        parser.preprocessors,
        ex -> TO.processcontractions(ex, treebuilder, treesorter, costcheck)
    )
    parser.contractioncostcheck = nothing
    push!(parser.preprocessors, ex -> _check_planarity(ex))
    push!(parser.preprocessors, ex -> _decompose_planar_contractions(ex, temporaries))

    return parser
end

"""
    @plansor [kwargs...] expr

Automatically dispatch between `@tensor` and `@planar` based on the braiding style 
of the tensors involved in the contraction.

This macro provides a unified interface for tensor contractions that works correctly 
for both bosonic tensors (trivial braiding) and fermionic/anyonic tensors (non-trivial 
braiding). At runtime, it checks `BraidingStyle(sectortype(tensor))` and:

- If `Bosonic`: uses the standard `@tensor` implementation (efficient, ignores braiding)
- If `Fermionic` or `Anyonic`: uses the `@planar` implementation (handles braiding correctly)

# Syntax

The syntax is the same as `@tensor` and `@planar`:

```julia
@plansor C[i; j] := A[i; k] * B[k; j]
@plansor C[i; j] := A[i; k] * τ[k l; m n] * B[m; n] * τ[n o; j l]
```

# Keyword arguments

The same keyword arguments as `@tensor` and `@planar` are supported:
- `opt=true` or `opt=optexpr`: optimize the contraction order
- `backend=backend`: specify the backend for the contraction
- `allocator=allocator`: specify the allocator for temporary tensors
- `order=(...)`: manually specify the contraction order
- `contractcheck=true`: add runtime checks for contraction validity

# When to use `@plansor` vs `@tensor` vs `@planar`

- Use `@tensor` when you know your tensors have trivial braiding (bosonic, or no symmetries)
  and want maximum performance
- Use `@planar` when you know your tensors have non-trivial braiding (fermionic, anyonic)
  and need correct handling of braiding
- Use `@plansor` when writing generic code that should work for both cases, or when you're 
  unsure about the braiding properties of your tensors

# Performance considerations

For bosonic tensors, `@plansor` has a small runtime overhead compared to `@tensor` due to 
the runtime dispatch. For performance-critical code with known bosonic tensors, prefer 
`@tensor`. For fermionic/anyonic tensors, `@plansor` and `@planar` have identical performance.

# Example

```julia
# Works correctly for both bosonic and fermionic tensors
function my_contraction(A::AbstractTensorMap, B::AbstractTensorMap)
    @plansor C[i; j] := A[i; k] * B[k; j]
    return C
end

# Bosonic case (uses @tensor internally)
V_bosonic = ℂ^2
A_bosonic = randn(V_bosonic ⊗ V_bosonic ← V_bosonic)
B_bosonic = randn(V_bosonic ← V_bosonic ⊗ V_bosonic)
C_bosonic = my_contraction(A_bosonic, B_bosonic)

# Fermionic case (uses @planar internally)
V_fermionic = Vect[FermionParity](0 => 2, 1 => 2)
A_fermionic = randn(V_fermionic ⊗ V_fermionic ← V_fermionic)
B_fermionic = randn(V_fermionic ← V_fermionic ⊗ V_fermionic)
C_fermionic = my_contraction(A_fermionic, B_fermionic)
```

See also: [`@planar`](@ref), [`@tensor`](@ref)
"""
macro plansor(args::Vararg{Expr})
    isempty(args) && throw(ArgumentError("No arguments passed to `planar`"))

    planarexpr = args[end]
    kwargs = TO.parse_tensor_kwargs(args[1:(end - 1)])
    return esc(_plansor(planarexpr, kwargs...))
end

function _plansor(expr, kwargs...)
    inputtensors = TO.getinputtensorobjects(expr)
    newtensors = TO.getnewtensorobjects(expr)

    # find the first non-braiding tensor to determine the braidingstyle
    targetobj = inputtensors[findfirst(x -> x != :τ, inputtensors)]
    if !isa(targetobj, Symbol)
        targetsym = gensym(string(targetobj))
        expr = TO.replacetensorobjects(expr) do obj, leftind, rightind
            return obj == targetobj ? targetsym : obj
        end
        args = Any[(:($targetsym = $targetobj))]
    else
        targetsym = targetobj
        args = Any[]
    end

    tparser = TO.tensorparser(expr, kwargs...)
    pparser = planarparser(expr, kwargs...)
    insert!(tparser.preprocessors, 4, _remove_braidingtensors)
    tensorex = tparser(expr)
    planarex = pparser(expr)

    push!(
        args,
        Expr(:if, :(BraidingStyle(sectortype($targetsym)) isa Bosonic), tensorex, planarex)
    )
    if !isa(targetobj, Symbol) && targetobj ∈ newtensors
        push!(args, :($targetobj = $targetsym))
    end
    return Expr(:block, args...)
end
