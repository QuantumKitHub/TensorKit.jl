# NEW MACROS: @planar and @plansor
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
