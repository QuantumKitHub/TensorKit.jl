"""
    const GLOBAL_CACHES

Registry of the global caches TensorKit maintains, as `name => cache` pairs.

See also [`empty_globalcaches!`](@ref) and [`global_cache_info`](@ref).
"""
const GLOBAL_CACHES = Pair{Symbol, Any}[]

"""
    empty_globalcaches!()

Empty every global cache in [`GLOBAL_CACHES`](@ref).
These mostly contain bookkeeping for various different index manipulations and tensor structures,
so clearing this out can free up some memory whenever you have a workflow that involves a large variety of structures.
For example, you may want to clear the cache when working with different symmetries, or in algorithms that dynamically alter the sizes of tensors.
Since everything is recomputed on demand, this is purely a memory measure.

See also [`global_cache_info`](@ref) to display the status.
"""
function empty_globalcaches!()
    foreach(empty! ∘ last, GLOBAL_CACHES)
    return nothing
end

"""
    global_cache_info([io::IO = stdout])

Print the hit/miss statistics and current size of every global cache in [`GLOBAL_CACHES`](@ref).

See also [`empty_globalcaches!`](@ref).
"""
function global_cache_info(io::IO = stdout)
    for (name, cache) in GLOBAL_CACHES
        println(io, name, ":\t", LRUCache.cache_info(cache))
    end
    return
end

abstract type CacheStyle end
struct NoCache <: CacheStyle end
struct TaskLocalCache{D <: AbstractDict} <: CacheStyle end
struct GlobalLRUCache <: CacheStyle end

const DEFAULT_GLOBALCACHE_SIZE = Ref(10^4)

function CacheStyle(args...)
    return GlobalLRUCache()
end

# category of the miss-path (construction) timer section of an `@cached` function
function _cached_category(fname::Symbol)
    return fname in (:fsbraid, :fstranspose, :treebraider, :treetransposer) ?
        "symmetry" : "bookkeeping"
end

# timer sections are controlled by the `timeit_debug_enabled` switch of the module using `@cached`
_timeit_expr(label, ex) =
    Expr(:macrocall, GlobalRef(TimerOutputs, Symbol("@timeit_debug")), LineNumberNode(@__LINE__, @__FILE__), GlobalRef(@__MODULE__, :GLOBAL_TIMER), label, ex)

macro cached(ex)
    return _cached(__module__, ex)
end

function _cached_error(msg, ex)
    ex isa Expr && (ex = Base.remove_linenums!(copy(ex)))
    Meta.isexpr(ex, :kw) && (ex = Expr(:(=), ex.args...)) # prints as `x = 1`
    throw(ArgumentError("@cached: $msg, got `$ex`"))
end

function _cached(mod::Module, ex)
    if !Meta.isexpr(ex, :function)
        Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[1], (:call, :where, :(::))) &&
            _cached_error("short-form function definitions are not supported, use `function ... end`", ex.args[1])
        _cached_error("expected a function definition", ex)
    end
    fcall = ex.args[1]
    if Meta.isexpr(fcall, :where)
        hasparams = true
        params = fcall.args[2:end]
        fcall = fcall.args[1]
    else
        hasparams = false
    end
    if Meta.isexpr(fcall, :(::))
        typed = true
        typeex = fcall.args[2]
        fcall = fcall.args[1]
    else
        typed = false
    end
    Meta.isexpr(fcall, :call) || _cached_error("expected a function signature", fcall)
    fname = fcall.args[1]
    # qualified names such as `TensorKit.treebraider` add methods to a function of another module
    basename = Meta.isexpr(fname, :.) ? fname.args[end].value : fname
    basename isa Symbol ||
        _cached_error("expected a function name or a qualified name such as `Module.f`", fname)
    # the arguments form the cache key, so they must all be named positional arguments
    for arg in fcall.args[2:end]
        Meta.isexpr(arg, :parameters) && _cached_error("keyword arguments are not supported", fcall)
        Meta.isexpr(arg, :kw) && _cached_error("default values are not supported", arg)
        Meta.isexpr(arg, :...) && _cached_error("varargs are not supported", arg)
        (arg isa Symbol || (Meta.isexpr(arg, :(::)) && length(arg.args) == 2)) ||
            _cached_error("all arguments must be named, since they form the cache key", arg)
    end
    # timer labels for the cache lookup and the miss-path construction
    lookuplabel = string("bookkeeping: cache ", basename)
    misslabel = string(_cached_category(basename), ": compute ", basename)
    fargs = fcall.args[2:end]
    fargnames = map(fargs) do arg
        if Meta.isexpr(arg, :(::))
            return arg.args[1]
        else
            return arg
        end
    end
    _fbody = ex.args[2]

    # actual implenetation, with underscore name
    _fname = Symbol(:_, basename)
    _fcall = Expr(:call, _fname, fargs...)
    if hasparams
        _fcall = Expr(:where, _fcall, params...)
    end
    _fex = Expr(:function, _fcall, _fbody)

    # implementation that chooses the cache style
    newfcall = fcall
    if hasparams
        newfcall = Expr(:where, newfcall, params...)
    end
    cachestylevar = gensym(:cachestyle)
    cachestyleex = Expr(
        :(=), cachestylevar, Expr(:call, GlobalRef(@__MODULE__, :CacheStyle), fname, fargnames...)
    )
    newfbody = Expr(
        :block, cachestyleex, Expr(:call, fname, fargnames..., cachestylevar)
    )
    newfex = Expr(:function, newfcall, newfbody)

    # nocache implementation
    fnocachecall = Expr(:call, fname, fargs..., :(::$NoCache))
    if hasparams
        fnocachecall = Expr(:where, fnocachecall, params...)
    end
    fnocachebody = _timeit_expr(misslabel, Expr(:call, _fname, fargnames...))
    if typed
        T = gensym(:T)
        fnocachebody = Expr(:block, Expr(:(=), T, typeex), Expr(:(::), fnocachebody, T))
    end
    fnocacheex = Expr(:function, fnocachecall, fnocachebody)

    # tasklocal cache implementation
    Dvar = gensym(:D)
    flocalcachecall = Expr(:call, fname, fargs..., :(::$TaskLocalCache{$Dvar}))
    if hasparams
        flocalcachecall = Expr(:where, flocalcachecall, params..., Dvar)
    else
        flocalcachecall = Expr(:where, flocalcachecall, Dvar)
    end
    globalcachename = Symbol(:GLOBAL_, uppercase(string(basename)), :_CACHE)
    localcachename = Symbol(:_tasklocal_, globalcachename)
    cachevar = gensym(:cache)
    getlocalcacheex = :(
        $cachevar::$Dvar = get!(task_local_storage(), $(QuoteNode(localcachename))) do
            return $Dvar()
        end
    )
    valvar = gensym(:val)
    if length(fargnames) == 1
        key = fargnames[1]
    else
        key = Expr(:tuple, fargnames...)
    end
    missex = _timeit_expr(misslabel, Expr(:call, _fname, fargnames...))
    getvalex = _timeit_expr(lookuplabel, :(get!(() -> $missex, $cachevar, $key)))
    if typed
        T = gensym(:T)
        flocalcachebody = Expr(
            :block,
            getlocalcacheex,
            Expr(:(=), T, typeex),
            Expr(:(=), Expr(:(::), valvar, T), getvalex),
            Expr(:return, valvar)
        )
    else
        flocalcachebody = Expr(
            :block,
            getlocalcacheex,
            Expr(:(=), valvar, getvalex),
            Expr(:return, valvar)
        )
    end
    flocalcacheex = Expr(:function, flocalcachecall, flocalcachebody)

    # # global cache implementation
    fglobalcachecall = Expr(:call, fname, fargs..., :(::$GlobalLRUCache))
    if hasparams
        fglobalcachecall = Expr(:where, fglobalcachecall, params...)
    end
    getglobalcachex = Expr(:(=), cachevar, globalcachename)
    if typed
        T = gensym(:T)
        fglobalcachebody = Expr(
            :block,
            getglobalcachex,
            Expr(:(=), T, typeex),
            Expr(:(=), Expr(:(::), valvar, T), getvalex),
            Expr(:return, valvar)
        )
    else
        fglobalcachebody = Expr(
            :block,
            getglobalcachex,
            Expr(:(=), valvar, getvalex),
            Expr(:return, valvar)
        )
    end
    fglobalcacheex = Expr(:function, fglobalcachecall, fglobalcachebody)
    fglobalcachedef = Expr(
        :const,
        Expr(:(=), globalcachename, :($LRU{Any, Any}(; maxsize = $DEFAULT_GLOBALCACHE_SIZE[])))
    )
    # Other modules register their caches explicitly from `__init__`, since
    # mutations of this registry during their precompilation do not survive loading.
    fglobalcacheregister = mod === (@__MODULE__) ? Expr(
            :call, :push!, GlobalRef(@__MODULE__, :GLOBAL_CACHES),
            :($(QuoteNode(globalcachename)) => $globalcachename)
        ) : nothing

    # # total expression
    return esc(
        Expr(
            :block, _fex, newfex, fnocacheex, flocalcacheex,
            fglobalcachedef, fglobalcacheregister, fglobalcacheex
        )
    )
end
