const GLOBAL_CACHES = Pair{Symbol, Any}[]
function empty_globalcaches!()
    foreach(empty! ∘ last, GLOBAL_CACHES)
    return nothing
end

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

macro cached(ex)
    Meta.isexpr(ex, :function) ||
        error("cached macro can only be used on function definitions")
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
    Meta.isexpr(fcall, :call) ||
        error("cached macro can only be used on function definitions")
    fname = fcall.args[1]
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
    _fname = Symbol(:_, fname)
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
        :(=), cachestylevar, Expr(:call, :CacheStyle, fname, fargnames...)
    )
    newfbody = Expr(
        :block, cachestyleex, Expr(:call, fname, fargnames..., cachestylevar)
    )
    newfex = Expr(:function, newfcall, newfbody)

    # nocache implementation
    fnocachecall = Expr(:call, fname, fargs..., :(::NoCache))
    if hasparams
        fnocachecall = Expr(:where, fnocachecall, params...)
    end
    fnocachebody = Expr(:call, _fname, fargnames...)
    if typed
        T = gensym(:T)
        fnocachebody = Expr(:block, Expr(:(=), T, typeex), Expr(:(::), fnocachebody, T))
    end
    fnocacheex = Expr(:function, fnocachecall, fnocachebody)

    # tasklocal cache implementation
    Dvar = gensym(:D)
    flocalcachecall = Expr(:call, fname, fargs..., :(::TaskLocalCache{$Dvar}))
    if hasparams
        flocalcachecall = Expr(:where, flocalcachecall, params..., Dvar)
    else
        flocalcachecall = Expr(:where, flocalcachecall, Dvar)
    end
    localcachename = Symbol(:_tasklocal_, fname, :_cache)
    cachevar = gensym(:cache)
    getlocalcacheex = :(
        $cachevar::$Dvar = get!(task_local_storage(), $localcachename) do
            return $Dvar()
        end
    )
    valvar = gensym(:val)
    if length(fargnames) == 1
        key = fargnames[1]
    else
        key = Expr(:tuple, fargnames...)
    end
    getvalex = :(
        get!($cachevar, $key) do
            return $_fname($(fargnames...))
        end
    )
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
    fglobalcachecall = Expr(:call, fname, fargs..., :(::GlobalLRUCache))
    if hasparams
        fglobalcachecall = Expr(:where, fglobalcachecall, params...)
    end
    globalcachename = Symbol(:GLOBAL_, uppercase(string(fname)), :_CACHE)
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
        Expr(:(=), globalcachename, :(LRU{Any, Any}(; maxsize = DEFAULT_GLOBALCACHE_SIZE[])))
    )
    fglobalcacheregister = Expr(
        :call, :push!, :GLOBAL_CACHES, :($(QuoteNode(globalcachename)) => $globalcachename)
    )

    # # total expression
    return esc(
        Expr(
            :block, _fex, newfex, fnocacheex, flocalcacheex,
            fglobalcachedef, fglobalcacheregister, fglobalcacheex
        )
    )
end
