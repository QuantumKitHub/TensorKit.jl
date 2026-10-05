# Caching of the fusion tree manipulations and tensor structures is provided by Cached.jl.
# `CacheStyle` is imported (not just used) so that the `CacheStyle(::typeof(f), ...)`
# specializations in TensorKit, and those of users, extend `Cached.CacheStyle`.
import Cached: CacheStyle
using Cached: Cached, @cached, NoCache, GlobalCache, Hashed, LRU

"""
    const GLOBAL_CACHES

Registry of additional global caches, which are not managed by `@cached` (e.g. by
extensions), as `name => cache` pairs. They are emptied by [`empty_globalcaches!`](@ref)
and listed by [`global_cache_info`](@ref), together with the caches of TensorKit's cached
functions.
"""
const GLOBAL_CACHES = Pair{Symbol, Any}[]

const DEFAULT_GLOBALCACHE_SIZE = Ref(10^4)

# Buffer payloads can be measured without visiting their elements when stored inline.
# Reference-containing sector/scalar types need the recursive fallback to include their data.
_cache_payload_size(a::Array{T}) where {T} = isbitstype(T) ? sizeof(a) : Base.summarysize(a)

# Include the index tables as well as the values (Dictionaries 0.4's Indices layout).
function _cache_payload_size(inds::Indices)
    return sizeof(getfield(inds, :slots)) + sizeof(getfield(inds, :hashes)) +
        _cache_payload_size(getfield(inds, :values))
end

"""
    empty_globalcaches!()

Empty the global caches of TensorKit.
These mostly contain bookkeeping for various different index manipulations and tensor structures,
so clearing this out can free up some memory whenever you have a workflow that involves a large variety of structures.
For example, you may want to clear the cache when working with different symmetries, or in algorithms that dynamically alter the sizes of tensors.
Since everything is recomputed on demand, this is purely a memory measure.

See also [`global_cache_info`](@ref) to display the status.
"""
function empty_globalcaches!()
    Cached.empty_caches!(TensorKit)
    foreach(empty! ∘ last, GLOBAL_CACHES)
    return nothing
end

"""
    global_cache_info([io::IO = stdout])

Print the hit/miss statistics and current size of every global cache of TensorKit.

See also [`empty_globalcaches!`](@ref).
"""
function global_cache_info(io::IO = stdout)
    for (f, cache) in Cached.cache_info(TensorKit)
        println(io, nameof(f), ":\t", cache)
    end
    for (name, cache) in GLOBAL_CACHES
        println(io, name, ":\t", cache)
    end
    return
end
