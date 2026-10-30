# Caching of the fusion tree manipulations and tensor structures is provided by MemoizationKit.jl.
# `CacheStyle` is imported (not just used) so that the `CacheStyle(::typeof(f), ...)`
# specializations in TensorKit, and those of users, extend `MemoizationKit.CacheStyle`.
import MemoizationKit: CacheStyle
using MemoizationKit: MemoizationKit, @cached, NoCache, GlobalCache, Hashed, LRU

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
    MemoizationKit.empty_caches!(TensorKit)
    return nothing
end

"""
    global_cache_info([io::IO = stdout])

Print the hit/miss statistics and current size of every global cache of TensorKit.

See also [`empty_globalcaches!`](@ref).
"""
function global_cache_info(io::IO = stdout)
    for (f, cache) in MemoizationKit.cache_info(TensorKit)
        println(io, nameof(f), ":\t", cache)
    end
    return
end
