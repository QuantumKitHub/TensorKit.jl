struct SingletonDict{K, V} <: AbstractDict{K, V}
    key::K
    value::V
    SingletonDict{K, V}(p::Pair{K, V}) where {K, V} = new{K, V}(p.first, p.second)
end
SingletonDict(p::Pair{K, V}) where {K, V} = SingletonDict{K, V}(p)
function SingletonDict(g::Base.Generator)
    s = iterate(g)
    @assert s !== nothing
    first, state = s
    @assert iterate(g, state) === nothing
    return SingletonDict(first)
end

Base.length(::SingletonDict) = 1
Base.keys(d::SingletonDict) = (d.key,)
Base.values(d::SingletonDict) = (d.value,)
Base.haskey(d::SingletonDict, key) = isequal(d.key, key)
Base.getindex(d::SingletonDict, key) = isequal(d.key, key) ? d.value : throw(KeyError(key))
Base.get(d::SingletonDict, key, default) = isequal(d.key, key) ? d.value : default

Base.iterate(d::SingletonDict, s = true) = s ? ((d.key => d.value), false) : nothing

struct VectorDict{K, V} <: AbstractDict{K, V}
    keys::Vector{K}
    values::Vector{V}
end
VectorDict{K, V}() where {K, V} = VectorDict{K, V}(Vector{K}(), Vector{V}())
VectorDict() = VectorDict{Any, Any}()

function VectorDict{K, V}(kvs) where {K, V}
    keys = Vector{K}()
    values = Vector{V}()
    if Base.IteratorSize(kv) !== SizeUnknown()
        sizehint!(keys, length(kvs))
        sizehint!(values, length(kvs))
    end
    for (k, v) in kvs
        push!(keys, k)
        push!(values, v)
    end
    return VectorDict{K, V}(keys, values)
end
VectorDict(kv1::Pair{K, V}, kvs::Pair{K, V}...) where {K, V} = VectorDict{K, V}((kv1, kvs...))
VectorDict(g::Base.Generator) = VectorDict(g...)

Base.length(d::VectorDict) = length(d.keys)
function Base.sizehint!(d::VectorDict, newsz)
    (sizehint!(d.keys, newsz); sizehint!(d.values, newsz); return d)
end

@propagate_inbounds getpair(d::VectorDict, i::Integer) = d.keys[i] => d.values[i]

Base.copy(d::VectorDict) = VectorDict(copy(d.keys), copy(d.values))
Base.empty(::VectorDict, ::Type{K}, ::Type{V}) where {K, V} = VectorDict{K, V}()
Base.empty!(d::VectorDict) = (empty!(d.keys); empty!(d.values); return d)

function Base.delete!(d::VectorDict, key)
    i = findfirst(isequal(key), d.keys)
    if !(i === nothing || i == 0)
        deleteat!(d.keys, i)
        deleteat!(d.values, i)
    end
    return d
end

Base.keys(d::VectorDict) = d.keys
Base.values(d::VectorDict) = d.values
Base.haskey(d::VectorDict, key) = key in d.keys
function Base.getindex(d::VectorDict, key)
    i = findfirst(isequal(key), d.keys)
    @inbounds begin
        return i !== nothing ? d.values[i] : throw(KeyError(key))
    end
end
function Base.setindex!(d::VectorDict, v, key)
    i = findfirst(isequal(key), d.keys)
    if i === nothing
        push!(d.keys, key)
        push!(d.values, v)
    else
        d.values[i] = v
    end
    return d
end

function Base.get(d::VectorDict, key, default)
    i = findfirst(isequal(key), d.keys)
    @inbounds begin
        return i !== nothing ? d.values[i] : default
    end
end

function Base.iterate(d::VectorDict, s = 1)
    @inbounds if s > length(d)
        return nothing
    else
        return (d.keys[s] => d.values[s]), s + 1
    end
end

struct SortedVectorDict{K, V} <: AbstractDict{K, V}
    keys::Vector{K}
    values::Vector{V}
    function SortedVectorDict{K, V}(pairs::Vector{Pair{K, V}}) where {K, V}
        pairs = sort!(pairs; by = first)
        return new{K, V}(first.(pairs), last.(pairs))
    end
    function SortedVectorDict{K, V}(keys::Vector{K}, values::Vector{V}) where {K, V}
        @assert issorted(keys)
        return new{K, V}(keys, values)
    end
    SortedVectorDict{K, V}() where {K, V} = new{K, V}(Vector{K}(undef, 0), Vector{V}(undef, 0))
end
SortedVectorDict{K, V}(kv::Pair{K, V}...) where {K, V} = SortedVectorDict{K, V}(kv)
function SortedVectorDict{K, V}(kv) where {K, V}
    d = SortedVectorDict{K, V}()
    if Base.IteratorSize(kv) !== SizeUnknown()
        sizehint!(d, length(kv))
    end
    for (k, v) in kv
        push!(d, k => v)
    end
    return d
end
SortedVectorDict(pairs::Vector{Pair{K, V}}) where {K, V} = SortedVectorDict{K, V}(pairs)
@noinline function _no_pair_error()
    msg = "SortedVectorDict(kv): kv needs to be an iterator of pairs"
    throw(ArgumentError(msg))
end
function SortedVectorDict(pairs::Vector)
    all(p -> isa(p, Pair), pairs) || _no_pair_error()
    pairs = sort!(pairs; by = first)
    keys = map(first, pairs)
    values = map(last, pairs)
    return SortedVectorDict{eltype(keys), eltype(values)}(keys, values)
end

SortedVectorDict(kv::Pair{K, V}...) where {K, V} = SortedVectorDict{K, V}(kv)

_getKV(::Type{Pair{K, V}}) where {K, V} = (K, V)
function SortedVectorDict(kv)
    if Base.IteratorEltype(kv) === Base.HasEltype()
        P = eltype(kv)
    elseif kv isa Base.Generator && kv.f isa Type
        P = kv.f
    else
        P = Base.Core.Compiler.return_type(first, Tuple{typeof(kv)})
    end
    if P <: Pair && Base.isconcretetype(P)
        K, V = _getKV(P)
        return SortedVectorDict{K, V}(kv)
    else
        return SortedVectorDict(collect(kv))
    end
end
SortedVectorDict() = SortedVectorDict{Any, Any}()

Base.length(d::SortedVectorDict) = length(d.keys)
function Base.sizehint!(d::SortedVectorDict, newsz)
    (sizehint!(d.keys, newsz); sizehint!(d.values, newsz); return d)
end

function Base.copy(d::SortedVectorDict{K, V}) where {K, V}
    return SortedVectorDict{K, V}(copy(d.keys), copy(d.values))
end
Base.empty(::SortedVectorDict, ::Type{K}, ::Type{V}) where {K, V} = SortedVectorDict{K, V}()
Base.empty!(d::SortedVectorDict) = (empty!(d.keys); empty!(d.values); return d)

# _searchsortedfirst(v::Vector, k) = searchsortedfirst(v, k)
function _searchsortedfirst(v::Vector, k)
    i = 1
    @inbounds while i <= length(v) && isless(v[i], k)
        i += 1
    end
    return i
end

function Base.delete!(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return d
    end
    i = _searchsortedfirst(d.keys, key)
    if i <= length(d) && isequal(d.keys[i], key)
        deleteat!(d.keys, i)
        deleteat!(d.values, i)
    end
    return d
end

Base.keys(d::SortedVectorDict) = d.keys
Base.values(d::SortedVectorDict) = d.values
function Base.haskey(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return false
    end
    i = _searchsortedfirst(d.keys, key)
    return (i <= length(d) && isequal(d.keys[i], key))
end
function Base.getindex(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        throw(KeyError(k))
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds if (i <= length(d) && isequal(d.keys[i], key))
        return d.values[i]
    else
        throw(KeyError(key))
    end
end
function Base.setindex!(d::SortedVectorDict{K}, v, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        throw(ArgumentError("$k is not a valid key for type $K"))
    end
    i = _searchsortedfirst(d.keys, key)
    if i <= length(d) && isequal(d.keys[i], key)
        d.values[i] = v
    else
        insert!(d.keys, i, key)
        insert!(d.values, i, v)
    end
    return d
end

function Base.get(d::SortedVectorDict{K}, k, default) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return default
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        return (i <= length(d) && isequal(d.keys[i], key)) ? d.values[i] : default
    end
end
function Base.get(f::Union{Function, Type}, d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return f()
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        return (i <= length(d) && isequal(d.keys[i], key)) ? d.values[i] : f()
    end
end
function Base.iterate(d::SortedVectorDict, i = 1)
    @inbounds if i > length(d)
        return nothing
    else
        return (d.keys[i] => d.values[i]), i + 1
    end
end

function Base.:(==)(d1::SortedVectorDict, d2::SortedVectorDict)
    length(d1) == length(d2) || return false
    for (k1, v1, k2, v2) in zip(d1.keys, d1.values, d2.keys, d2.values)
        if !(isequal(k1, k2) && isequal(v1, v2))
            return false
        end
    end
    return true
end
