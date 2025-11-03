# Scheduler implementation
# ------------------------
function select_scheduler(scheduler = OhMyThreads.Implementation.NotGiven(); kwargs...)
    return if scheduler == OhMyThreads.Implementation.NotGiven() && isempty(kwargs)
        Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
    else
        OhMyThreads.Implementation._scheduler_from_userinput(scheduler; kwargs...)
    end
end

"""
    const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

The default scheduler used when looping over different blocks in the matrix representation of a
tensor.
For controlling this value, see also [`set_blockscheduler`](@ref) and [`with_blockscheduler`](@ref).
"""
const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

"""
    with_blockscheduler(f, [scheduler]; kwargs...)

Run `f` in a scope where the `blockscheduler` is determined by `scheduler' and `kwargs...`.
"""
@inline function with_blockscheduler(
        f, scheduler = OhMyThreads.Implementation.NotGiven(); kwargs...
    )
    return @with blockscheduler => select_scheduler(scheduler; kwargs...) f()
end

# TODO: disable for trivial symmetry or small tensors?
default_blockscheduler(t::AbstractTensorMap) = default_blockscheduler(typeof(t))
default_blockscheduler(::Type{T}) where {T <: AbstractTensorMap} = blockscheduler[]
