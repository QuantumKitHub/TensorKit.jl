import Base: transpose

for f in (:rand, :randn, :zeros, :ones)
    @eval begin
        Base.@deprecate TensorMap(::typeof($f), T::Type, P::HomSpace) $f(T, P)
        Base.@deprecate TensorMap(::typeof($f), P::HomSpace) $f(P)
        Base.@deprecate TensorMap(::typeof($f), T::Type, cod::TensorSpace, dom::TensorSpace) $f(T, cod, dom)
        Base.@deprecate TensorMap(::typeof($f), cod::TensorSpace, dom::TensorSpace) $f(cod, dom)
        Base.@deprecate Tensor(::typeof($f), T::Type, space::TensorSpace) $f(T, space)
        Base.@deprecate Tensor(::typeof($f), space::TensorSpace) $f(space)
    end
end

Base.@deprecate(randuniform(dims::Base.Dims), rand(dims))
Base.@deprecate(randuniform(T::Type{<:Number}, dims::Base.Dims), rand(T, dims))
Base.@deprecate(randnormal(dims::Base.Dims), randn(dims))
Base.@deprecate(randnormal(T::Type{<:Number}, dims::Base.Dims), randn(T, dims))
Base.@deprecate(randhaar(dims::Base.Dims), randisometry(dims))
Base.@deprecate(randhaar(T::Type{<:Number}, dims::Base.Dims), randisometry(T, dims))

for (f1, f2) in ((:randuniform, :rand), (:randnormal, :randn), (:randisometry, :randisometry), (:randhaar, :randisometry))
    @eval begin
        Base.@deprecate TensorMap(::typeof($f1), T::Type, P::HomSpace) $f2(T, P)
        Base.@deprecate TensorMap(::typeof($f1), P::HomSpace) $f2(P)
        Base.@deprecate TensorMap(::typeof($f1), T::Type, cod::TensorSpace, dom::TensorSpace) $f2(T, P, cod, dom)
        Base.@deprecate TensorMap(::typeof($f1), cod::TensorSpace, dom::TensorSpace) $f2(cod, dom)
        Base.@deprecate Tensor(::typeof($f1), T::Type, space::TensorSpace) $f2(T, space)
        Base.@deprecate Tensor(::typeof($f1), space::TensorSpace) $f2(space)
    end
end

Base.@deprecate EuclideanProduct() EuclideanInnerProduct()

Base.@deprecate insertunit(P::ProductSpace, args...; kwargs...) insertleftunit(args...; kwargs...)

#! format: on
