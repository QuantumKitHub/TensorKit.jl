@is_primitive DefaultCtx Tuple{typeof(scale!), AbstractTensorMap, Number}
@is_primitive DefaultCtx Tuple{typeof(scale!), AbstractTensorMap, AbstractTensorMap, Number}
@is_primitive DefaultCtx Tuple{typeof(add!), AbstractTensorMap, AbstractTensorMap, Number, Number}
@is_primitive DefaultCtx Tuple{typeof(inner), AbstractTensorMap, AbstractTensorMap}
