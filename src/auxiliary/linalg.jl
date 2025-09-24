# Simple reference to getting and setting BLAS threads
#------------------------------------------------------
set_num_blas_threads(n::Integer) = LinearAlgebra.BLAS.set_num_threads(n)
get_num_blas_threads() = LinearAlgebra.BLAS.get_num_threads()
