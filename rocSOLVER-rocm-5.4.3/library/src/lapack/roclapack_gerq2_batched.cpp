/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gerq2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gerq2_batched_impl(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int lda,
                                            T* ipiv,
                                            const rocblas_stride stridep,
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gerq2_batched", "-m", m, "-n", n, "--lda", lda, "--strideP", stridep,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gerq2_gerqf_argCheck(handle, m, n, lda, A, ipiv, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARF and LARFG
    size_t size_Abyx_norms;
    // size of temporary array to store diagonal elements
    size_t size_diag;
    rocsolver_gerq2_getMemorySize<true, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                           &size_Abyx_norms, &size_diag);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_diag);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gerq2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, stridep,
                                       batch_count, (T*)scalars, work_workArr, (T*)Abyx_norms,
                                       (T*)diag);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgerq2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* ipiv,
                                        const rocblas_stride stridep,
                                        const rocblas_int batch_count)
{
    return rocsolver_gerq2_batched_impl<float>(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

rocblas_status rocsolver_dgerq2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* ipiv,
                                        const rocblas_stride stridep,
                                        const rocblas_int batch_count)
{
    return rocsolver_gerq2_batched_impl<double>(handle, m, n, A, lda, ipiv, stridep, batch_count);
}

rocblas_status rocsolver_cgerq2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* ipiv,
                                        const rocblas_stride stridep,
                                        const rocblas_int batch_count)
{
    return rocsolver_gerq2_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, stridep,
                                                               batch_count);
}

rocblas_status rocsolver_zgerq2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* ipiv,
                                        const rocblas_stride stridep,
                                        const rocblas_int batch_count)
{
    return rocsolver_gerq2_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, stridep,
                                                                batch_count);
}

} // extern C
