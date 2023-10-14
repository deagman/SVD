#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocblas.h>
#include <rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <lapacke.h>
#include <exception>

struct my_timer
{
    struct timeval start_time, end_time;
    double time_use; // us
    void start() {
        gettimeofday(&start_time, NULL);
    }
    void stop() {
        gettimeofday(&end_time, NULL);
        time_use = (end_time.tv_sec - start_time.tv_sec) * 1.0e6 + end_time.tv_usec - start_time.tv_usec;
        time_use /= 1000;
    }
};

struct loc_timer
{
    struct timeval st, ed;
    double run_time = 0;
    int iters = 0;
    void start()
    {
	gettimeofday(&st, NULL);
    }
    void stop()
    {
	gettimeofday(&ed, NULL);
    }
    void add()
    {
	run_time += ((double)(ed.tv_sec - st.tv_sec)) * 1000 + ((double)(ed.tv_usec - st.tv_usec)) / 1000;
	iters++;
    }
    void reset()
    {
	run_time = 0;
	iters = 0;
    }
};

template<typename T>
__global__ void sub_kernel(int n, T vec1, T vec2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < n)
	vec1[idx] -= vec2[idx];
}

template<typename T>
void sub_vector(rocblas_handle handle, int n, T vec1, T vec2)
{
    const int BLOCK_SIZE = 512;
    int blocks = (n-1) / BLOCK_SIZE + 1;

    dim3 dimBlk(BLOCK_SIZE);
    dim3 dimGrid(blocks);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    hipLaunchKernelGGL(sub_kernel, dimGrid, dimBlk, 0, stream, n, vec1, vec2);
}

template<typename T>
void matrix_init(int m, int n, int ld, T* mat)
{
    srand(time(0));
    for(int i = 0; i < n; ++i)
    {
	for(int j = 0; j < m; ++j)
	{
	    T tmp = T(rand()) / T(RAND_MAX);
	    mat[j + i * ld] = i == j ? tmp + 400 : tmp + 40;
	    //mat[j + i * ld] = tmp;
	}
    }
}

template<>
void matrix_init<rocblas_double_complex>(int m, int n, int ld, rocblas_double_complex* mat)
{
    srand(time(0));
    for(int i = 0; i < n; ++i)
    {
	for(int j = 0; j < m; ++j)
	{
	    double real = double(rand()) / double(RAND_MAX);
	    double imag = double(rand()) / double(RAND_MAX);
	    mat[j + i * ld] = i == j ?  \
	      rocblas_double_complex{real + 400, imag + 400} : \
	      rocblas_double_complex{real + 40,  imag + 40};
	}
    }
}

template<typename T, typename S>
__global__ void type_trans_kernel(int n, S* src, T* dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < n)
	dst[idx] = T(src[idx]);
}

template<typename T, typename S>
void type_trans_template(rocblas_handle handle, int n, S* src, T* dst)
{
    const int BLOCK_SIZE = 512;
    int blocks = (n-1) / BLOCK_SIZE + 1;

    dim3 dimBlk(BLOCK_SIZE);
    dim3 dimGrid(blocks);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    hipLaunchKernelGGL(type_trans_kernel, dimGrid, dimBlk, 0, stream, n, src, dst);
}

void type_trans_host(int n, rocblas_double_complex* src, lapack_complex_double* dst)
{
    for(int i = 0; i < n; ++i)
    {
	dst[i] = {src[i].real(), src[i].imag()};
    }
}

void test_zgesvd(int m, int n, int lda, int test, int iters)
{
    printf("test_zgesvd\n");
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    rocblas_double_complex* A, * U, * VT;// and sing .val . matrix S on the host
    lapack_complex_double* A_lapack, * U_lapack, * VT_lapack;
    double* S, * E;// and sing .val . matrix S on the host
    A        = (rocblas_double_complex*)malloc(m * n * sizeof(rocblas_double_complex));
    U        = (rocblas_double_complex*)malloc(m * m * sizeof(rocblas_double_complex));
    VT       = (rocblas_double_complex*)malloc(n * n * sizeof(rocblas_double_complex));
    A_lapack = (lapack_complex_double *)malloc(m * n * sizeof(lapack_complex_double));
    U_lapack = (lapack_complex_double *)malloc(m * m * sizeof(lapack_complex_double));
    VT_lapack= (lapack_complex_double *)malloc(n * n * sizeof(lapack_complex_double));
    S        = (double*)malloc(m     * sizeof(double));
    E        = (double*)malloc((n-1) * sizeof(double));
    int info;

    // matrix_init
    matrix_init(m, n, lda, A);

    // the factorized matrix d_A , orthogonal matrices d_U , d_VT
    rocblas_double_complex* d_A, * d_U, * d_VT, * d_z_S; // and sing .val . matrix d_S
    double* d_S, * d_S_lapack; // and sing .val . matrix d_S
    int* devInfo; // on the device
    double* d_E;
    rocblas_double_complex* d_W; // auxiliary device array (d_W = d_S * d_VT )
    int lwork = 0;
    int info_gpu = 0; // info copied from device to host
    const rocblas_double_complex h_one = 1;
    const rocblas_double_complex h_minus_one = -1;

    // prepare memory on the device
    hipMalloc((void**)&d_A, sizeof(rocblas_double_complex) * m * n);
    hipMalloc((void**)&d_S, sizeof(double) * n);
    hipMalloc((void**)&d_z_S, sizeof(rocblas_double_complex) * n);
    hipMalloc((void**)&d_S_lapack, sizeof(double) * n);
    hipMalloc((void**)&d_U, sizeof(rocblas_double_complex) * m * m);
    hipMalloc((void**)&d_VT, sizeof(rocblas_double_complex) * n * n);
    hipMalloc((void**)&devInfo, sizeof(int));
    hipMalloc((void**)&d_W, sizeof(rocblas_double_complex) * n * n);
    hipMalloc((void**)&d_E, sizeof(double) * (n-1));

    // warm
    if(iters > 1)
    {
	hipMemcpy(d_A, A, sizeof(rocblas_double_complex) * m * n, hipMemcpyHostToDevice); // copy A- >d_A

	// svd 
	my_timer timer1;
	timer1.start();
	rocsolver_zgesvd(handle, rocblas_svect_all, rocblas_svect_all, m, n, d_A, lda, d_S, d_U, lda, d_VT, n, d_E, rocblas_outofplace, devInfo);
	hipDeviceSynchronize();
	timer1.stop();
	//printf("SVD time : %lf ms .\n", timer1.time_use); // print elapsed time
	//hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost); // devInfo - > info_gpu
	//printf("after gesvd : info_gpu = %d\n", info_gpu);
    }

    loc_timer timer;
    for(int i = 0; i < iters; ++i)
    {
	hipMemcpy(d_A, A, sizeof(rocblas_double_complex) * m * n, hipMemcpyHostToDevice); // copy A- >d_A
	
	timer.start();
	// gesvd
	rocsolver_zgesvd(handle, rocblas_svect_all, rocblas_svect_all, m, n, d_A, lda, d_S, d_U, lda, d_VT, n, d_E, rocblas_outofplace, devInfo);

	timer.stop();

	if(*devInfo == 0)
	    timer.add();
    }

    if(timer.iters > 0)
    {
	printf("iters = %d, run_time = %fms\n", timer.iters, timer.run_time/timer.iters);

	int index;
	if(test & 0x1)
	{
	    int ldu = m;
	    int ldv = n;
	    type_trans_host(m*n, A, A_lapack);
	    LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, A_lapack, lda, S, U_lapack, ldu, VT_lapack, ldv, E);
	    hipMemcpy(d_S_lapack, S, sizeof(double) * n, hipMemcpyHostToDevice); // copy S -> d_S_lapack
	    // d_S_lapack -= d_S
	    sub_vector(handle, n, d_S_lapack, d_S);

	    rocblas_idamax(handle, n, d_S_lapack, 1, &index);
	    double S_error = abs(d_S_lapack[index]);
	    printf("S_error            = %E\n", S_error);

	    if(S_error < 2 * min(m, n) * 1.0E-12)
		printf("Test 1 success\n");
	    else
		printf("Test 1 error\n");
	}

	if(test & 0x2)
	{
	    // double d_S -> complex d_S
	    type_trans_template(handle, n, d_S, d_z_S);
	    // multiply d_VT by the diagonal matrix corresponding to d_z_S
	    rocblas_zdgmm(handle, rocblas_side_left, n, n, d_VT, n, d_z_S, 1, d_W, n); // d_W =d_S * d_VT

	    hipMemcpy(d_A, A, sizeof(rocblas_double_complex) * lda * n, hipMemcpyHostToDevice); // copy A- >d_A
	    double Anorm2 = 0.0;
	    rocblas_dznrm2(handle, lda * n, d_A, 1, &Anorm2);

	    // compute the difference d_A - d_U * d_z_S * d_VT
	    rocblas_zgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, n, &h_minus_one, d_U, lda, d_W, n, &h_one, d_A, lda);

	    double dR_fro = 0.0; // variable for the norm
	    // compute the norm of the difference d_A -d_U *d_S * d_VT
	    rocblas_dznrm2(handle, lda * n, d_A, 1, &dR_fro);
	    // compute max_error
	    rocblas_izamax(handle, lda * n, d_A, 1, &index);
	    // TODO
	    double max_error = std::abs(d_A[index]);

	    printf("max_error          = %E \n", max_error);
	    printf("max_error / |A|    = %E \n", max_error / Anorm2);
	    //printf("|A - U*S*VT|       = %E \n", dR_fro); // print the norm
	    //printf("|A - U*S*VT| / |A| = %E \n", dR_fro / Anorm2); // print the norm

	    //if(max_error / Anorm2 < 2 * min(m, n) * 1.0E-14)
	    if(max_error / Anorm2 < 1.0E-14)
		printf("Test 2 success\n");
	    else
		printf("Test 2 error\n");
	}
    }
    else
    {
	printf("error: rocsolver_dgesvd info = %d\n", *devInfo);
    }

    // free host memory
    free(A);
    free(U);
    free(S);
    free(VT);
    free(E);
    free(A_lapack);
    free(U_lapack);
    free(VT_lapack);

    // free device memory
    hipFree(d_A);
    hipFree(d_S);
    hipFree(d_z_S);
    hipFree(d_S_lapack);
    hipFree(d_U);
    hipFree(d_VT);
    hipFree(devInfo);
    hipFree(d_W);
    hipFree(d_E);

    rocblas_destroy_handle(handle);
    hipDeviceReset();
}

void test_dgesvd(int m, int n, int lda, int test, int iters)
{
    printf("test_dgesvd\n");
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    double* A, * A_org, * U, * VT, * S, * E;// and sing .val . matrix S on the host
    A     = (double*)malloc(m * n * sizeof(double));
    A_org = (double*)malloc(m * n * sizeof(double));
    U     = (double*)malloc(m * m * sizeof(double));
    VT    = (double*)malloc(n * n * sizeof(double));
    S     = (double*)malloc(m     * sizeof(double));
    E     = (double*)malloc((n-1) * sizeof(double));
    int info;

    // matrix_init
    matrix_init(m, n, lda, A);

    // the factorized matrix d_A , orthogonal matrices d_U , d_VT
    double* d_A, * d_U, * d_VT, * d_S, * d_S_lapack; // and sing .val . matrix d_S
    int* devInfo; // on the device
    double* d_E;
    double* d_W; // auxiliary device array (d_W = d_S * d_VT )
    int lwork = 0;
    int info_gpu = 0; // info copied from device to host
    const double h_one = 1;
    const double h_minus_one = -1;

    // prepare memory on the device
    hipMalloc((void**)&d_A, sizeof(double) * m * n);
    hipMalloc((void**)&d_S, sizeof(double) * n);
    hipMalloc((void**)&d_S_lapack, sizeof(double) * n);
    hipMalloc((void**)&d_U, sizeof(double) * m * m);
    hipMalloc((void**)&d_VT, sizeof(double) * n * n);
    hipMalloc((void**)&devInfo, sizeof(int));
    hipMalloc((void**)&d_W, sizeof(double) * n * n);
    hipMalloc((void**)&d_E, sizeof(double) * (n-1));

    // warm
    if(iters > 1)
    {
	hipMemcpy(d_A, A, sizeof(double) * m * n, hipMemcpyHostToDevice); // copy A- >d_A

	// svd 
	my_timer timer1;
	timer1.start();
	rocsolver_dgesvd(handle, rocblas_svect_all, rocblas_svect_all, m, n, d_A, lda, d_S, d_U, lda, d_VT, n, d_E, rocblas_outofplace, devInfo);
	hipDeviceSynchronize();
	timer1.stop();
	//printf("SVD time : %lf ms .\n", timer1.time_use); // print elapsed time
	//hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost); // devInfo - > info_gpu
	//printf("after gesvd : info_gpu = %d\n", info_gpu);
    }

    loc_timer timer;
    for(int i = 0; i < iters; ++i)
    {
	hipMemcpy(d_A, A, sizeof(double) * m * n, hipMemcpyHostToDevice); // copy A- >d_A
	
	timer.start();
	// gesvd
	rocsolver_dgesvd(handle, rocblas_svect_all, rocblas_svect_all, m, n, d_A, lda, d_S, d_U, lda, d_VT, n, d_E, rocblas_outofplace, devInfo);

	timer.stop();

	if(*devInfo == 0)
	    timer.add();
    }

    if(timer.iters > 0)
    {
	printf("iters = %d, run_time = %fms\n", timer.iters, timer.run_time/timer.iters);

	int index;
	if(test & 0x1)
	{
	    int ldu = m;
	    int ldv = n;
	    hipMemcpy(A_org, A, sizeof(double) * m * n, hipMemcpyHostToHost); // copy A- >d_A
	    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, A_org, lda, S, U, ldu, VT, ldv, E);
	    hipMemcpy(d_S_lapack, S, sizeof(double) * n, hipMemcpyHostToDevice); // copy S -> d_S_lapack
	    // d_S_lapack -= d_S
	    sub_vector(handle, n, d_S_lapack, d_S);

	    rocblas_idamax(handle, n, d_S_lapack, 1, &index);
	    double S_error = abs(d_S_lapack[index]);
	    printf("S_error            = %E\n", S_error);

	    if(S_error < 2 * min(m, n) * 1.0E-12)
		printf("Test 1 success\n");
	    else
		printf("Test 1 error\n");
	}

	if(test & 0x2)
	{
	    // multiply d_VT by the diagonal matrix corresponding to d_S
	    rocblas_ddgmm(handle, rocblas_side_left, n, n, d_VT, n, d_S, 1, d_W, n); // d_W =d_S * d_VT

	    hipMemcpy(d_A, A, sizeof(double) * lda * n, hipMemcpyHostToDevice); // copy A- >d_A
	    double Anorm2 = 0.0;
	    rocblas_dnrm2(handle, lda * n, d_A, 1, &Anorm2);

	    // compute the difference d_A -d_U *d_S * d_VT
	    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, n, &h_minus_one, d_U, lda, d_W, n, &h_one, d_A, lda);

	    double dR_fro = 0.0; // variable for the norm
	    // compute the norm of the difference d_A -d_U *d_S * d_VT
	    rocblas_dnrm2(handle, lda * n, d_A, 1, &dR_fro);
	    // TODO compute max_error
	    rocblas_idamax(handle, lda * n, d_A, 1, &index);
	    double max_error = abs(d_A[index]);

	    printf("max_error          = %E \n", max_error);
	    printf("max_error / |A|    = %E \n", max_error / Anorm2);
	    //printf("|A - U*S*VT|       = %E \n", dR_fro); // print the norm
	    //printf("|A - U*S*VT| / |A| = %E \n", dR_fro / Anorm2); // print the norm

	    //if(max_error / Anorm2 < 2 * min(m, n) * 1.0E-14)
	    if(max_error / Anorm2 < 1.0E-14)
		printf("Test 2 success\n");
	    else
		printf("Test 2 error\n");
	}
    }
    else
    {
	printf("error: rocsolver_dgesvd info = %d\n", *devInfo);
    }

    // free host memory
    free(A);
    free(U);
    free(S);
    free(VT);
    free(E);

    // free device memory
    hipFree(d_A);
    hipFree(d_S);
    hipFree(d_S_lapack);
    hipFree(d_U);
    hipFree(d_VT);
    hipFree(devInfo);
    hipFree(d_W);
    hipFree(d_E);

    rocblas_destroy_handle(handle);
    hipDeviceReset();
}

#define INPUT_TEST

int main(int argc, char* argv[])
{
#ifdef INPUT_TEST
    int m, n;
    int func = 1;
    int test  = 2;
    int iters = 10; 
    
    switch(argc)
    {
	case 1:
	    printf("Please input m and n.\n");
	    return 0;
	case 2:
	    printf("Please input m and n.\n");
	    return 0;
	case 3:
	    m     = atoi(argv[1]); // number of rows of A
	    n     = atoi(argv[2]); // number of columns of A
	    break;
	case 4:
	    m     = atoi(argv[1]); // number of rows of A
	    n     = atoi(argv[2]); // number of columns of A
	    func  = atoi(argv[3]); 
	    break;
	case 5:
	    m     = atoi(argv[1]); // number of rows of A
	    n     = atoi(argv[2]); // number of columns of A
	    func  = atoi(argv[3]); 
	    test  = atoi(argv[4]); 
	    break;
	case 6:
	    m      = atoi(argv[1]); // number of rows of A
	    n      = atoi(argv[2]); // number of columns of A
	    func   = atoi(argv[3]); 
	    test   = atoi(argv[4]); 
	    iters  = atoi(argv[5]); 
	    break;
    }

    std::string func_names[2] = {"rocsolver_dgesvd", "rocsolver_zgesvd"};

    if(m < 1 || n < 1)
    {
	printf("m or n is invalid\n");
	return 0;
    }
    if(m > 10000 || n > 10000)
    {
	printf("m or n is too large\n");
	return 0;
    }
    if(func != 1 && func != 2)
    {
	printf("function is invalid\n");
	return 0;
    }
    if(test < 0 || test > 3)
    {
	printf("Test only suport 0, 1, 2 and 3\n");
	return 0;
    }
    if(iters < 1 || iters > 20)
    {
	printf("iters should less than 20\n");
	return 0;
    }
    printf("m = %d, n = %d, function = %s, test = %d, iters = %d\n", m, n, func_names[func-1].c_str(), test, iters);

    int lda = m;

    if(func == 1)
	test_dgesvd(m, n, lda, test, iters); 
    else if(func == 2)
	test_zgesvd(m, n, lda, test, iters); 
#else
    //
    int m_array[] = {500, 1000, 1000, 2000, 2000, 4000, 4000};
    int n_array[] = {500, 500,  1000, 500,  2000, 500,  4000};

    int case_num = sizeof(n_array) / sizeof(int);
    printf("case_num = %d\n", case_num);

    int iters = 10;
    int test = 3;

    for(int c = 0; c < case_num; ++c)
    {
	const int m = m_array[c]; // number of rows of A
	const int n = n_array[c]; // number of columns of A
	const int lda = m; // leading dimension of A

	test_dgesvd(m, n, lda, test, iters); 
	test_zgesvd(m, n, lda, test, iters); 
    }
#endif

    return 0;
}
