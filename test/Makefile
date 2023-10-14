CC=hipcc

CUR_PATH=$(shell pwd)

LAPACK_DIR=${CUR_PATH}/../depend/lapack-install
LAPACK_LIB=${LAPACK_DIR}/lib64
LAPACK_INC=${LAPACK_DIR}/include

all=gesvd_test

gesvd_test: gesvd_test.o
	${CC}  $^ -o $@  -lrocsolver -lrocblas -L${LAPACK_LIB} -llapacke -lcblas -llapack -lblas -lgfortran

gesvd_test.o: gesvd_test.cpp
	${CC} -std=c++14 -I${LAPACK_INC} -c $^  -o $@  

.PHONY: clean
clean:
	rm -f *.o gesvd_test
