g++ -c gemm.cc -std=c++11 -mavx
g++ -o gemm gemm.o
./gemm
