#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern "C"
{
#include <immintrin.h>
}

#define MS 1000
#define NS 1000
#define KS 1000
#define REPEAT 1

void CommonGemm_base(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    return ;
}

void CommonGemm_advance(float *A, float *B, float *C, int M, int N, int K)
{
    register float c01, c02, c03, c04;
    int i = 0, j = 0, k = 0;
    for (j = 0; j < N; j += 4) {
        for (i = 0; i < M; i++) {
            c01 = 0.0;
            c02 = 0.0;
            c03 = 0.0;
            c04 = 0.0;

            for (k = 0; k < K; k++) {
                register float regA = A[i * K + k];

                c01 += regA * B[k * N + j];
                c02 += regA * B[k * N + j + 1];
                c03 += regA * B[k * N + j + 2];
                c04 += regA * B[k * N + j + 3];
            }

            C[i * N + j]     = c01;
            C[i * N + j + 1] = c02;
            C[i * N + j + 2] = c03;
            C[i * N + j + 3] = c04;
        }
    }

    int jR = N % 4;
    if (jR) {
        for (j = N - jR; j < N; j++) {
            for (i = 0; i < M; i++) {
                for (k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }

    return ;
}

void copy(float *input, float *output, int M, int N)
{
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

#if 1
void CommonGemm_avx(float *A, float *B, float *C, int M, int N, int K)
{
    float *copyB = (float *)malloc(sizeof(float) * N * K);
    float tempC[8];
    float tempZero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    copy(B, copyB, K, N);

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    float *pA = A;
    float *ppA = A;

    float *pB = copyB;
    float *ppB = copyB;

    float *pC = C;

    int i = 0, j = 0, k = 0;
    for (i = 0; i < M; i++) {
        ppB = pB;
        pA  = ppA;
        for (j = 0; j < N; j++) {
            ppA = pA;
            ymm1 = __builtin_ia32_loadups256(tempZero);

            for (k = 0; k < K; k += 8) {
                ymm8 = __builtin_ia32_loadups256(ppB);
                ymm0 = __builtin_ia32_loadups256(ppA);

                ymm0 = __builtin_ia32_mulps256(ymm0, ymm8);
                ymm1 = __builtin_ia32_addps256(ymm0, ymm1);

                ppB += 8;
                ppA += 8;
            }
            __builtin_ia32_storeups256(tempC, ymm1);
            for (int p = 0; p < 8; p++) {
                C[i * N + j] += tempC[p];
            }

            int kR = k % 4;
            if (kR) {
                for (k = K - kR; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }

    free(copyB);
}
#endif 

bool VerifyResult(float *A, float *B, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i * N + j] != B[i * N + j]) {
                
                printf("<%d, %d>: A = %lf, B = %lf\n", i, j,  A[i * N + j], B[i * N + j]);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    srand(200);
    float *A = (float *)malloc(sizeof(float) * MS * KS);
    float *B = (float *)malloc(sizeof(float) * KS * NS);
    float *C = (float *)malloc(sizeof(float) * MS * NS);
    float *C2 = (float *)malloc(sizeof(float) * MS * NS);

    for (int i = 0; i < MS * KS; i++) {
        A[i] = rand() % 100;
    }

    for (int i = 0; i < KS * NS; i++) {
        B[i] = rand() % 100;
    }

    for (int i = 0; i < MS * NS; i++) {
        C[i] = 0;
    }

    clock_t startTime, endTime;
    startTime = clock();
    for (int i = 0; i < REPEAT; i++) {
        CommonGemm_base(A, B, C, MS, NS, KS);
    }
    endTime = clock();
    fprintf(stdout, "base TotalTime = %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

    // startTime = clock();
    // for (int i = 0; i < REPEAT; i++) {
    //     CommonGemm_advance(A, B, C2, MS, NS, KS);
    // }
    // endTime   = clock();
    // fprintf(stdout, "advance TotalTiem = %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    
    startTime = clock();
    for (int i = 0; i < REPEAT; i++) {
        CommonGemm_avx(A, B, C2, MS, NS, KS);
    }
    endTime   = clock();
    fprintf(stdout, "avx TotalTiem = %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

    bool r = VerifyResult(C, C2, MS, NS);
    if (r) {
        printf("[success].\n");
    } else {
        printf("[failed].\n");
    }

    free(C2);
    free(C);
    free(B);
    free(A);

    return 0;
}
