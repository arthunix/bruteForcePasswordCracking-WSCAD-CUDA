#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md5.cuh"
#include "device_launch_parameters.h"

#define MAX 7
#define LETTERS_LEN 36

typedef unsigned char byte;

char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
__device__ __constant__ char dLetters[LETTERS_LEN];
__device__ __constant__ byte dHash1[MD5_DIGEST_LENGTH];

void strHex_to_byte(char*, byte*);
__device__ int dstrncmp(const char*, const char*, size_t);
__device__ __host__ void print_digest(byte*);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void iterate(int len, int* dOk) {
	char str[MAX];
	byte hash2[MD5_DIGEST_LENGTH];
	long long index, divisor;

	index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < len; i++) {
		divisor = (int)pow((double)LETTERS_LEN, (double)(len - i - 1));
		str[i] = dLetters[(index / divisor) % LETTERS_LEN];
	}

	//printf("idx: %lld.str: %s\n", index, str);

	MD5((byte*)str, len, (unsigned int*)hash2);
	if (dstrncmp((char*)dHash1, (char*)hash2, MD5_DIGEST_LENGTH) == 0) {
		printf("found: ");
		for(int i = 0; i < len; i++)
			printf("%c", str[i]);
		printf("\n");
		*dOk = 1;
	}
	delete[] str;
}

int main(int argc, char** argv) {
	int lenMax = MAX;
	int len, r;
	char hash1_str[2 * MD5_DIGEST_LENGTH + 1];
	byte hash1[MD5_DIGEST_LENGTH]; // password hash
	int *dOk;

	// Input:
	r = scanf("%s", hash1_str);

	// Check input.
	if (r == EOF || r == 0) {
		fprintf(stderr, "Error!\n");
		exit(1);
	}

	// Convert hexadecimal string to hash byte.
	strHex_to_byte(hash1_str, hash1);
	print_digest(hash1);

	gpuErrchk( cudaMemcpyToSymbol(dLetters, &letters, LETTERS_LEN, 0, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpyToSymbol(dHash1, &hash1, MD5_DIGEST_LENGTH, 0, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMallocManaged(&dOk, 1*sizeof(int)) );
	*dOk = 0;

	// Generate all possible passwords of different sizes.
	for (len = 1; len <= lenMax && *dOk != 1; len++) {
		dim3 gridArch = { (unsigned int)pow(LETTERS_LEN, len-1), 1, 1 };
		dim3 blckArch = { LETTERS_LEN, 1, 1 };

		iterate <<< gridArch, blckArch >>> (len, dOk);

		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}

	gpuErrchk( cudaFree(dOk) );
}

void strHex_to_byte(char* str, byte* hash) {
	char* pos = str;
	int i;
	for (i = 0; i < MD5_DIGEST_LENGTH / sizeof * hash; i++) {
		sscanf(pos, "%2hhx", &hash[i]);
		pos += 2;
	}
}

__device__ int dstrncmp(const char* _l, const char* _r, size_t n) {
	unsigned char ll, lr;
	if (n == 0) {
		return 0;
	}
	while (n-- > 0 && *_l == *_r) {
		if (n == 0 || *_l == '\0') {
			return 0;
		}
		_l++;
		_r++;
	}
	ll = (*(unsigned char*)_l);
	lr = (*(unsigned char*)_r);
	return ((ll < lr) ? -1 : (ll > lr));
}

__device__ __host__ void print_digest(byte* hash) {
	int x;
	for (x = 0; x < MD5_DIGEST_LENGTH; x++) {
		printf("%02x", hash[x]);
	}
	printf("\n");
}
