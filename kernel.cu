#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md5.cuh"
#include "device_launch_parameters.h"

#define MAX 3
#define LETTERS_LEN 36

typedef unsigned char byte;

char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
__device__ __constant__ char dLetters[LETTERS_LEN];
__device__ __constant__ byte dHash1[MD5_DIGEST_LENGTH];

void strHex_to_byte(char*, byte*);
__device__ int dstrncmp(const char*, const char*, size_t);
__device__ size_t dstrlen(const char*);
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

__global__ void iterate(int len) {
	char* str = new char[len];

	if (len == 1) {
		str[0] = dLetters[threadIdx.x];
		printf("%s\n", str);
	}
	else if (len == 2) {
		str[0] = dLetters[blockIdx.x];
		str[1] = dLetters[threadIdx.x];
		printf("%s\n", str);
	}

	delete[] str;
}

int main(int argc, char** argv) {
	int lenMax = MAX;
	int len;
	//int ok = 0;
	int r;
	char hash1_str[2 * MD5_DIGEST_LENGTH + 1];
	byte hash1[MD5_DIGEST_LENGTH]; // password hash

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

	// Generate all possible passwords of different sizes.
	for (len = 1; len <= lenMax; len++) {
		iterate <<<(unsigned int)pow(36, len-1),36>>>(len);
		gpuErrchk(cudaPeekAtLastError());
	}
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

__device__ size_t dstrlen(const char* str) {
	const char* s;
	for (s = str; *s; ++s);
	return(s - str);
}

__device__ __host__ void print_digest(byte* hash) {
	int x;
	for (x = 0; x < MD5_DIGEST_LENGTH; x++) {
		printf("%02x", hash[x]);
	}
	printf("\n");
}