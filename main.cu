#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md5.cuh"

#define MAX 3
#define LETTERS_LEN 36

typedef unsigned char byte;

char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
__device__ __constant__ char dLetters[LETTERS_LEN];

void strHex_to_byte(char*, byte*);
void digest(byte* hash);
__device__ int ustrncmp(const char*, const char*, size_t);
__device__ size_t ustrlen(const char*);
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

__global__ void iterate(byte* hash, int len, int* ok) {
	char* str = new char[len];

	if (len == 1) {
		str[0] = dLetters[blockIdx.x];
	}
	else if (len == 2) {
		str[0] = dLetters[blockIdx.x];
		str[1] = dLetters[blockIdx.x];
	}
	else if (len == 3) {
		str[0] = dLetters[blockIdx.x];
		str[1] = dLetters[blockIdx.x];
		str[2] = dLetters[blockIdx.x];
	}

	printf("%s\n", str);
	delete[] str;
}

int main(int argc, char** argv) {
	int lenMax = MAX;
	int len;
	int ok = 0, r;
	char hash1_str[2 * MD5_DIGEST_LENGTH + 1];
	byte hash1[MD5_DIGEST_LENGTH]; // password hash

	cudaMemcpyToSymbol(dLetters, &letters, LETTERS_LEN, 0, cudaMemcpyHostToDevice);

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

	// Generate all possible passwords of different sizes.
	for (len = 1; len <= lenMax; len++) {
		iterate <<< pow(36, len), 36 >>> (hash1, len, &ok);
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

__device__ int ustrncmp(const char* s1, const char* s2, size_t n) {
	unsigned char uc1, uc2;
	if (n == 0) {
		return 0;
	}
	while (n-- > 0 && *s1 == *s2) {
		if (n == 0 || *s1 == '\0') {
			return 0;
		}
		s1++;
		s2++;
	}
	uc1 = (*(unsigned char*)s1);
	uc2 = (*(unsigned char*)s2);
	return ((uc1 < uc2) ? -1 : (uc1 > uc2));
}

__device__ size_t ustrlen(const char* str) {
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
