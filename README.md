# 14th Marathon of Parallel Programming – SBAC-PAD & WSCAD – 2019 Brute-Force Password Cracking

<p align="justify">Uma forma de quebrar senhas é por meio de ataques com algoritmos de força bruta, que testa todas as possíveis combinações de senhas exaustivamente até encontrar a senha correta. Em geral sistemas guardam apenas o hash da senha dos usuários - hashes criptográficos que não são retornáveis aos dados originais, e que são usados como assinatura de comparação com o hash da senha inserida. Hashes mapeiam senhas com tamanho variável para uma senha de tamanho fixo.

Exemplos de funções criptográficas de hash:

*   MD4, MD5
*   SHA-1
*   SHA-2 (SHA256, SHA512)
*   SHA-3 (SHA3-224, SHA3-256, SHA3-384, SHA3-512)
*   SHAKE128, SHAKE256
*   HMAC

<p align="justify">É possível fazer a comparação dos hashes nos ataques criar tabelas de hashes com senhas comuns, essa técnica ocupa mais espaço que as outras. As técnicas de Brute Force mais comuns são:

*   Dictionary-based Password Attack
*   Rainbow Table Password Cracking
*   Try Common or Default Usernames and Passwords
*   Password Spraying

https://capec.mitre.org/data/definitions/565.html

## The Problem

### Input
<p align="justify">An input consists of only one case of test. The single line contains a string with 32-hexadecimal characters representing the value of MD5 hash of the password to be cracked.
Consider that the possible passwords used to generate the hashes have the length N, with 1 ≤ N ≤ 10.

The input must be read from the standard input.

### Output
<p align="justify">The output is a single line that contains the password value found.
The output must be written to the standard output.

## Os testes
```sh
echo "0aec9f876b8e62cba16c2704d99559d3" > password-5.in # for HELIO
echo "afa345bc5ced1b9bf90a3ff76d8ac111" > password-6.in # for HPCMPP
```

## Medindo
```sh
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 2080 SUPER (UUID: GPU-***)
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:18:00.0 Off |                  N/A |
| 25%   33C    P0    37W / 250W |      0MiB /  8192MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
### Com senha de 5 dígitos:
```sh
0aec9f876b8e62cba16c2704d99559d3
==225786== NVPROF is profiling process 225786, command: ./bfpc
found: HELIO
==225786== Profiling application: ./bfpc
==225786== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  316.05ms         5  63.209ms  16.512us  308.84ms  iterate(int, int*)
                    0.00%  2.0160us         2  1.0080us     992ns  1.0240us  [CUDA memcpy HtoD]
      API calls:   65.03%  316.05ms         5  63.211ms  17.531us  308.84ms  cudaDeviceSynchronize
                   30.39%  147.69ms         2  73.847ms  6.8270us  147.69ms  cudaMemcpyToSymbol
                    4.24%  20.620ms         1  20.620ms  20.620ms  20.620ms  cudaMallocManaged
                    0.21%  1.0347ms       456  2.2680us     141ns  119.85us  cuDeviceGetAttribute
                    0.06%  270.77us         5  54.153us  5.0810us  246.67us  cudaLaunchKernel
                    0.03%  142.04us         1  142.04us  142.04us  142.04us  cuLibraryLoadData
                    0.02%  100.82us         4  25.205us  22.157us  34.008us  cuDeviceGetName
                    0.02%  81.557us         1  81.557us  81.557us  81.557us  cudaFree
                    0.00%  17.250us         4  4.3120us  1.3540us  10.123us  cuDeviceGetPCIBusId
                    0.00%  3.7880us         3  1.2620us     230ns  2.8140us  cuDeviceGetCount
                    0.00%  1.5790us         8     197ns     138ns     520ns  cuDeviceGet
                    0.00%  1.2330us         4     308ns     269ns     400ns  cuDeviceTotalMem
                    0.00%  1.0270us         5     205ns     122ns     479ns  cudaPeekAtLastError
                    0.00%     710ns         4     177ns     155ns     232ns  cuDeviceGetUuid
                    0.00%     310ns         1     310ns     310ns     310ns  cuModuleGetLoadingMode

==225786== Unified Memory profiling result:
Device "NVIDIA GeForce RTX 2080 SUPER (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       2  32.000KB  4.0000KB  60.000KB  64.00000KB  8.576000us  Host To Device
       2  32.000KB  4.0000KB  60.000KB  64.00000KB  6.656000us  Device To Host
       1         -         -         -           -  778.7500us  Gpu page fault groups
Total CPU Page faults: 2
```
### Com senha de 6 dígitos:
```sh
afa345bc5ced1b9bf90a3ff76d8ac111
==225818== NVPROF is profiling process 225818, command: ./bfpc
found: HPCMPP
==225818== Profiling application: ./bfpc
==225818== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  11.4116s         6  1.90193s  16.192us  11.1031s  iterate(int, int*)
                    0.00%  1.8560us         2     928ns     768ns  1.0880us  [CUDA memcpy HtoD]
      API calls:   98.21%  11.4116s         6  1.90194s  16.477us  11.1031s  cudaDeviceSynchronize
                    1.60%  185.49ms         2  92.743ms  14.840us  185.47ms  cudaMemcpyToSymbol
                    0.18%  20.657ms         1  20.657ms  20.657ms  20.657ms  cudaMallocManaged
                    0.01%  1.1967ms       456  2.6240us     191ns  136.18us  cuDeviceGetAttribute
                    0.00%  278.07us         6  46.344us  5.0700us  247.78us  cudaLaunchKernel
                    0.00%  168.81us         1  168.81us  168.81us  168.81us  cudaFree
                    0.00%  165.36us         1  165.36us  165.36us  165.36us  cuLibraryLoadData
                    0.00%  117.63us         4  29.407us  25.443us  39.226us  cuDeviceGetName
                    0.00%  15.654us         4  3.9130us  1.8620us  9.9400us  cuDeviceGetPCIBusId
                    0.00%  2.2620us         8     282ns     184ns     895ns  cuDeviceGet
                    0.00%  1.8370us         3     612ns     307ns  1.1490us  cuDeviceGetCount
                    0.00%  1.6510us         4     412ns     233ns     875ns  cuDeviceTotalMem
                    0.00%  1.2660us         6     211ns     128ns     548ns  cudaPeekAtLastError
                    0.00%  1.0590us         4     264ns     236ns     331ns  cuDeviceGetUuid
                    0.00%     408ns         1     408ns     408ns     408ns  cuModuleGetLoadingMode

==225818== Unified Memory profiling result:
Device "NVIDIA GeForce RTX 2080 SUPER (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       2  32.000KB  4.0000KB  60.000KB  64.00000KB  8.512000us  Host To Device
       2  32.000KB  4.0000KB  60.000KB  64.00000KB  6.657000us  Device To Host
       1         -         -         -           -  746.8460us  Gpu page fault groups
Total CPU Page faults: 2
```