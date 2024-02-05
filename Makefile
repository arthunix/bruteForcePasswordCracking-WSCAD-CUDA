objects = kernel.obj md5.obj

CFLAGS = -arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80

all: $(objects)
	nvcc $(CFLAGS) $(objects) -o bfpc

%.obj: %.cu
	nvcc -x cu $(CFLAGS) -I. -dc $< -o $@

clean:
	rm -f *.o *.obj *.exe *.exp *.lib bpfc
