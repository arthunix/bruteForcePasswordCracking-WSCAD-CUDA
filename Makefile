objects = kernel.obj md5.obj

CFLAGS = -arch=all -G -allow-unsupported-compiler

all: $(objects)
	nvcc $(CFLAGS) $(objects) -o bfpc

%.obj: %.cu
	nvcc -x cu $(CFLAGS) -I. -dc $< -o $@

clean:
	rm -f *.o *.obj *.exe *.exp *.lib bpfc
