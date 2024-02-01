objects = main.obj md5.obj

all: $(objects)
	nvcc $(CFLAGS) $(objects) -o bfpc

%.obj: %.cu
	nvcc -x cu $(CFLAGS) -I. -dc $< -o $@

clean:
	rm -f *.o *.obj *.exe *.exp *.lib bpfc