CC=nvcc
CFLAGS=
IFLAGS=-I${HOME}/softs/FreeImage/include
LFLAGS=-L${HOME}/softs/FreeImage/lib/
LDFLAGS=-lfreeimage
EXE=modif_img.exe

all: modif_img.o
	$(CC) $(CFLAGS) -o $(EXE) $^ $(LFLAGS) $(LDFLAGS)

modif_img.o: modif_img.cu
	$(CC) $(CFLAGS) -c $< $(IFLAGS)
 
clean: mrproper
	rm -f *.o

mrproper:
	rm -f $(EXE)
