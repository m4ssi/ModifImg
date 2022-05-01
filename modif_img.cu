#include <iostream>
#include <string.h>
#include <cuda.h>
#include <math.h>
#include "FreeImage.h"

#include <cstdlib>
#include <cstdio>

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CHECK_LAST_ERROR_ASYNC CUDA_CHECK(cudaGetLastError())

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

__global__ void saturation_kernel ( char * d_img, unsigned width, unsigned height, int p)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  
  // Kernel
  if ( g_thread_i < (width * height))
  {
    d_img[idx + p] = (char) 255;
  }
}

__global__ void keep_one_kernel ( char * d_img, unsigned width, unsigned height, int p)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  
  // Kernel
  if ( g_thread_i < (width * height))
  {
    for ( int i = 0; i < 3; i++)
      if ( i != p)
        d_img[idx + i] = (char) 0;
 }
}

__global__ void grey_kernel ( char * d_img, unsigned width, unsigned height)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  
  // Kernel
  if ( g_thread_i < (width * height))
  {
    d_img[idx + 0] = 0.299 * d_img[idx + 0] + 0.587 * d_img[idx + 1] + 0.114 * d_img[idx + 2];
    d_img[idx + 1] = 0.299 * d_img[idx + 0] + 0.587 * d_img[idx + 1] + 0.114 * d_img[idx + 2];
    d_img[idx + 2] = 0.299 * d_img[idx + 0] + 0.587 * d_img[idx + 1] + 0.114 * d_img[idx + 2];
  }
}

__global__ void diapositif_kernel ( char * d_img, unsigned width, unsigned height)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  
  // Kernel
  if ( g_thread_i < (width * height))
  {
    d_img[idx + 0] = (char) 255 - d_img[idx + 0];
    d_img[idx + 1] = (char) 255 - d_img[idx + 1];
    d_img[idx + 2] = (char) 255 - d_img[idx + 2];
  }
}

__global__ void mirror_kernel ( char * d_img, char * d_mirror, unsigned width, unsigned height)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  int _idx  = ((width * height) - 1) - g_thread_i;		  // Indice du pixel RGB (sans controle de bord)
  _idx = _idx * 3;
  // Kernel
  if ( g_thread_i < (width * height))
  {
	  
     d_mirror[_idx + 0] = d_img[idx+0];
     d_mirror[_idx + 1] = d_img[idx+1];
     d_mirror[_idx + 2] = d_img[idx+2];
 }
}

__global__ void blurry_kernel (  char * d_img, char * d_blurry, unsigned width, unsigned height)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.y * blockDim.x + threadIdx.x, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  int acc_r = 0,
      acc_g = 0,
      acc_b = 0,
      nb = 0;
    // Kernel
  if (g_thread_i < (width * height)) {
	// printf("%d\n", g_thread_i);
	
	// horizontal
	if (g_thread_i % width > 0) {
		acc_r += d_img[(idx + 0) - 3];
		acc_g += d_img[(idx + 1) - 3];
		acc_b += d_img[(idx + 2) - 3];
		nb++;
	}
	if (g_thread_i % width < (width - 1)) {
		acc_r += d_img[(idx + 0) + 3];
		acc_g += d_img[(idx + 1) + 3];
		acc_b += d_img[(idx + 2) + 3];
		nb++;
	}
	// vertical
	if ((g_thread_i / width) % height > 0) {
		acc_r += d_img[(idx + 0) - 3 * width];
		acc_g += d_img[(idx + 1) - 3 * width];
		acc_b += d_img[(idx + 2) - 3 * width];
		nb++;
	}
	if ((g_thread_i / width) % height < (height - 1)) {
		acc_r += d_img[(idx + 0) + 3 * width];
		acc_g += d_img[(idx + 1) + 3 * width];
		acc_b += d_img[(idx + 2) + 3 * width];
		nb++;
	}
	d_blurry[idx + 0] = (char) acc_r / nb;
	d_blurry[idx + 1] = (char) acc_g / nb;
	d_blurry[idx + 2] = (char) acc_b / nb;
  }
}

__global__ void sobel_kernel ( char * d_img, char * d_sobel, int * sobel, unsigned width, unsigned height)
{
  int g_block_i  = blockIdx.y * gridDim.x + blockIdx.x,    // Indice global du block
      n_threads  = blockDim.x * blockDim.y,                // Nombre de thread par block
      b_thread_i = threadIdx.x * blockDim.y + threadIdx.y, // Indice local du thread (au sein d'un bloc)
      g_thread_i = g_block_i * n_threads + b_thread_i;     // Indice global du thread

  int idx  = g_thread_i * 3;			                  // Indice du pixel RGB (sans controle de bord)
  int  Gx = 0,
       Gy = 0,
       grey = 0;
                
    // Kernel
  if (g_thread_i < (width * height)) {
    if ( (g_thread_i % width > 0) && (g_thread_i % width < (width - 1)) && ((g_thread_i / width) % height > 0) && ((g_thread_i / width) % height < (height - 1)))
    {
      Gx += -1 * d_img [ idx - 3 - 3 * width] + 1 * d_img[ idx - 3 + 3 * width];
      Gx += -2 * d_img [ idx - 3] + 2 * d_img[ idx + 3];
      Gx += -1 * d_img [ idx + 3 - 3 * width] + 1 * d_img[ idx + 3 + 3 * width];

      Gy += -1 * d_img [ idx - 3 - 3 * width]  - 2 * d_img [ idx - 3] - 1 * d_img[ idx - 3 + 3 * width];
      Gy += 1 * d_img [ idx + 3 - 3 * width]  + 2 * d_img [ idx + 3] + 1 * d_img[ idx + 3 + 3 * width];
      
      
	  grey = (int) sqrt( (float) (Gx * Gx + Gy * Gy));
	  //~ grey = grey % 255;
	  //~ if ( grey < 50) grey = 180;
      d_sobel[idx + 0] = (char) grey;
      d_sobel[idx + 1] = (char) grey;
      d_sobel[idx + 2] = (char) grey;
	}
    else
	{
      d_sobel[idx + 0] = (char) 255;
      d_sobel[idx + 1] = (char) 255;
      d_sobel[idx + 2] = (char) 255;
    }
  }
}

int main (int argc , char** argv)
{
	
  if ( argc < 2)
  {
    fprintf ( stderr, "Usage : %s MODE [ARGS]\n", argv[0]);
    fprintf ( stderr, "      MODE : 1 Saturation (ARGS= 0:R, 1:G, 2:B\n");
    fprintf ( stderr, "      MODE : 2 Mirroir\n");
    fprintf ( stderr, "      MODE : 3 Flou\n");
    fprintf ( stderr, "      MODE : 4 Gris\n");
    fprintf ( stderr, "      MODE : 5 Controus\n");
    fprintf ( stderr, "      MODE : 6 Keep one pixel\n");
    fprintf ( stderr, "      MODE : 7 Keep one pixel\n");
    exit(1);
  }
  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  int mode = atoi ( argv[1]);
  int pix_sat = -1;
  if ( mode == 1 || mode == 6)
    pix_sat = atoi(argv[2]);

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  char *img = (char*) malloc(sizeof(char) * 3 * width * height);
  int *sobel = (int*) malloc(sizeof(int) * 9);// * width * height);
  //~ int sobel[9] = {-1, 0, 2, -1, 0, 2, -1, 0, 2};
  sobel[0] = -1;
  sobel[1] = 0;
  sobel[2] = 2;
  sobel[3] = -1;
  sobel[4] = 0;
  sobel[5] = 2;
  sobel[6] = -1;
  sobel[7] = 0;
  sobel[8] = 2;

  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);

  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  char *d_img;
  char *d_tmp;
  int *d_sobel;
  cudaMalloc ( (void **) &d_img, sizeof(char) * 3 * width * height);
  cudaMalloc ( (void **) &d_tmp, sizeof(char) * 3 * width * height);
  cudaMalloc ( (void **) &d_sobel, sizeof(int) * 9);

  cudaMemcpy(d_img, img, 3 * width * height * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sobel, sobel, 9 * sizeof(int), cudaMemcpyHostToDevice);

  int x_block = 32;
  int y_block = 32;
  int x_grid = height / x_block + 1;
  int y_grid = width / y_block + 1;

  dim3 dimBlock (x_block, y_block);
  dim3 dimGrid (x_grid, y_grid);


  if ( mode == 1)
  {
	  saturation_kernel<<<dimGrid, dimBlock>>> ( d_img, width, height, pix_sat);
	  cudaMemcpy ( img, d_img, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);
  }
  if ( mode == 2)
  {
	  mirror_kernel <<<dimGrid, dimBlock>>> ( d_img, d_tmp, width, height);
	  cudaMemcpy ( img, d_tmp, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);
  }
  if ( mode == 3)
  {
	for ( int i = 0; i < 200; i++)
	{
	  blurry_kernel <<<dimGrid, dimBlock>>> ( d_img, d_tmp, width, height);
	  cudaMemcpy ( d_img, d_tmp, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(img, d_tmp, 3 * width * height * sizeof ( int), cudaMemcpyDeviceToHost);
  }
  if ( mode == 4)
  {
	  grey_kernel<<<dimGrid, dimBlock>>> ( d_img, width, height);
	  cudaMemcpy ( img, d_img, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);
  }
  if ( mode == 5)
  {
    grey_kernel <<<dimGrid, dimBlock>>>(d_img, width, height);
    sobel_kernel <<<dimGrid, dimBlock>>>(d_img, d_tmp, d_sobel , width, height);
    cudaMemcpy(img, d_tmp, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);  
  }
  if ( mode == 6)
  {
	  keep_one_kernel<<<dimGrid, dimBlock>>> ( d_img, width, height, pix_sat);
	  cudaMemcpy ( img, d_img, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);
  }
  if ( mode == 7)
  {
	  diapositif_kernel<<<dimGrid, dimBlock>>> ( d_img, width, height);
	  cudaMemcpy ( img, d_img, 3 * width * height * sizeof ( char), cudaMemcpyDeviceToHost);
  }

  


  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !
  
  free(img);
  free(sobel);
  cudaFree(d_img);
  cudaFree(d_tmp);
  cudaFree(d_sobel);
}
