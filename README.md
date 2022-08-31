# Mini Projet CUDA

Utilisation de nVidia CUDA pour faire du traîtement sur image.

## Effets implémentés


1. Saturation 
2. Symétrie horizontale (miroir)
3. Flou
4. Niveau de gris
5. Contours (méthode de SOBEL)
6. Monochromatique
7.  Negatif

## Utilisation

```bash
Usage : ./modif_img.exe MODE [ARGS]
```
Avec les modes :
>1. Saturation ARGS
>2. Mirroir
>3. Flou
>4. Gris
>5. Controus
>6. Monochrome ARGS
>7. Negatif

	>ARGS = 0 : Pixel rouge
            1 : Pixel vert
            2 : Pixel bleu


## Dépendances

### FreeImage

Le programme de traîtement d'image proposé utilise la bibliothèque [FreeImage](https://freeimage.sourceforge.io/ "The FreeImage Project website"). Le matériel de compilation suppose que cette bibliothèque est installé dans le répertoire :
>${HOME}/softs/FreeImage/

### CUDA

Les noyaux de calcul du programme étant écrit en CUDA, il faut également avoir installé [nvcc](https://developer.nvidia.com/cuda-downloads "Nvidia CUDA Compiler") et posséder un matériel compatible CUDA.
