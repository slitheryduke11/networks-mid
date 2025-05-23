#ifndef BMP_UTILS_H
#define BMP_UTILS_H

#include <stdio.h>

// Estructura RGB
typedef struct { unsigned char b, g, r; } Pixel;

// Variables globales accesibles
extern int width, height;
extern unsigned char header[54];

// Funciones BMP
void readHeader(FILE *in);
void writeBMP(int img, const char *suffix, Pixel *buf, int kernel_size);
void createFolder(const char *path);

#endif
