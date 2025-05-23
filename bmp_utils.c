#include "bmp_utils.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

int width, height;
unsigned char header[54];

void readHeader(FILE *in) {
    if (fread(header, sizeof(header), 1, in) != 1) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    width = *(int *)&header[18];
    height = *(int *)&header[22];
}

void createFolder(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) != 0) {
            perror("[ERROR] mkdir");
            exit(EXIT_FAILURE);
        }
    }
}

void writeBMP(int img, const char *suffix, Pixel *buf, int kernel_size) {
    char oname[128];
    snprintf(oname, sizeof(oname), "salidas/%06d_%s_%d.bmp", img, suffix, kernel_size);
    FILE *fout = fopen(oname, "wb");
    if (!fout) {
        fprintf(stderr, "[ERROR] No se puede crear '%s'\n", oname);
        return;
    }
    fwrite(header, sizeof(header), 1, fout);
    size_t npix = (size_t)width * height;
    fwrite(buf, sizeof(Pixel), npix, fout);
    fclose(fout);
}
