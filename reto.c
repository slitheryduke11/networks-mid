#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#define KERNEL_SIZE 5
#define MAX_IMAGES 100

int ancho, alto;
unsigned char header[54];
long total_lecturas = 0;
long total_escrituras = 0;

typedef struct {
    unsigned char r, g, b;
} Pixel;

void leerHeader(FILE *in) {
    fread(header, sizeof(unsigned char), 54, in);
    ancho = *(int*)&header[18];
    alto = *(int*)&header[22];
}

Pixel** leerImagen(FILE *in) {
    Pixel **matriz = malloc(alto * sizeof(Pixel*));
    for (int i = 0; i < alto; i++) {
        matriz[i] = malloc(ancho * sizeof(Pixel));
        for (int j = 0; j < ancho; j++) {
            fread(&matriz[i][j].b, sizeof(unsigned char), 1, in);
            fread(&matriz[i][j].g, sizeof(unsigned char), 1, in);
            fread(&matriz[i][j].r, sizeof(unsigned char), 1, in);
            total_lecturas += 3;
        }
    }
    return matriz;
}

void escribirImagen(Pixel **matriz, FILE *out) {
    fwrite(header, sizeof(unsigned char), 54, out);
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            fwrite(&matriz[i][j].b, sizeof(unsigned char), 1, out);
            fwrite(&matriz[i][j].g, sizeof(unsigned char), 1, out);
            fwrite(&matriz[i][j].r, sizeof(unsigned char), 1, out);
            total_escrituras += 3;
        }
    }
}

Pixel** escalaGrises(Pixel **original) {
    Pixel **gris = malloc(alto * sizeof(Pixel*));
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        gris[i] = malloc(ancho * sizeof(Pixel));
        for (int j = 0; j < ancho; j++) {
            unsigned char pixel = 0.21 * original[i][j].r + 0.72 * original[i][j].g + 0.07 * original[i][j].b;
            gris[i][j].r = gris[i][j].g = gris[i][j].b = pixel;
        }
    }
    return gris;
}

Pixel** espejoHorizontal(Pixel **original) {
    Pixel **espejo = malloc(alto * sizeof(Pixel*));
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        espejo[i] = malloc(ancho * sizeof(Pixel));
        for (int j = 0; j < ancho; j++) {
            espejo[i][j] = original[i][ancho - j - 1];
        }
    }
    return espejo;
}

Pixel** espejoVertical(Pixel **original) {
    Pixel **espejo = malloc(alto * sizeof(Pixel*));
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        espejo[i] = malloc(ancho * sizeof(Pixel));
        for (int j = 0; j < ancho; j++) {
            espejo[i][j] = original[alto - i - 1][j];
        }
    }
    return espejo;
}

Pixel** desenfoque(Pixel **original) {
    Pixel **des = malloc(alto * sizeof(Pixel*));
    int k = KERNEL_SIZE / 2;
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        des[i] = malloc(ancho * sizeof(Pixel));
        for (int j = 0; j < ancho; j++) {
            int r = 0, g = 0, b = 0, count = 0;
            for (int ki = -k; ki <= k; ki++) {
                for (int kj = -k; kj <= k; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < alto && nj >= 0 && nj < ancho) {
                        r += original[ni][nj].r;
                        g += original[ni][nj].g;
                        b += original[ni][nj].b;
                        count++;
                    }
                }
            }
            des[i][j].r = r / count;
            des[i][j].g = g / count;
            des[i][j].b = b / count;
        }
    }
    return des;
}

void liberar(Pixel **img) {
    for (int i = 0; i < alto; i++) free(img[i]);
    free(img);
}

int main() {
    const double start = omp_get_wtime();
    FILE *log = fopen("estadisticas.txt", "w");

    for (int img_num = 1; img_num <= MAX_IMAGES; img_num++) {
        char input_filename[100];
        char output_filename[100];
        snprintf(input_filename, sizeof(input_filename), "imagenes_bmp_final/%06d.bmp", img_num);

        FILE *in = fopen(input_filename, "rb");
        if (!in) {
            fprintf(stderr, "No se pudo abrir %s\n", input_filename);
            continue;
        }

        leerHeader(in);
        Pixel **original = leerImagen(in);
        fclose(in);

        Pixel **gris = escalaGrises(original);
        Pixel **esp_h = espejoHorizontal(original);
        Pixel **esp_v = espejoVertical(original);
        Pixel **esp_h_gris = espejoHorizontal(gris);
        Pixel **esp_v_gris = espejoVertical(gris);
        Pixel **blur = desenfoque(original);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_gris.bmp", img_num);
        FILE *out1 = fopen(output_filename, "wb"); escribirImagen(gris, out1); fclose(out1);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_esp_h.bmp", img_num);
        FILE *out2 = fopen(output_filename, "wb"); escribirImagen(esp_h, out2); fclose(out2);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_esp_v.bmp", img_num);
        FILE *out3 = fopen(output_filename, "wb"); escribirImagen(esp_v, out3); fclose(out3);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_esp_h_gris.bmp", img_num);
        FILE *out4 = fopen(output_filename, "wb"); escribirImagen(esp_h_gris, out4); fclose(out4);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_esp_v_gris.bmp", img_num);
        FILE *out5 = fopen(output_filename, "wb"); escribirImagen(esp_v_gris, out5); fclose(out5);

        snprintf(output_filename, sizeof(output_filename), "salidas/%06d_blur.bmp", img_num);
        FILE *out6 = fopen(output_filename, "wb"); escribirImagen(blur, out6); fclose(out6);

        liberar(original); liberar(gris); liberar(esp_h); liberar(esp_v); liberar(esp_h_gris); liberar(esp_v_gris); liberar(blur);

        fprintf(log, "Imagen %06d.bmp -> Lecturas: %ld, Escrituras: %ld\n", img_num, total_lecturas, total_escrituras);

        total_lecturas = 0;
        total_escrituras = 0;
    }

    const double stop = omp_get_wtime();
    double tiempo = stop - start;
    fprintf(log, "\nTiempo total: %.2lf segundos\n", tiempo);
    fclose(log);
    return 0;
}
