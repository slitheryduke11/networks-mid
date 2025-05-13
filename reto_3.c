#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// Variables globales de dimensiones y cabecera BMP
static int ancho, alto;
static unsigned char header[54];

// Contadores de I/O por ejecución y totales
static long total_lecturas = 0, total_escrituras = 0;
static long all_lecturas = 0, all_escrituras = 0;

// Parámetros por defecto
static int KERNEL_SIZE = 5;
static int MAX_IMAGES = 100;

// Estructura de píxel RGB
typedef struct { unsigned char b, g, r; } Pixel;

// Leer cabecera BMP y obtener ancho/alto
void leerHeader(FILE *in) {
    if (fread(header, sizeof(unsigned char), 54, in) != 54) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    ancho = *(int *)&header[18];
    alto  = *(int *)&header[22];
}

void crearCarpeta(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) != 0) {
            perror("[ERROR] mkdir");
            exit(EXIT_FAILURE);
        }
    }
}

// Escribir BMP con buffer de píxeles
void writeBMP(int img, const char *suffix, Pixel *buf) {
    char oname[128];
    snprintf(oname, sizeof(oname), "salidas/%06d_%s_%d.bmp", img, suffix, KERNEL_SIZE);
    FILE *fout = fopen(oname, "wb");
    if (!fout) {
        fprintf(stderr, "[ERROR] No se puede crear '%s'\n", oname);
        return;
    }
    fwrite(header, sizeof(unsigned char), 54, fout);
    size_t npix = (size_t)ancho * (size_t)alto;
    fwrite(buf, sizeof(Pixel), npix, fout);
    fclose(fout);
    total_escrituras += 3 * (long)npix;
    printf("[LOG] Imagen %06d: '%s' escrita (Escrituras=%ld)\n", img, oname, total_escrituras);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <KERNEL_SIZE> <MAX_IMAGES>\n", argv[0]);
        return EXIT_FAILURE;
    }
    KERNEL_SIZE = atoi(argv[1]);
    MAX_IMAGES  = atoi(argv[2]);
    printf("[LOG] Inicio: KERNEL_SIZE=%d, MAX_IMAGES=%d\n", KERNEL_SIZE, MAX_IMAGES);

    // Leer dimensiones de la primera imagen
    char tmp_name[128];
    snprintf(tmp_name, sizeof(tmp_name), "imagenes_reto/imagenes_bmp_final/000001.bmp");
    FILE *tmpf = fopen(tmp_name, "rb");
    if (!tmpf) { perror("[ERROR] Abrir primera BMP"); return EXIT_FAILURE; }
    leerHeader(tmpf);
    fclose(tmpf);
    printf("[LOG] Dimensiones detectadas: ancho=%d, alto=%d\n", ancho, alto);

    // Reservar buffers contiguos
    size_t npix = (size_t)ancho * (size_t)alto;
    Pixel *buf_orig      = malloc(npix * sizeof(Pixel));
    Pixel *buf_gris      = malloc(npix * sizeof(Pixel));
    Pixel *buf_espH      = malloc(npix * sizeof(Pixel));
    Pixel *buf_espV      = malloc(npix * sizeof(Pixel));
    Pixel *buf_espH_gris = malloc(npix * sizeof(Pixel));
    Pixel *buf_espV_gris = malloc(npix * sizeof(Pixel));
    Pixel *buf_tmp       = malloc(npix * sizeof(Pixel));
    Pixel *buf_blur      = malloc(npix * sizeof(Pixel));
    if (!buf_orig || !buf_gris || !buf_espH || !buf_espV || !buf_espH_gris || !buf_espV_gris || !buf_tmp || !buf_blur) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        return EXIT_FAILURE;
    }

    crearCarpeta("salidas");
    double t0 = omp_get_wtime();

    for (int img = 1; img <= MAX_IMAGES; img++) {
        printf("[LOG] Procesando imagen %06d...\n", img);
        char iname[128];
        snprintf(iname, sizeof(iname), "imagenes_reto/imagenes_bmp_final/%06d.bmp", img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", iname); continue; }
        leerHeader(fin);

        // Leer todos los píxeles de una vez
        for (size_t i = 0; i < npix; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[ERROR] Fallo al leer píxel %zu de imagen %06d", i, img);
                break;
            }
            buf_orig[i].b = rgb[0];
            buf_orig[i].g = rgb[1];
            buf_orig[i].r = rgb[2];
            total_lecturas += 3;
        }
        fclose(fin);
        printf("[LOG] Imagen %06d: Lecturas=%ld\n", img, total_lecturas);

        // 1) Escala de grises
        printf("[LOG] Imagen %06d: Escala de grises\n", img);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r + 0.72f * buf_orig[i].g + 0.07f * buf_orig[i].b);
            buf_gris[i].r = buf_gris[i].g = buf_gris[i].b = lum;
        }

        // 2) Espejos color
        printf("[LOG] Imagen %06d: Espejo horizontal y vertical (color)\n", img);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                size_t idx = (size_t)y * ancho + x;
                buf_espH[idx] = buf_orig[(size_t)y * ancho + (ancho - 1 - x)];
                buf_espV[idx] = buf_orig[(size_t)(alto - 1 - y) * ancho + x];
            }
        }

        // 3) Espejos gris
        printf("[LOG] Imagen %06d: Espejo horizontal y vertical (gris)\n", img);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            int y = i / ancho, x = i % ancho;
            buf_espH_gris[i] = buf_gris[(size_t)y * ancho + (ancho - 1 - x)];
            buf_espV_gris[i] = buf_gris[(size_t)(alto - 1 - y) * ancho + x];
        }

        // 4) Desenfoque separable
        printf("[LOG] Imagen %06d: Desenfoque separable (K=%d)\n", img, KERNEL_SIZE);
        int k = KERNEL_SIZE / 2;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                int sr=0, sg=0, sb=0, cnt=0;
                for (int dx = -k; dx <= k; dx++) {
                    int xx = x + dx;
                    if (xx >= 0 && xx < ancho) {
                        Pixel *p = &buf_orig[(size_t)y * ancho + xx];
                        sr += p->r; sg += p->g; sb += p->b; cnt++;
                    }
                }
                buf_tmp[(size_t)y * ancho + x].r = sr / cnt;
                buf_tmp[(size_t)y * ancho + x].g = sg / cnt;
                buf_tmp[(size_t)y * ancho + x].b = sb / cnt;
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                int sr=0, sg=0, sb=0, cnt=0;
                for (int dy = -k; dy <= k; dy++) {
                    int yy = y + dy;
                    if (yy >= 0 && yy < alto) {
                        Pixel *p = &buf_tmp[(size_t)yy * ancho + x];
                        sr += p->r; sg += p->g; sb += p->b; cnt++;
                    }
                }
                buf_blur[(size_t)y * ancho + x].r = sr / cnt;
                buf_blur[(size_t)y * ancho + x].g = sg / cnt;
                buf_blur[(size_t)y * ancho + x].b = sb / cnt;
            }
        }

        // Escribir salidas
        printf("[LOG] Imagen %06d: Escribiendo resultados...\n", img);
        writeBMP(img, "gris",      buf_gris);
        writeBMP(img, "esp_h",     buf_espH);
        writeBMP(img, "esp_v",     buf_espV);
        writeBMP(img, "esp_h_gris",buf_espH_gris);
        writeBMP(img, "esp_v_gris",buf_espV_gris);
        writeBMP(img, "blur",      buf_blur);

        all_lecturas   += total_lecturas;
        all_escrituras += total_escrituras;
        printf("[LOG] Imagen %06d: Lecturas=%ld, Escrituras=%ld\n", img, total_lecturas, total_escrituras);
        total_lecturas = total_escrituras = 0;
    }

    double t1 = omp_get_wtime();
    double tiempo = t1 - t0;

    long instr_mem = (all_lecturas + all_escrituras) * 20;
    long instr_pix = (long)ancho * alto * 6 * MAX_IMAGES;
    double mips    = ((double)(instr_mem + instr_pix) / tiempo) / 1e6;

    printf("[LOG] Fin: Tiempo=%.2f s, MIPS=%.4f\n", tiempo, mips);

    free(buf_orig);
    free(buf_gris);
    free(buf_espH);
    free(buf_espV);
    free(buf_espH_gris);
    free(buf_espV_gris);
    free(buf_tmp);
    free(buf_blur);

    return EXIT_SUCCESS;
}
