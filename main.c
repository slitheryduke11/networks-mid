#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "bmp_utils.h"

// Contadores de operaciones
static long total_reads = 0, total_writes = 0;
static long all_reads = 0, all_writes = 0;

// Parámetros configurables
static int KERNEL_SIZE = 55;
static int MAX_IMAGES = 100;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <KERNEL_SIZE> <MAX_IMAGES>\n", argv[0]);
        return EXIT_FAILURE;
    }

    KERNEL_SIZE = atoi(argv[1]);
    MAX_IMAGES  = atoi(argv[2]);
    printf("[LOG] Inicio: KERNEL_SIZE=%d, MAX_IMAGES=%d\n", KERNEL_SIZE, MAX_IMAGES);

    FILE *log = fopen("estadisticas.txt", "w");
    if (!log) { perror("[ERROR] stats file"); return EXIT_FAILURE; }

    // Ruta actualizada donde están tus imágenes
    const char *base_path = "/Users/hugomr18/Documents/Semestre8/imagenes_reto/imagenes_bmp_final";

    // Leer dimensiones desde 000002.bmp (porque 000001.bmp no existe)
    char tmp_name[256];
    snprintf(tmp_name, sizeof(tmp_name), "%s/000002.bmp", base_path);
    FILE *tmpf = fopen(tmp_name, "rb");
    if (!tmpf) { perror("[ERROR] Abrir imagen inicial"); return EXIT_FAILURE; }
    readHeader(tmpf);
    fclose(tmpf);
    printf("[LOG] Dimensiones detectadas: width=%d, height=%d\n", width, height);

    size_t npix = (size_t)width * height;
    Pixel *buf_orig   = malloc(npix * sizeof(Pixel));
    Pixel *buf_gray   = malloc(npix * sizeof(Pixel));
    Pixel *buf_hmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_vmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_hgray  = malloc(npix * sizeof(Pixel));
    Pixel *buf_vgray  = malloc(npix * sizeof(Pixel));
    Pixel *buf_tmp    = malloc(npix * sizeof(Pixel));
    Pixel *buf_blur   = malloc(npix * sizeof(Pixel));
    if (!buf_orig || !buf_gray || !buf_hmirror || !buf_vmirror ||
        !buf_hgray || !buf_vgray || !buf_tmp || !buf_blur) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        return EXIT_FAILURE;
    }

    createFolder("salidas");
    double t0_global = omp_get_wtime();
    double t_total_read = 0.0, t_total_gray = 0.0, t_total_mirror = 0.0;
    double t_total_blur = 0.0, t_total_write = 0.0;

    for (int img = 2; img < 2 + MAX_IMAGES; img++) {
        total_reads = total_writes = 0;
        double img_start = omp_get_wtime();
        printf("[LOG] Procesando imagen %06d...\n", img);

        char iname[256];
        snprintf(iname, sizeof(iname), "%s/%06d.bmp", base_path, img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", iname); continue; }
        readHeader(fin);
        double t0 = omp_get_wtime();

        for (size_t i = 0; i < npix; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[ERROR] Fallo al leer píxel %zu\n", i);
                break;
            }
            buf_orig[i].b = rgb[0];
            buf_orig[i].g = rgb[1];
            buf_orig[i].r = rgb[2];
            total_reads += 3;
        }
        fclose(fin);
        double t1 = omp_get_wtime();
        double t_read = t1 - t0;
        t_total_read += t_read;

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r + 0.72f * buf_orig[i].g + 0.07f * buf_orig[i].b);
            buf_gray[i].r = buf_gray[i].g = buf_gray[i].b = lum;
        }
        t1 = omp_get_wtime();
        double t_gray = t1 - t0;
        t_total_gray += t_gray;

        t0 = omp_get_wtime();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = (size_t)y * width + x;
                buf_hmirror[idx] = buf_orig[(size_t)y * width + (width - 1 - x)];
                buf_vmirror[idx] = buf_orig[(size_t)(height - 1 - y) * width + x];
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            int y = i / width, x = i % width;
            buf_hgray[i] = buf_gray[(size_t)y * width + (width - 1 - x)];
            buf_vgray[i] = buf_gray[(size_t)(height - 1 - y) * width + x];
        }
        t1 = omp_get_wtime();
        double t_mirror = t1 - t0;
        t_total_mirror += t_mirror;

        int k = KERNEL_SIZE / 2;
        t0 = omp_get_wtime();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sr = 0, sg = 0, sb = 0, cnt = 0;
                for (int d = -k; d <= k; d++) {
                    int xx = x + d;
                    if (xx >= 0 && xx < width) {
                        Pixel *p = &buf_orig[(size_t)y * width + xx];
                        sr += p->r; sg += p->g; sb += p->b; cnt++;
                    }
                }
                buf_tmp[(size_t)y * width + x].r = sr / cnt;
                buf_tmp[(size_t)y * width + x].g = sg / cnt;
                buf_tmp[(size_t)y * width + x].b = sb / cnt;
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sr = 0, sg = 0, sb = 0, cnt = 0;
                for (int d = -k; d <= k; d++) {
                    int yy = y + d;
                    if (yy >= 0 && yy < height) {
                        Pixel *p = &buf_tmp[(size_t)yy * width + x];
                        sr += p->r; sg += p->g; sb += p->b; cnt++;
                    }
                }
                buf_blur[(size_t)y * width + x].r = sr / cnt;
                buf_blur[(size_t)y * width + x].g = sg / cnt;
                buf_blur[(size_t)y * width + x].b = sb / cnt;
            }
        }
        t1 = omp_get_wtime();
        double t_blur = t1 - t0;
        t_total_blur += t_blur;

        t0 = omp_get_wtime();
        writeBMP(img, "gris",      buf_gray, KERNEL_SIZE);
        writeBMP(img, "esp_h",     buf_hmirror, KERNEL_SIZE);
        writeBMP(img, "esp_v",     buf_vmirror, KERNEL_SIZE);
        writeBMP(img, "esp_h_gris",buf_hgray, KERNEL_SIZE);
        writeBMP(img, "esp_v_gris",buf_vgray, KERNEL_SIZE);
        writeBMP(img, "blur",      buf_blur, KERNEL_SIZE);
        t1 = omp_get_wtime();
        double t_write = t1 - t0;
        t_total_write += t_write;

        double img_end = omp_get_wtime();
        double img_time = img_end - img_start;
        double mbytes_per_sec = ((double)npix * sizeof(Pixel)) / img_time / 1e6;
        fprintf(log, "Img %06d: read=%.4f s, gray=%.4f s, mirror=%.4f s, blur=%.4f s, write=%.4f s, total=%.4f s, MB/s=%.2f\n",
                img, t_read, t_gray, t_mirror, t_blur, t_write, img_time, mbytes_per_sec);

        all_reads += total_reads;
        all_writes += total_writes;
    }

    double t1_global = omp_get_wtime();
    double total_time = t1_global - t0_global;
    long instr_mem = width * height * 3 * 20;
    double mips = instr_mem / total_time / 1e6;
    double avg_mbps = (double)(width * height * sizeof(Pixel) * MAX_IMAGES) / total_time / 1e6;

    printf("[LOG] Fin: Tiempo=%.2f s, MIPS=%.4f\n", total_time, mips);
    fprintf(log, "Total=%.2f s, MIPS=%.4f, Promedio MB/s=%.2f\n", total_time, mips, avg_mbps);
    fclose(log);

    free(buf_orig); free(buf_gray); free(buf_hmirror); free(buf_vmirror);
    free(buf_hgray); free(buf_vgray); free(buf_tmp); free(buf_blur);

    return EXIT_SUCCESS;
}
