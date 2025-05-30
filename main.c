#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>
#include "bmp_utils.h"

// Contadores de operaciones
static long total_reads = 0, total_writes = 0;
static long all_reads = 0, all_writes = 0;

static int KERNEL_SIZE = 55;

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 || argc > 600) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <KERNEL_SIZE> <IMAGEN1> [IMAGEN2 ... IMAGEN_N]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    KERNEL_SIZE = atoi(argv[1]);
    int total_images = argc - 2;

    if (rank == 0)
        printf("[LOG] Inicio (MPI ranks=%d): KERNEL_SIZE=%d, Total Imágenes=%d\n", size, KERNEL_SIZE, total_images);

    FILE *log = NULL;
    if (rank == 0) {
        log = fopen("estadisticas.txt", "w");
        if (!log) { perror("[ERROR] stats file"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    }

    FILE *tmpf = fopen(argv[2], "rb");
    if (!tmpf) { perror("[ERROR] Abrir imagen inicial"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    readHeader(tmpf);
    fclose(tmpf);
    if (rank == 0)
        printf("[LOG] Dimensiones detectadas: width=%d, height=%d\n", width, height);

    size_t npix = (size_t)width * height;
    Pixel *buf_orig   = malloc(npix * sizeof(Pixel));
    Pixel *buf_gray   = malloc(npix * sizeof(Pixel));
    Pixel *buf_tmp    = malloc(npix * sizeof(Pixel));
    Pixel *buf_blur   = malloc(npix * sizeof(Pixel));
    Pixel *buf_hmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_vmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_hgray   = malloc(npix * sizeof(Pixel));
    Pixel *buf_vgray   = malloc(npix * sizeof(Pixel));

    if (!buf_orig || !buf_gray || !buf_tmp || !buf_blur || !buf_hmirror || !buf_vmirror || !buf_hgray || !buf_vgray) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) createFolder("salidas");
    MPI_Barrier(MPI_COMM_WORLD);

    double t0_global = omp_get_wtime();
    double t_total_read = 0.0, t_total_gray = 0.0, t_total_mirror = 0.0;
    double t_total_blur = 0.0, t_total_write = 0.0;

    for (int i = rank; i < total_images; i += size) {
        const char *filename = argv[2 + i];
        total_reads = total_writes = 0;
        double img_start = omp_get_wtime();

        FILE *fin = fopen(filename, "rb");
        if (!fin) {
            fprintf(stderr, "[RANK %d] [ERROR] No se puede abrir %s\n", rank, filename);
            continue;
        }
        readHeader(fin);

        double t0 = omp_get_wtime();
        for (size_t j = 0; j < npix; j++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[RANK %d] [ERROR] Fallo al leer píxel %zu\n", rank, j);
                break;
            }
            buf_orig[j].b = rgb[0];
            buf_orig[j].g = rgb[1];
            buf_orig[j].r = rgb[2];
            total_reads += 3;
        }
        fclose(fin);
        double t1 = omp_get_wtime();
        double t_read = t1 - t0;
        t_total_read += t_read;

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < npix; j++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[j].r + 0.72f * buf_orig[j].g + 0.07f * buf_orig[j].b);
            buf_gray[j].r = buf_gray[j].g = buf_gray[j].b = lum;
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
        for (size_t j = 0; j < npix; j++) {
            int y = j / width, x = j % width;
            buf_hgray[j] = buf_gray[(size_t)y * width + (width - 1 - x)];
            buf_vgray[j] = buf_gray[(size_t)(height - 1 - y) * width + x];
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

        int img_id = i + 2;
        t0 = omp_get_wtime();
        writeBMP(img_id, "gris", buf_gray, KERNEL_SIZE);
        writeBMP(img_id, "esp_h", buf_hmirror, KERNEL_SIZE);
        writeBMP(img_id, "esp_v", buf_vmirror, KERNEL_SIZE);
        writeBMP(img_id, "esp_h_gris", buf_hgray, KERNEL_SIZE);
        writeBMP(img_id, "esp_v_gris", buf_vgray, KERNEL_SIZE);
        writeBMP(img_id, "blur", buf_blur, KERNEL_SIZE);
        t1 = omp_get_wtime();
        double t_write = t1 - t0;
        t_total_write += t_write;

        double img_time = omp_get_wtime() - img_start;
        double mbytes_per_sec = ((double)npix * sizeof(Pixel)) / img_time / 1e6;

        if (log && rank == 0) {
            fprintf(log, "Img %s: read=%.4fs, gray=%.4fs, mirror=%.4fs, blur=%.4fs, write=%.4fs, total=%.4fs, MB/s=%.2f\n",
                    filename, t_read, t_gray, t_mirror, t_blur, t_write, img_time, mbytes_per_sec);
        }
        all_reads += total_reads;
        all_writes += total_writes;
    }

    double t1_global = omp_get_wtime();
    double total_time = t1_global - t0_global;
    long instr_mem = width * height * 3 * 20;
    double mips = instr_mem / total_time / 1e6;
    double avg_mbps = (double)(width * height * sizeof(Pixel) * total_images) / total_time / 1e6;

    printf("[RANK %d] Tiempo total: %.2fs, MIPS=%.4f, Hilos utilizados (OpenMP): %d\n", rank, total_time, mips, omp_get_max_threads());

    if (log && rank == 0) {
        fprintf(log, "Total=%.2fs, MIPS=%.4f, Promedio MB/s=%.2f\n", total_time, mips, avg_mbps);
        fclose(log);
    }

    free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
    free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
