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

// Parámetros configurables
static int KERNEL_SIZE = 55;
static int MAX_IMAGES = 100;

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <KERNEL_SIZE> <MAX_IMAGES> <RUTA_IMAGENES>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    KERNEL_SIZE = atoi(argv[1]);
    MAX_IMAGES  = atoi(argv[2]);
    const char *base_path = argv[3];

    struct stat st;
    if (stat(base_path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        if (rank == 0)
            fprintf(stderr, "[ERROR] La ruta '%s' no existe o no es un directorio\n", base_path);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0)
        printf("[LOG] Inicio (MPI ranks=%d): KERNEL_SIZE=%d, MAX_IMAGES=%d, RUTA=%s\n", size, KERNEL_SIZE, MAX_IMAGES, base_path);

    FILE *log = NULL;
    if (rank == 0) {
        log = fopen("estadisticas.txt", "w");
        if (!log) { perror("[ERROR] stats file"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    }

    char tmp_name[256];
    snprintf(tmp_name, sizeof(tmp_name), "%s/000002.bmp", base_path);
    FILE *tmpf = fopen(tmp_name, "rb");
    if (!tmpf) { perror("[ERROR] Abrir imagen inicial"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    readHeader(tmpf);
    fclose(tmpf);
    if (rank == 0)
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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) createFolder("salidas");
    MPI_Barrier(MPI_COMM_WORLD);

    double t0_global = omp_get_wtime();

    for (int img = 2 + rank; img < 2 + MAX_IMAGES; img += size) {
        total_reads = total_writes = 0;
        double img_start = omp_get_wtime();
        printf("[RANK %d] Procesando imagen %06d...\n", rank, img);

        char iname[256];
        snprintf(iname, sizeof(iname), "%s/%06d.bmp", base_path, img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[RANK %d] [ERROR] No se puede abrir %s\n", rank, iname); continue; }
        readHeader(fin);
        double t0 = omp_get_wtime();

        for (size_t i = 0; i < npix; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[RANK %d] [ERROR] Fallo al leer píxel %zu\n", rank, i);
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

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r + 0.72f * buf_orig[i].g + 0.07f * buf_orig[i].b);
            buf_gray[i].r = buf_gray[i].g = buf_gray[i].b = lum;
        }
        t1 = omp_get_wtime();
        double t_gray = t1 - t0;

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

        t0 = omp_get_wtime();
        writeBMP(img, "gris",      buf_gray, KERNEL_SIZE);
        writeBMP(img, "esp_h",     buf_hmirror, KERNEL_SIZE);
        writeBMP(img, "esp_v",     buf_vmirror, KERNEL_SIZE);
        writeBMP(img, "esp_h_gris",buf_hgray, KERNEL_SIZE);
        writeBMP(img, "esp_v_gris",buf_vgray, KERNEL_SIZE);
        writeBMP(img, "blur",      buf_blur, KERNEL_SIZE);
        t1 = omp_get_wtime();
        double t_write = t1 - t0;

        double img_time = omp_get_wtime() - img_start;
        double mbytes_per_sec = ((double)npix * sizeof(Pixel)) / img_time / 1e6;

        if (log && rank == 0) {
            fprintf(log, "Img %06d: read=%.4fs, gray=%.4fs, mirror=%.4fs, blur=%.4fs, write=%.4fs, total=%.4fs, MB/s=%.2f\n",
                    img, t_read, t_gray, t_mirror, t_blur, t_write, img_time, mbytes_per_sec);
        }

        all_reads += total_reads;
        all_writes += total_writes;
    }

    if (rank == 0) {
        double total_time = omp_get_wtime() - t0_global;
        long instr_mem = width * height * 3 * 20;
        double mips = instr_mem / total_time / 1e6;
        double avg_mbps = (double)(width * height * sizeof(Pixel) * MAX_IMAGES) / total_time / 1e6;
        printf("[LOG] Fin: Tiempo=%.2fs, MIPS=%.4f\n", total_time, mips);
        fprintf(log, "Total=%.2fs, MIPS=%.4f, Promedio MB/s=%.2f\n", total_time, mips, avg_mbps);
        fclose(log);
    }

    free(buf_orig); free(buf_gray); free(buf_hmirror); free(buf_vmirror);
    free(buf_hgray); free(buf_vgray); free(buf_tmp); free(buf_blur);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
