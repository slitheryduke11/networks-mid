#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>
#include "bmp_utils.h"

#define TASK_REQUEST 1
#define TASK_ASSIGNMENT 2
#define NO_MORE_TASKS 3

static long total_reads = 0, total_writes = 0;
static long all_reads = 0, all_writes = 0;

static int KERNEL_SIZE = 55;

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &hostname_len);

    printf("[RANK %d] Ejecutando en host: %s\n", rank, hostname);

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

    int width_local = width, height_local = height;
    size_t npix = (size_t)width * height;

    if (rank == 0)
        printf("[LOG] Dimensiones detectadas: width=%d, height=%d\n", width, height);

    if (rank == 0) createFolder("salidas");
    MPI_Barrier(MPI_COMM_WORLD);

    double t0_global = omp_get_wtime();

    if (rank == 0) {
        // Maestro
        int next_task = 0;
        int dummy;
        int active_workers = size - 1;
        MPI_Status status;

        while (active_workers > 0) {
            MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TASK_REQUEST, MPI_COMM_WORLD, &status);
            int worker = status.MPI_SOURCE;

            if (next_task < total_images) {
                MPI_Send(&next_task, 1, MPI_INT, worker, TASK_ASSIGNMENT, MPI_COMM_WORLD);
                next_task++;
            } else {
                MPI_Send(&dummy, 1, MPI_INT, worker, NO_MORE_TASKS, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        double total_time = omp_get_wtime() - t0_global;
        fprintf(log, "Tiempo total maestro: %.2fs\n", total_time);
        fclose(log);

    } else {
        // Trabajador
        while (1) {
            int dummy = 0;
            int task_id;
            MPI_Send(&dummy, 1, MPI_INT, 0, TASK_REQUEST, MPI_COMM_WORLD);

            MPI_Status status;
            MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == NO_MORE_TASKS) break;

            const char *filename = argv[2 + task_id];

            Pixel *buf_orig = malloc(npix * sizeof(Pixel));
            Pixel *buf_gray = malloc(npix * sizeof(Pixel));
            Pixel *buf_tmp = malloc(npix * sizeof(Pixel));
            Pixel *buf_blur = malloc(npix * sizeof(Pixel));
            Pixel *buf_hmirror = malloc(npix * sizeof(Pixel));
            Pixel *buf_vmirror = malloc(npix * sizeof(Pixel));
            Pixel *buf_hgray = malloc(npix * sizeof(Pixel));
            Pixel *buf_vgray = malloc(npix * sizeof(Pixel));

            FILE *fin = fopen(filename, "rb");
            if (!fin) { fprintf(stderr, "[RANK %d] [ERROR] No se puede abrir %s\n", rank, filename); continue; }
            readHeader(fin);

            for (size_t j = 0; j < npix; j++) {
                unsigned char rgb[3];
                fread(rgb, 1, 3, fin);
                buf_orig[j].b = rgb[0];
                buf_orig[j].g = rgb[1];
                buf_orig[j].r = rgb[2];
            }
            fclose(fin);

            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < npix; j++) {
                unsigned char lum = (unsigned char)(0.21f * buf_orig[j].r + 0.72f * buf_orig[j].g + 0.07f * buf_orig[j].b);
                buf_gray[j].r = buf_gray[j].g = buf_gray[j].b = lum;
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 0; y < height_local; y++) {
                for (int x = 0; x < width_local; x++) {
                    size_t idx = (size_t)y * width_local + x;
                    buf_hmirror[idx] = buf_orig[(size_t)y * width_local + (width_local - 1 - x)];
                    buf_vmirror[idx] = buf_orig[(size_t)(height_local - 1 - y) * width_local + x];
                }
            }

            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < npix; j++) {
                int y = j / width_local, x = j % width_local;
                buf_hgray[j] = buf_gray[(size_t)y * width_local + (width_local - 1 - x)];
                buf_vgray[j] = buf_gray[(size_t)(height_local - 1 - y) * width_local + x];
            }

            int k = KERNEL_SIZE / 2;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 0; y < height_local; y++) {
                for (int x = 0; x < width_local; x++) {
                    int sr = 0, sg = 0, sb = 0, cnt = 0;
                    for (int d = -k; d <= k; d++) {
                        int xx = x + d;
                        if (xx >= 0 && xx < width_local) {
                            Pixel *p = &buf_orig[(size_t)y * width_local + xx];
                            sr += p->r; sg += p->g; sb += p->b; cnt++;
                        }
                    }
                    buf_tmp[(size_t)y * width_local + x].r = sr / cnt;
                    buf_tmp[(size_t)y * width_local + x].g = sg / cnt;
                    buf_tmp[(size_t)y * width_local + x].b = sb / cnt;
                }
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 0; y < height_local; y++) {
                for (int x = 0; x < width_local; x++) {
                    int sr = 0, sg = 0, sb = 0, cnt = 0;
                    for (int d = -k; d <= k; d++) {
                        int yy = y + d;
                        if (yy >= 0 && yy < height_local) {
                            Pixel *p = &buf_tmp[(size_t)yy * width_local + x];
                            sr += p->r; sg += p->g; sb += p->b; cnt++;
                        }
                    }
                    buf_blur[(size_t)y * width_local + x].r = sr / cnt;
                    buf_blur[(size_t)y * width_local + x].g = sg / cnt;
                    buf_blur[(size_t)y * width_local + x].b = sb / cnt;
                }
            }

            int img_id = task_id + 2;
            writeBMP(img_id, "gris", buf_gray, KERNEL_SIZE);
            writeBMP(img_id, "esp_h", buf_hmirror, KERNEL_SIZE);
            writeBMP(img_id, "esp_v", buf_vmirror, KERNEL_SIZE);
            writeBMP(img_id, "esp_h_gris", buf_hgray, KERNEL_SIZE);
            writeBMP(img_id, "esp_v_gris", buf_vgray, KERNEL_SIZE);
            writeBMP(img_id, "blur", buf_blur, KERNEL_SIZE);

            free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
            free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
