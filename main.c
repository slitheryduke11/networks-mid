#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "bmp_utils.h"

#define TASK_REQUEST 1
#define TASK_ASSIGNMENT 2
#define NO_MORE_TASKS 3

static long total_reads = 0, total_writes = 0;
static long all_reads = 0, all_writes = 0;

static int KERNEL_SIZE = 55;

char **get_filenames_from_dir(const char *dirname, int *count) {
    DIR *dir;
    struct dirent *entry;
    char **filenames = NULL;
    int capacity = 10;
    *count = 0;

    dir = opendir(dirname);
    if (!dir) {
        perror("[ERROR] No se pudo abrir el directorio");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    filenames = malloc(capacity * sizeof(char *));
    if (!filenames) {
        perror("[ERROR] malloc filenames");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".bmp")) {
            if (*count >= capacity) {
                capacity *= 2;
                filenames = realloc(filenames, capacity * sizeof(char *));
                if (!filenames) {
                    perror("[ERROR] realloc filenames");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }
            char fullpath[512];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", dirname, entry->d_name);
            filenames[*count] = strdup(fullpath);
            if (!filenames[*count]) {
                perror("[ERROR] strdup filename");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            (*count)++;
        }
    }
    closedir(dir);
    return filenames;
}

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &hostname_len);
    omp_set_num_threads(4);
    printf("[RANK %d] Usando 4 threads por proceso\n", rank);
    printf("[RANK %d] Ejecutando en host: %s\n", rank, hostname);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <KERNEL_SIZE> <DIRECTORIO_IMAGENES>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    KERNEL_SIZE = atoi(argv[1]);
    char *image_dir = argv[2];

    char **image_files = NULL;
    int total_images = 0;

    if (rank == 0) {
        image_files = get_filenames_from_dir(image_dir, &total_images);
        printf("[LOG] Encontradas %d imágenes en el directorio %s\n", total_images, image_dir);
    }

    MPI_Bcast(&total_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char *filenames_buffer = malloc((size_t)total_images * 512);
    if (!filenames_buffer) {
        fprintf(stderr, "[RANK %d] Error malloc filenames_buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        for (int i = 0; i < total_images; i++) {
            strncpy(&filenames_buffer[i * 512], image_files[i], 512);
            free(image_files[i]);
        }
        free(image_files);
    }

    MPI_Bcast(filenames_buffer, total_images * 512, MPI_CHAR, 0, MPI_COMM_WORLD);

    FILE *log = NULL;
    if (rank == 0) {
        log = fopen("estadisticas.txt", "w");
        if (!log) { perror("[ERROR] stats file"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    }

    FILE *tmpf = fopen(&filenames_buffer[0], "rb");
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

        long total_leidas = (long)width_local * height_local * total_images;
        long total_escritas = total_leidas * 6;
        long total_operaciones = total_leidas + total_escritas;
        long total_instrucciones = total_operaciones * 20;

        double pixeles_por_segundo = total_escritas / total_time;
        double mips = (double)total_instrucciones / (1e6 * total_time);

        // Guardar en archivo
        fprintf(log, "Tiempo total maestro: %.2fs\n", total_time);
        fprintf(log, "Total de localidades leídas (entrada): %ld\n", total_leidas);
        fprintf(log, "Total de localidades escritas (salidas): %ld\n", total_escritas);
        fprintf(log, "Pixeles procesados por segundo: %.3e\n", pixeles_por_segundo);
        fprintf(log, "Total instrucciones estimadas (ensamblador): %ld\n", total_instrucciones);
        fprintf(log, "Rendimiento estimado: %.3f MIPS\n", mips);

        // Mostrar también en consola
        printf("[LOG] Tiempo total de ejecución: %.2fs\n", total_time);
        printf("[LOG] Total leídas: %ld, escritas: %ld\n", total_leidas, total_escritas);
        printf("[LOG] Rendimiento: %.3e pixeles/seg\n", pixeles_por_segundo);
        printf("[LOG] Instrucciones estimadas: %ld, MIPS: %.3f\n", total_instrucciones, mips);


        fclose(log);

    } else {
        while (1) {
            int dummy = 0;
            int task_id;
            MPI_Send(&dummy, 1, MPI_INT, 0, TASK_REQUEST, MPI_COMM_WORLD);

            MPI_Status status;
            MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == NO_MORE_TASKS) break;

            const char *filename = &filenames_buffer[task_id * 512];
            printf("[RANK %d] Procesando imagen %d: %s\n", rank, task_id, filename);

            Pixel *buf_orig = malloc(npix * sizeof(Pixel));
            Pixel *buf_gray = malloc(npix * sizeof(Pixel));
            Pixel *buf_tmp = malloc(npix * sizeof(Pixel));
            Pixel *buf_blur = malloc(npix * sizeof(Pixel));
            Pixel *buf_hmirror = malloc(npix * sizeof(Pixel));
            Pixel *buf_vmirror = malloc(npix * sizeof(Pixel));
            Pixel *buf_hgray = malloc(npix * sizeof(Pixel));
            Pixel *buf_vgray = malloc(npix * sizeof(Pixel));

            if (!buf_orig || !buf_gray || !buf_tmp || !buf_blur ||
                !buf_hmirror || !buf_vmirror || !buf_hgray || !buf_vgray) {
                fprintf(stderr, "[RANK %d] Error malloc buffers en imagen %d\n", rank, task_id);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            FILE *fin = fopen(filename, "rb");
            if (!fin) {
                fprintf(stderr, "[RANK %d] [ERROR] No se puede abrir %s\n", rank, filename);
                free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
                free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
                continue;
            }

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
            printf("[RANK %d] Terminó imagen %d\n", rank, task_id);

            free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
            free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
        }
    }
    free(filenames_buffer);
    MPI_Finalize();
    return EXIT_SUCCESS;
}