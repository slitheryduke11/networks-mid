#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>
#include "bmp_utils.h"

#define TASK_REQUEST      1
#define TASK_ASSIGNMENT   2
#define NO_MORE_TASKS     3
#define HEARTBEAT_TAG     4     

static int KERNEL_SIZE = 55;


// Estructura para pasar parámetros al hilo de heartbeat en workers
typedef struct {
    int rank;
    int master_rank;
    volatile int *keep_running;  // bandera para detener el hilo
} hb_args_t;

// Hilo que, en cada worker, envía un heartbeat al maestro cada 2 s
void *heartbeat_thread(void *arg) {
    hb_args_t *a = (hb_args_t *) arg;
    int rank = a->rank;
    int master = a->master_rank;
    int rc;

    while (*(a->keep_running)) {
        rc = MPI_Send(&rank, 1, MPI_INT, master, HEARTBEAT_TAG, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS) {
            break;
        }
        sleep(1); 
    }
    return NULL;
}


// Recolecta todos los nombres de archivos .bmp en un directorio
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

    // Establecemos handler para que MPI_ERRORS_RETURN funcione
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    omp_set_num_threads(4);
    printf("[RANK %d] Usando 4 threads por proceso en host %s\n", rank, hostname);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <KERNEL_SIZE> <DIRECTORIO_IMAGENES>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    KERNEL_SIZE = atoi(argv[1]);
    char *image_dir = argv[2];

    if (rank == 0) {
        int total_images = 0;
        char **image_files = get_filenames_from_dir(image_dir, &total_images);
        printf("[MAESTRO] Encontradas %d imágenes en %s\n", total_images, image_dir);

        // Preparamos buffer de nombres (cada nombre 512 bytes)
        char *filenames_buffer = malloc((size_t)total_images * 512);
        if (!filenames_buffer) {
            fprintf(stderr, "[MAESTRO] Error malloc filenames_buffer\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < total_images; i++) {
            strncpy(&filenames_buffer[i * 512], image_files[i], 512);
            free(image_files[i]);
        }
        free(image_files);
        MPI_Bcast(&total_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(filenames_buffer, total_images * 512, MPI_CHAR, 0, MPI_COMM_WORLD);

        FILE *log = fopen("estadisticas.txt", "w");
        if (!log) {
            perror("[MAESTRO] Error abrir estadisticas.txt");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        FILE *tmpf = fopen(&filenames_buffer[0], "rb");
        if (!tmpf) {
            perror("[MAESTRO] Error abrir imagen inicial");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        readHeader(tmpf);
        fclose(tmpf);
        int width_local = width, height_local = height;
        size_t npix = (size_t)width * height;
        printf("[MAESTRO] Dimensiones: width=%d, height=%d, npix=%zu\n",
               width_local, height_local, npix);

        createFolder("salidas");
        MPI_Barrier(MPI_COMM_WORLD);
        const double HEARTBEAT_INTERVAL = 60.0;  
        const int    MAX_MISSED         = 3;
        double *last_heartbeat = calloc(size, sizeof(double));
        if (!last_heartbeat) {
            fprintf(stderr, "[MAESTRO] Error malloc last_heartbeat\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int *missed = calloc(size, sizeof(int));
        if (!missed) {
            fprintf(stderr, "[MAESTRO] Error malloc missed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        double now = MPI_Wtime();
        for (int i = 1; i < size; i++) {
            last_heartbeat[i] = now;
            missed[i]  = 0;
        }
        int *assigned_task = malloc(size * sizeof(int));
        if (!assigned_task) {
            fprintf(stderr, "[MAESTRO] Error malloc assigned_task\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < size; i++) {
            assigned_task[i] = -1;
        }
        int *alive = malloc(size * sizeof(int));
        if (!alive) {
            fprintf(stderr, "[MAESTRO] Error malloc alive\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < size; i++) {
            alive[i] = (i == 0 ? 0 : 1); 
        }

        int *task_queue = malloc(sizeof(int) * (2 * total_images));
        if (!task_queue) {
            fprintf(stderr, "[MAESTRO] Error malloc task_queue\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int queue_head = 0, queue_tail = total_images;
        for (int i = 0; i < total_images; i++) {
            task_queue[i] = i;
        }
        int active_workers = size - 1;
        MPI_Status status;
        double start_time = MPI_Wtime();
        while (active_workers > 0) {
            int flag;

            MPI_Iprobe(MPI_ANY_SOURCE, HEARTBEAT_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                int hb_rank;
                MPI_Recv(&hb_rank, 1, MPI_INT, status.MPI_SOURCE,
                         HEARTBEAT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (alive[hb_rank]) {
                    last_heartbeat[hb_rank] = MPI_Wtime();
                    missed[hb_rank]  = 0;  

                    printf("[MAESTRO] Recibido heartbeat de worker %d (missed[%d]=0)\n", hb_rank, hb_rank);
                    fflush(stdout);
                }
            }

            double ahora = MPI_Wtime();

            for (int w = 1; w < size; w++) {
                if (alive[w] && assigned_task[w] >= 0) {
                    double dt = ahora - last_heartbeat[w];

                    if (dt > (missed[w] + 1) * HEARTBEAT_INTERVAL) {
                        missed[w]++;
                        printf("[MAESTRO] Worker %d: latido tardío #%d (dt=%.1f s)\n",
                               w, missed[w], dt);
                        fflush(stdout);
                    }

                    if (missed[w] >= MAX_MISSED) {
                        printf("[MAESTRO] Worker %d marcado como MUERTO (missed=%d). Reasignando tarea %d.\n",
                               w, missed[w], assigned_task[w]);
                        fflush(stdout);

                        int tarea_caida = assigned_task[w];
                        task_queue[ queue_tail++ ] = tarea_caida;

                        assigned_task[w] = -1;
                        alive[w]         = 0;
                        active_workers--;
                    }
                }
            }

            MPI_Iprobe(MPI_ANY_SOURCE, TASK_REQUEST, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                int src = status.MPI_SOURCE;

                if (!alive[src]) {
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, src,
                             TASK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    continue;
                }

                int dummy;
                int rc_recv = MPI_Recv(&dummy, 1, MPI_INT, src,
                                       TASK_REQUEST, MPI_COMM_WORLD, &status);
                if (rc_recv != MPI_SUCCESS) {
                    // Este worker murió justo en la petición:
                    if (assigned_task[src] >= 0) {
                        task_queue[queue_tail++] = assigned_task[src];
                    }
                    assigned_task[src] = -1;
                    alive[src] = 0;
                    active_workers--;
                    continue;

                }

                // Si hay tareas pendientes, se la asignamos:
                if (queue_head < queue_tail) {
                    int tarea_id = task_queue[queue_head++];
                    assigned_task[src] = tarea_id;

                    int rc_send = MPI_Send(&tarea_id, 1, MPI_INT, src,
                                           TASK_ASSIGNMENT, MPI_COMM_WORLD);
                    if (rc_send != MPI_SUCCESS) {
                        // Si falló el envío, ese worker murió justo antes de recibir:
                        printf("[MAESTRO] Worker %d murió antes de recibir tarea %d.\n", src, tarea_id);
                        fflush(stdout);

                        assigned_task[src] = -1;
                        task_queue[queue_tail++] = tarea_id;
                        alive[src] = 0;
                        active_workers--;
                    } else {
                        printf("[MAESTRO] Asignada tarea %d a worker %d\n", tarea_id, src);
                        fflush(stdout);
                    }
                } else {
                    // No quedan tareas pendientes; enviamos NO_MORE_TASKS
                    int rc_send = MPI_Send(&dummy, 1, MPI_INT, src,
                                           NO_MORE_TASKS, MPI_COMM_WORLD);
                    if (rc_send != MPI_SUCCESS) {
                        if (assigned_task[src] >= 0) {
                            task_queue[queue_tail++] = assigned_task[src];
                        }
                        assigned_task[src] = -1;
                        alive[src] = 0;
                    } else {
                        assigned_task[src] = -1;
                        alive[src] = 0;
                        active_workers--;
                        printf("[MAESTRO] Worker %d recibió NO_MORE_TASKS y finaliza.\n", src);
                        fflush(stdout);
                    }
                }
            }

            // 7.4) Pequeño sleep para no saturar CPU
            usleep(10000);
        }
        // Al terminar, volcamos métricas a ‘estadisticas.txt’
        double total_time = MPI_Wtime() - start_time;
        long total_leidas = (long)width_local * height_local * total_images;
        long total_escritas = total_leidas * 6;
        long total_operaciones = total_leidas + total_escritas;
        long total_instrucciones = total_operaciones * 20;

        double pixeles_por_segundo = total_escritas / total_time;
        double mips = (double)total_instrucciones / (1e6 * total_time);

        fprintf(log, "Tiempo total maestro: %.2fs\n", total_time);
        fprintf(log, "Total de localidades leídas (entrada): %ld\n", total_leidas);
        fprintf(log, "Total de localidades escritas (salidas): %ld\n", total_escritas);
        fprintf(log, "Pixeles procesados por segundo: %.3e\n", pixeles_por_segundo);
        fprintf(log, "Total instrucciones estimadas (ensamblador): %ld\n", total_instrucciones);
        fprintf(log, "Rendimiento estimado: %.3f MIPS\n", mips);
        fclose(log);

        printf("[MAESTRO] Todos los workers terminaron; entrando en barrera final...\n");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        free(filenames_buffer);
        free(last_heartbeat);
        free(missed);
        free(assigned_task);
        free(task_queue);
        free(alive);
        printf("[MAESTRO] Llamando a MPI_Finalize() y saliendo.\n");
        fflush(stdout);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    else {
        int total_images;
        printf("[WORKER %d] Arrancando. Esperando broadcast de total_images...\n", rank);
        fflush(stdout);

        MPI_Bcast(&total_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("[WORKER %d] Recibido total_images = %d\n", rank, total_images);
        fflush(stdout);

        char **image_files = NULL;
        {
            char *filenames_buffer = malloc((size_t)total_images * 512);
            MPI_Bcast(filenames_buffer, total_images * 512, MPI_CHAR, 0, MPI_COMM_WORLD);

            image_files = malloc(total_images * sizeof(char *));
            for (int i = 0; i < total_images; i++) {
                image_files[i] = strdup(&filenames_buffer[i * 512]);
            }
            free(filenames_buffer);
        }

        printf("[WORKER %d] Primera imagen en lista: %s\n",
               rank, image_files[0]);
        fflush(stdout);

        volatile int keep_running = 1;
        pthread_t hb_thread;
        hb_args_t hb_args;
        hb_args.rank = rank;
        hb_args.master_rank = 0;
        hb_args.keep_running = &keep_running;

        if (pthread_create(&hb_thread, NULL, heartbeat_thread, &hb_args) != 0) {
            fprintf(stderr, "[WORKER %d] No se pudo crear hilo de heartbeat\n", rank);
            fflush(stderr);
        } else {
            printf("[WORKER %d] Hilo de heartbeat lanzado.\n", rank);
            fflush(stdout);
        }

        int npix;
        {
            FILE *tmpf2 = fopen(image_files[0], "rb");
            if (!tmpf2) {
                perror("[ERROR Worker] Abrir imagen inicial");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            readHeader(tmpf2);
            fclose(tmpf2);
            npix = width * height;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        printf("[WORKER %d] Entrando en bucle principal de tareas.\n", rank);
        fflush(stdout);

        while (1) {
            int dummy = 0;
            printf("[WORKER %d] Enviando petición de tarea (TASK_REQUEST)...\n", rank);
            fflush(stdout);

            int rc_send = MPI_Send(&dummy, 1, MPI_INT, 0,
                                   TASK_REQUEST, MPI_COMM_WORLD);
            if (rc_send != MPI_SUCCESS) {
                printf("[WORKER %d] El maestro no responde, rc_send=%d. Finalizando.\n", rank, rc_send);
                fflush(stdout);
                break;
            }
            printf("[WORKER %d] Petición de tarea enviada con éxito.\n", rank);
            printf("[WORKER %d] Esperando MPI_Recv (TASK_ASSIGNMENT o NO_MORE_TASKS)...\n", rank);
            fflush(stdout);

            int task_id;
            MPI_Status status2;
            int rc_recv = MPI_Recv(&task_id, 1, MPI_INT, 0,
                                   MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
            if (rc_recv != MPI_SUCCESS) {
                printf("[WORKER %d] No se pudo recibir respuesta del maestro. Saliendo.\n", rank);
                fflush(stdout);
                break;
            }
            if (status2.MPI_TAG == NO_MORE_TASKS) {
                printf("[WORKER %d] Recibido NO_MORE_TASKS. Terminando.\n", rank);
                fflush(stdout);
                break;
            }

            const char *filename = image_files[task_id];
            printf("[WORKER %d] Procesando imagen %d: %s\n", rank, task_id, filename);
            fflush(stdout);

            Pixel *buf_orig    = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_gray    = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_tmp     = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_blur    = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_hmirror = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_vmirror = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_hgray   = malloc((size_t)npix * sizeof(Pixel));
            Pixel *buf_vgray   = malloc((size_t)npix * sizeof(Pixel));
            if (!buf_orig || !buf_gray || !buf_tmp || !buf_blur ||
                !buf_hmirror || !buf_vmirror || !buf_hgray || !buf_vgray) {
                fprintf(stderr, "[WORKER %d] Error malloc buffers en imagen %d\n", rank, task_id);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            FILE *fin = fopen(filename, "rb");
            if (!fin) {
                fprintf(stderr, "[WORKER %d] [ERROR] No se puede abrir %s\n", rank, filename);
                free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
                free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
                continue;  // devolvemos la tarea al maestro cuando detecte caída
            }
            for (size_t j = 0; j < (size_t)npix; j++) {
                unsigned char rgb[3];
                (void) fread(rgb, 1, 3, fin);
                buf_orig[j].b = rgb[0];
                buf_orig[j].g = rgb[1];
                buf_orig[j].r = rgb[2];
            }
            fclose(fin);

            // Convertir a gris
            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < (size_t)npix; j++) {
                unsigned char lum = (unsigned char)(
                    0.21f * buf_orig[j].r +
                    0.72f * buf_orig[j].g +
                    0.07f * buf_orig[j].b
                );
                buf_gray[j].r = buf_gray[j].g = buf_gray[j].b = lum;
            }

            // Espejo horizontal y vertical
            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    size_t idx = (size_t)y * width + x;
                    buf_hmirror[idx] = buf_orig[(size_t)y * width + (width - 1 - x)];
                    buf_vmirror[idx] = buf_orig[(size_t)(height - 1 - y) * width + x];
                }
            }

            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < (size_t)npix; j++) {
                int y = j / width, x = j % width;
                buf_hgray[j] = buf_gray[(size_t)y * width + (width - 1 - x)];
                buf_vgray[j] = buf_gray[(size_t)(height - 1 - y) * width + x];
            }

            // Blur de kernel KERNEL_SIZE × KERNEL_SIZE
            int k = KERNEL_SIZE / 2;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int sr=0, sg=0, sb=0, cnt=0;
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
                    int sr=0, sg=0, sb=0, cnt=0;
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

            // Guardar resultados
            writeBMP(task_id + 2, "gris",      buf_gray,    KERNEL_SIZE);
            writeBMP(task_id + 2, "esp_h",     buf_hmirror, KERNEL_SIZE);
            writeBMP(task_id + 2, "esp_v",     buf_vmirror, KERNEL_SIZE);
            writeBMP(task_id + 2, "esp_h_gris",buf_hgray,   KERNEL_SIZE);
            writeBMP(task_id + 2, "esp_v_gris",buf_vgray,   KERNEL_SIZE);
            writeBMP(task_id + 2, "blur",      buf_blur,    KERNEL_SIZE);

            printf("[WORKER %d] Terminó imagen %d\n", rank, task_id);
            fflush(stdout);

            free(buf_orig); free(buf_gray); free(buf_tmp); free(buf_blur);
            free(buf_hmirror); free(buf_vmirror); free(buf_hgray); free(buf_vgray);
        }

        // Finalmente, indicamos al hilo de heartbeat que termine
        keep_running = 0;
        pthread_join(hb_thread, NULL);

        for (int i = 0; i < total_images; i++) {
            free(image_files[i]);
        }
        free(image_files);

        printf("[WORKER %d] LLegué al final, esperando en barrera para finalizar MPI...\n", rank);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        printf("[WORKER %d] Saliendo (MPI_Finalize).\n", rank);
        fflush(stdout);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
}

