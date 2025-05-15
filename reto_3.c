#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// Variables globales que almacenan el width y height de las imágenes
static int width, height;
// Buffer para leer la cabecera BMP (54 bytes)
static unsigned char header[54];

// Contadores de operaciones de lectura y escritura por imagen y totales
static long total_reads = 0, total_writes = 0;
static long all_reads = 0, all_writes = 0;

// Parámetros configurables: tamaño de kernel de blur y número de imágenes
static int KERNEL_SIZE = 55;
static int MAX_IMAGES = 100;

// Definición de un píxel RGB (3 bytes)
typedef struct { unsigned char b, g, r; } Pixel;

// fread la cabecera BMP y extrae width y height
void readHeader(FILE *in) {
    // Leer 54 bytes de cabecera
    if (fread(header, sizeof(header), 1, in) != 1) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    // Extraer width y height de los bytes apropiados
    width = *(int *)&header[18];
    height  = *(int *)&header[22];
}

// Crear carpeta 'salidas' si no existe
void createFolder(const char *path) {
    struct stat st;
    // stat falla si no existe; entonces creamos carpeta
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) != 0) {
            perror("[ERROR] mkdir");
            exit(EXIT_FAILURE);
        }
    }
}

// Escribe un BMP a disco con un sufijo en el nombre
void writeBMP(int img, const char *suffix, Pixel *buf) {
    char oname[128];
    // Formato: salidas/000001_gris_5.bmp
    snprintf(oname, sizeof(oname), "salidas/%06d_%s_%d.bmp", img, suffix, KERNEL_SIZE);
    FILE *fout = fopen(oname, "wb");
    if (!fout) {
        fprintf(stderr, "[ERROR] No se puede crear '%s'\n", oname);
        return;
    }
    // Escribir cabecera y luego datos de píxeles
    fwrite(header, sizeof(header), 1, fout);
    size_t npix = (size_t)width * height;
    fwrite(buf, sizeof(Pixel), npix, fout);
    fclose(fout);
    // Contar escrituras: 3 bytes por píxel
    total_writes += 3 * (long)npix;
}

// Función principal: controla el flujo completo
int main(int argc, char *argv[]) {
    // Validación de argumentos
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <KERNEL_SIZE> <MAX_IMAGES>\n", argv[0]);
        return EXIT_FAILURE;
    }
    // Leer parámetros de entrada
    KERNEL_SIZE = atoi(argv[1]);
    MAX_IMAGES  = atoi(argv[2]);
    printf("[LOG] Inicio: KERNEL_SIZE=%d, MAX_IMAGES=%d\n", KERNEL_SIZE, MAX_IMAGES);

    // Abrir archivo de estadísticas para salida
    FILE *log = fopen("estadisticas.txt", "w");
    if (!log) { perror("[ERROR] stats file"); return EXIT_FAILURE; }

    // Leer dimensiones de la primera BMP (para reservar buffers)
    char tmp_name[128];
    snprintf(tmp_name, sizeof(tmp_name), "imagenes_reto/imagenes_bmp_final/000001.bmp");
    FILE *tmpf = fopen(tmp_name, "rb");
    if (!tmpf) { perror("[ERROR] Abrir primera BMP"); return EXIT_FAILURE; }
    readHeader(tmpf);
    fclose(tmpf);
    printf("[LOG] Dimensiones detectadas: width=%d, height=%d\n", width, height);

    // Reservar buffers grandes contiguos para todos los efectos
    size_t npix = (size_t)width * height;
    Pixel *buf_orig      = malloc(npix * sizeof(Pixel)); // Imagen original
    Pixel *buf_gray      = malloc(npix * sizeof(Pixel)); // Escala de grises
    Pixel *buf_hmirror      = malloc(npix * sizeof(Pixel)); // Espejo horizontal
    Pixel *buf_vmirror      = malloc(npix * sizeof(Pixel)); // Espejo vertical
    Pixel *buf_hgray = malloc(npix * sizeof(Pixel)); // Espejo gris horizontal
    Pixel *buf_vgray = malloc(npix * sizeof(Pixel)); // Espejo gris vertical
    Pixel *buf_tmp       = malloc(npix * sizeof(Pixel)); // Buffer intermedio blur
    Pixel *buf_blur      = malloc(npix * sizeof(Pixel)); // Resultado blur
    // Verificar malloc
    if (!buf_orig || !buf_gray || !buf_hmirror || !buf_vmirror ||
        !buf_hgray || !buf_vgray || !buf_tmp || !buf_blur) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        return EXIT_FAILURE;
    }

    // Asegurar carpeta de salida
    createFolder("salidas");
    // Marca de tiempo de inicio global
    double t0_global = omp_get_wtime();
    // Acumuladores de tiempo por etapa
    double t_total_read = 0.0, t_total_gray = 0.0, t_total_mirror = 0.0;
    double t_total_blur = 0.0, t_total_write = 0.0;

    // Bucle principal: procesa cada imagen
    for (int img = 1; img <= MAX_IMAGES; img++) {
        total_reads = total_writes = 0;
        double img_start = omp_get_wtime();
        printf("[LOG] Procesando imagen %06d...\n", img);

        // Construir ruta y abrir BMP
        char iname[128];
        snprintf(iname, sizeof(iname), "imagenes_reto/imagenes_bmp_final/%06d.bmp", img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", iname); continue; }
        readHeader(fin);
        double t0 = omp_get_wtime();
        // 1) Lectura de todos los píxeles (un fread de 3 bytes)
        for (size_t i = 0; i < npix; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[ERROR] Fallo al leer píxel %zu de imagen %06d\n", i, img);
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

        // 2) Escala de grises con paralelismo OpenMP
        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r
                                              + 0.72f * buf_orig[i].g
                                              + 0.07f * buf_orig[i].b);
            buf_gray[i].r = buf_gray[i].g = buf_gray[i].b = lum;
        }
        t1 = omp_get_wtime();
        double t_gray = t1 - t0;
        t_total_gray += t_gray;

        // 3) Espejos horizontales y verticales (color)
        t0 = omp_get_wtime();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = (size_t)y * width + x;
                buf_hmirror[idx] = buf_orig[(size_t)y * width + (width - 1 - x)];
                buf_vmirror[idx] = buf_orig[(size_t)(height - 1 - y) * width + x];
            }
        }

        // 4) Espejos horizontales y verticales (gris)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            int y = i / width, x = i % width;
            buf_hgray[i] = buf_gray[(size_t)y * width + (width - 1 - x)];
            buf_vgray[i] = buf_gray[(size_t)(height - 1 - y) * width + x];
        }
        t1 = omp_get_wtime();
        double t_mirror = t1 - t0;
        t_total_mirror += t_mirror;

        // 5) Desenfoque separable (horizontal + vertical)
        int k = KERNEL_SIZE / 2;
        t0 = omp_get_wtime();
        // Pasa horizontal
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
        // Pasa vertical
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
        t1 = omp_get_wtime();
        double t_blur = t1 - t0;
        t_total_blur += t_blur;

        // 6) Guardar resultados finales en archivos BMP
        t0 = omp_get_wtime();
        writeBMP(img, "gris",      buf_gray);
        writeBMP(img, "esp_h",     buf_hmirror);
        writeBMP(img, "esp_v",     buf_vmirror);
        writeBMP(img, "esp_h_gris",buf_hgray);
        writeBMP(img, "esp_v_gris",buf_vgray);
        writeBMP(img, "blur",      buf_blur);
        t1 = omp_get_wtime();
        double t_write = t1 - t0;
        t_total_write += t_write;

        // 7) Métricas por imagen: width de banda y tiempos
        double img_end = omp_get_wtime();
        double img_time = img_end - img_start;
        long bytes = (long)npix * sizeof(Pixel);  // bytes procesados útiles
        double bytes_per_sec = img_time > 0 ? (bytes / img_time) : 0;
        double mbytes_per_sec = bytes_per_sec/1000000;
        fprintf(log, "Img %06d: read=%.4f s,transform gray:%.4fs,transform mirror:%.4fs,transform blur:%.4fs, write=%.4f s, total=%.4f s, mBytes/s=%.2f\n",
                img, t_read, t_gray, t_mirror, t_blur, t_write, img_time, mbytes_per_sec);
        all_reads   += total_reads;
        all_writes += total_writes;
        total_reads = total_writes = 0;
    }

    // Cálculo global de MIPS y cierre de archivos
    double t1_global = omp_get_wtime();
    double tiempo_total = t1_global - t0_global;
    long instr_mem = width * height * 3 * 20;
    double mips    = instr_mem / tiempo_total / 1e6;
    // Calcular promedio Bytes/s global
    size_t total_bytes = (size_t)width * height * sizeof(Pixel) * MAX_IMAGES;
    double avg_bps = tiempo_total > 0 ? ((double)total_bytes / tiempo_total) : 0;
    double avg_mbps = avg_bps/1000000;

    printf("[LOG] Fin: Tiempo=%.2f s, MIPS=%.4f\n", tiempo_total, mips);
    printf("Promedios (s): read=%.4f, gray=%.4f, mirror=%.4f, blur=%.4f, write=%.4f\n",
           t_total_read/MAX_IMAGES, t_total_gray/MAX_IMAGES,
           t_total_mirror/MAX_IMAGES, t_total_blur/MAX_IMAGES, t_total_write/MAX_IMAGES);
    // Agregar promedio Bytes/s al log
    fprintf(log, "Tiempo total: %.2f s, MIPS: %.4f, Promedio MegaBytes/s: %.2f\n",tiempo_total, mips, avg_mbps);

    fclose(log);
    free(buf_orig); free(buf_gray); free(buf_hmirror); free(buf_vmirror);
    free(buf_hgray); free(buf_vgray); free(buf_tmp); free(buf_blur);
    return EXIT_SUCCESS;
}
