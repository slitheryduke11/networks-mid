#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// ================================================================
// Programa optimizado de procesamiento de imágenes BMP con OpenMP
// ================================================================

// Variables globales que almacenan el ancho y alto de las imágenes
static int ancho, alto;
// Buffer para leer la cabecera BMP (54 bytes)
static unsigned char header[54];

// Contadores de operaciones de lectura y escritura por imagen y totales
static long total_lecturas = 0, total_escrituras = 0;
static long all_lecturas = 0, all_escrituras = 0;

// Parámetros configurables: tamaño de kernel de blur y número de imágenes
static int KERNEL_SIZE = 55;
static int MAX_IMAGES = 100;

// Definición de un píxel RGB (3 bytes)
typedef struct { unsigned char b, g, r; } Pixel;

// ------------------------------------------------
// fread la cabecera BMP y extrae ancho y alto
// ------------------------------------------------
void leerHeader(FILE *in) {
    // Leer 54 bytes de cabecera
    if (fread(header, sizeof(header), 1, in) != 1) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    // Extraer ancho y alto de los bytes apropiados
    ancho = *(int *)&header[18];
    alto  = *(int *)&header[22];
}

// ------------------------------------------------
// Crear carpeta 'salidas' si no existe
// ------------------------------------------------
void crearCarpeta(const char *path) {
    struct stat st;
    // stat falla si no existe; entonces creamos carpeta
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) != 0) {
            perror("[ERROR] mkdir");
            exit(EXIT_FAILURE);
        }
    }
}

// ------------------------------------------------
// Escribe un BMP a disco con un sufijo en el nombre
// ------------------------------------------------
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
    size_t npix = (size_t)ancho * alto;
    fwrite(buf, sizeof(Pixel), npix, fout);
    fclose(fout);
    // Contar escrituras: 3 bytes por píxel
    total_escrituras += 3 * (long)npix;
}

// ================================================================
// Función principal: controla el flujo completo
// ================================================================
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
    leerHeader(tmpf);
    fclose(tmpf);
    printf("[LOG] Dimensiones detectadas: ancho=%d, alto=%d\n", ancho, alto);

    // Reservar buffers grandes contiguos para todos los efectos
    size_t npix = (size_t)ancho * alto;
    Pixel *buf_orig      = malloc(npix * sizeof(Pixel)); // Imagen original
    Pixel *buf_gris      = malloc(npix * sizeof(Pixel)); // Escala de grises
    Pixel *buf_espH      = malloc(npix * sizeof(Pixel)); // Espejo horizontal
    Pixel *buf_espV      = malloc(npix * sizeof(Pixel)); // Espejo vertical
    Pixel *buf_espH_gris = malloc(npix * sizeof(Pixel)); // Espejo gris horizontal
    Pixel *buf_espV_gris = malloc(npix * sizeof(Pixel)); // Espejo gris vertical
    Pixel *buf_tmp       = malloc(npix * sizeof(Pixel)); // Buffer intermedio blur
    Pixel *buf_blur      = malloc(npix * sizeof(Pixel)); // Resultado blur
    // Verificar malloc
    if (!buf_orig || !buf_gris || !buf_espH || !buf_espV ||
        !buf_espH_gris || !buf_espV_gris || !buf_tmp || !buf_blur) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        return EXIT_FAILURE;
    }

    // Asegurar carpeta de salida
    crearCarpeta("salidas");
    // Marca de tiempo de inicio global
    double t0 = omp_get_wtime();

    // ------------------------------------------------
    // Bucle principal: procesa cada imagen
    // ------------------------------------------------
    for (int img = 1; img <= MAX_IMAGES; img++) {
        total_lecturas = total_escrituras = 0;
        double img_start = omp_get_wtime();
        printf("[LOG] Procesando imagen %06d...\n", img);

        // Construir ruta y abrir BMP
        char iname[128];
        snprintf(iname, sizeof(iname), "imagenes_reto/imagenes_bmp_final/%06d.bmp", img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", iname); continue; }
        leerHeader(fin);

        // ------------------------------------------------
        // 1) Lectura de todos los píxeles (un fread de 3 bytes)
        // ------------------------------------------------
        for (size_t i = 0; i < npix; i++) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fin) != 3) {
                fprintf(stderr, "[ERROR] Fallo al leer píxel %zu de imagen %06d\n", i, img);
                break;
            }
            buf_orig[i].b = rgb[0];
            buf_orig[i].g = rgb[1];
            buf_orig[i].r = rgb[2];
            total_lecturas += 3;
        }
        fclose(fin);

        // ------------------------------------------------
        // 2) Escala de grises con paralelismo OpenMP
        // ------------------------------------------------
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r
                                              + 0.72f * buf_orig[i].g
                                              + 0.07f * buf_orig[i].b);
            buf_gris[i].r = buf_gris[i].g = buf_gris[i].b = lum;
        }

        // ------------------------------------------------
        // 3) Espejos horizontales y verticales (color)
        // ------------------------------------------------
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                size_t idx = (size_t)y * ancho + x;
                buf_espH[idx] = buf_orig[(size_t)y * ancho + (ancho - 1 - x)];
                buf_espV[idx] = buf_orig[(size_t)(alto - 1 - y) * ancho + x];
            }
        }

        // ------------------------------------------------
        // 4) Espejos horizontales y verticales (gris)
        // ------------------------------------------------
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < npix; i++) {
            int y = i / ancho, x = i % ancho;
            buf_espH_gris[i] = buf_gris[(size_t)y * ancho + (ancho - 1 - x)];
            buf_espV_gris[i] = buf_gris[(size_t)(alto - 1 - y) * ancho + x];
        }

        // ------------------------------------------------
        // 5) Desenfoque separable (horizontal + vertical)
        // ------------------------------------------------
        int k = KERNEL_SIZE / 2;
        // Pasa horizontal
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                int sr=0, sg=0, sb=0, cnt=0;
                for (int d = -k; d <= k; d++) {
                    int xx = x + d;
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
        // Pasa vertical
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < alto; y++) {
            for (int x = 0; x < ancho; x++) {
                int sr=0, sg=0, sb=0, cnt=0;
                for (int d = -k; d <= k; d++) {
                    int yy = y + d;
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

        // ------------------------------------------------
        // 6) Guardar resultados finales en archivos BMP
        // ------------------------------------------------
        writeBMP(img, "gris",      buf_gris);
        writeBMP(img, "esp_h",     buf_espH);
        writeBMP(img, "esp_v",     buf_espV);
        writeBMP(img, "esp_h_gris",buf_espH_gris);
        writeBMP(img, "esp_v_gris",buf_espV_gris);
        writeBMP(img, "blur",      buf_blur);

        // ------------------------------------------------
        // 7) Métricas por imagen: ancho de banda y tiempos
        // ------------------------------------------------
        double img_end = omp_get_wtime();
        double img_time = img_end - img_start;
        long bytes = (long)npix * sizeof(Pixel);  // bytes procesados útiles
        double bytes_per_sec = img_time > 0 ? (bytes / img_time) : 0;
        fprintf(log, "Imagen %06d: Lecturas=%ld, Escrituras=%ld, Bytes/s=%.2f\n",
                img, total_lecturas, total_escrituras, bytes_per_sec);

        all_lecturas   += total_lecturas;
        all_escrituras += total_escrituras;
        total_lecturas = total_escrituras = 0;
    }

    // ------------------------------------------------
    // Cálculo global de MIPS y cierre de archivos
    // ------------------------------------------------
    double t1 = omp_get_wtime();
    double tiempo = t1 - t0;
    long instr_mem = ancho * alto * 3 * 20;
    double mips    = instr_mem / tiempo / 1e6;
    printf("[LOG] Fin: Tiempo=%.2f s, MIPS=%.4f\n", tiempo, mips);
    fprintf(log, "Tiempo total: %.2f s, MIPS: %.4f\n", tiempo, mips);

    fclose(log);

    // Liberar memoria de buffers
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
