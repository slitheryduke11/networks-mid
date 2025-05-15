# Instituto Tecnológico y de Estudios Superiores de Monterrey  
## Campus Puebla
# Avances del Reto

**Integrantes**  
- Hugo Muñoz Rodríguez - A01736149  
- Rogelio Hernández Cortés - A01735819  
- Hedguhar Domínguez González - A01730640  

## Apendice

**Link de repositorio**

https://github.com/slitheryduke11/networks-mid.git

**Código**

```c
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
    if (fread(header, sizeof(header), 1, in) != 1) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    width = *(int *)&header[18];
    height  = *(int *)&header[22];
}

// Crear carpeta 'salidas' si no existe
void createFolder(const char *path) {
    struct stat st;
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
    snprintf(oname, sizeof(oname), "salidas/%06d_%s_%d.bmp", img, suffix, KERNEL_SIZE);
    FILE *fout = fopen(oname, "wb");
    if (!fout) {
        fprintf(stderr, "[ERROR] No se puede crear '%s'\n", oname);
        return;
    }
    fwrite(header, sizeof(header), 1, fout);
    size_t npix = (size_t)width * height;
    fwrite(buf, sizeof(Pixel), npix, fout);
    fclose(fout);
    total_writes += 3 * (long)npix;
}

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

    char tmp_name[128];
    snprintf(tmp_name, sizeof(tmp_name), "imagenes_reto/imagenes_bmp_final/000001.bmp");
    FILE *tmpf = fopen(tmp_name, "rb");
    if (!tmpf) { perror("[ERROR] Abrir primera BMP"); return EXIT_FAILURE; }
    readHeader(tmpf);
    fclose(tmpf);
    printf("[LOG] Dimensiones detectadas: width=%d, height=%d\n", width, height);

    size_t npix = (size_t)width * height;
    Pixel *buf_orig = malloc(npix * sizeof(Pixel));
    Pixel *buf_gray = malloc(npix * sizeof(Pixel));
    Pixel *buf_hmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_vmirror = malloc(npix * sizeof(Pixel));
    Pixel *buf_hgray = malloc(npix * sizeof(Pixel));
    Pixel *buf_vgray = malloc(npix * sizeof(Pixel));
    Pixel *buf_tmp = malloc(npix * sizeof(Pixel));
    Pixel *buf_blur = malloc(npix * sizeof(Pixel));
    if (!buf_orig || !buf_gray || !buf_hmirror || !buf_vmirror ||
        !buf_hgray || !buf_vgray || !buf_tmp || !buf_blur) {
        fprintf(stderr, "[ERROR] malloc falló\n");
        return EXIT_FAILURE;
    }

    createFolder("salidas");
    double t0_global = omp_get_wtime();
    double t_total_read = 0.0, t_total_gray = 0.0, t_total_mirror = 0.0;
    double t_total_blur = 0.0, t_total_write = 0.0;

    for (int img = 1; img <= MAX_IMAGES; img++) {
        total_reads = total_writes = 0;
        double img_start = omp_get_wtime();
        printf("[LOG] Procesando imagen %06d...\n", img);

        char iname[128];
        snprintf(iname, sizeof(iname), "imagenes_reto/imagenes_bmp_final/%06d.bmp", img);
        FILE *fin = fopen(iname, "rb");
        if (!fin) { fprintf(stderr, "[ERROR] No se puede abrir %s\n", iname); continue; }
        readHeader(fin);
        double t0 = omp_get_wtime();
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
        t1 = omp_get_wtime();
        double t_blur = t1 - t0;
        t_total_blur += t_blur;

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

        double img_end = omp_get_wtime();
        double img_time = img_end - img_start;
        long bytes = (long)npix * sizeof(Pixel);
        double bytes_per_sec = img_time > 0 ? (bytes / img_time) : 0;
        double mbytes_per_sec = bytes_per_sec/1000000;
        fprintf(log, "Img %06d: read=%.4f s,transform gray:%.4fs,transform mirror:%.4fs,transform blur:%.4fs, write=%.4f s, total=%.4f s, mBytes/s=%.2f\n",
                img, t_read, t_gray, t_mirror, t_blur, t_write, img_time, mbytes_per_sec);
        all_reads += total_reads;
        all_writes += total_writes;
    }

    double t1_global = omp_get_wtime();
    double tiempo_total = t1_global - t0_global;
    long instr_mem = width * height * 3 * 20;
    double mips = instr_mem / tiempo_total / 1e6;
    size_t total_bytes = (size_t)width * height * sizeof(Pixel) * MAX_IMAGES;
    double avg_bps = tiempo_total > 0 ? ((double)total_bytes / tiempo_total) : 0;
    double avg_mbps = avg_bps / 1000000;

    printf("[LOG] Fin: Tiempo=%.2f s, MIPS=%.4f\n", tiempo_total, mips);
    printf("Promedios (s): read=%.4f, gray=%.4f, mirror=%.4f, blur=%.4f, write=%.4f\n",
           t_total_read/MAX_IMAGES, t_total_gray/MAX_IMAGES,
           t_total_mirror/MAX_IMAGES, t_total_blur/MAX_IMAGES, t_total_write/MAX_IMAGES);
    fprintf(log, "Tiempo total: %.2f s, MIPS: %.4f, Promedio MegaBytes/s: %.2f\n",tiempo_total, mips, avg_mbps);

    fclose(log);
    free(buf_orig); free(buf_gray); free(buf_hmirror); free(buf_vmirror);
    free(buf_hgray); free(buf_vgray); free(buf_tmp); free(buf_blur);
    return EXIT_SUCCESS;
}
```


## Descripción

Este programa fue desarrollado en lenguaje C con la finalidad de procesar imágenes BMP aplicando distintos efectos visuales como escala de grises, reflejos (espejos) tanto vertical como horizontalmente, y desenfoque. Se usa paralelismo con OpenMP para acelerar algunas operaciones que se pueden realizar de forma simultánea.

## Procedimiento

### Lectura de Parámetros de Entrada
Se reciben dos argumentos desde la terminal:
- **KERNEL_SIZE**: tamaño del filtro de desenfoque (por ejemplo, 5x5).
- **MAX_IMAGES**: cantidad total de imágenes a procesar (por ejemplo, 100).

### Inicialización
- Se crea una carpeta llamada `salidas` si no existe, para guardar los resultados.
```c
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
```
- Se lee la cabecera de la imagen BMP usando:

```c
void readHeader(FILE *in) {
    if (fread(header, sizeof(header), 1, in) != 1) {
        fprintf(stderr, "[ERROR] Lectura de cabecera fallida\n");
        exit(EXIT_FAILURE);
    }
    width = *(int *)&header[18];
    height = *(int *)&header[22];
}
```

Esta función usa `fread` para leer los primeros 54 bytes del archivo BMP y extrae de forma manual los valores de ancho (`width`) y alto (`height`) desde los bytes 18 y 22 respectivamente del encabezado BMP.

- Se reservan buffers para cada tipo de transformación:

```c
typedef struct { unsigned char b, g, r; } Pixel;

Pixel *buf_orig, *buf_gray, *buf_hmirror, *buf_vmirror, *buf_hgray, *buf_vgray, *buf_tmp, *buf_blur;
```

Cada uno de estos apuntadores almacenará temporalmente los píxeles modificados para cada tipo de efecto aplicado.

### Proceso por Imagen (en bucle)
Para cada imagen, el programa realiza lo siguiente:

1. **Lectura de la imagen original**:
   - Se construye la ruta a la imagen BMP y se abre con `fopen`.
   - Se lee píxel por píxel (3 bytes por píxel: B, G, R) y se almacena en `buf_orig`.

```c
for (size_t i = 0; i < npix; i++) {
    fread(rgb, 1, 3, fin);
    buf_orig[i].b = rgb[0];
    buf_orig[i].g = rgb[1];
    buf_orig[i].r = rgb[2];
}
```

2. **Conversión a escala de grises usando OpenMP:**
   - Se usa una fórmula de luminancia ponderada y se copia ese valor a los tres canales RGB.
   - Se paraleliza usando `#pragma omp parallel for`.

```c
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < npix; i++) {
    unsigned char lum = (unsigned char)(0.21f * buf_orig[i].r + 0.72f * buf_orig[i].g + 0.07f * buf_orig[i].b);
    buf_gray[i].r = buf_gray[i].g = buf_gray[i].b = lum;
}
```

3. **Aplicación de efecto espejo (color):**
   - Se invierten coordenadas horizontal y verticalmente usando acceso por índice en la matriz lineal.
   - `collapse(2)` permite paralelizar dos bucles anidados.

```c
#pragma omp parallel for collapse(2) schedule(static)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        size_t idx = y * width + x;
        buf_hmirror[idx] = buf_orig[y * width + (width - 1 - x)];
        buf_vmirror[idx] = buf_orig[(height - 1 - y) * width + x];
    }
}
```

4. **Espejo en escala de grises:**
   - Se aplica el mismo principio pero sobre la imagen ya convertida a gris.

```c
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < npix; i++) {
    int y = i / width, x = i % width;
    buf_hgray[i] = buf_gray[y * width + (width - 1 - x)];
    buf_vgray[i] = buf_gray[(height - 1 - y) * width + x];
}
```

5. **Aplicación de desenfoque separable (blur):**
   - Se usa un enfoque separable: primero horizontal, luego vertical.
   - Se recorre una ventana de tamaño `KERNEL_SIZE` centrada en el píxel.

**Paso Horizontal:**
```c
#pragma omp parallel for collapse(2) schedule(static)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        int sr=0, sg=0, sb=0, cnt=0;
        for (int d = -k; d <= k; d++) {
            int xx = x + d;
            if (xx >= 0 && xx < width) {
                Pixel *p = &buf_orig[y * width + xx];
                sr += p->r; sg += p->g; sb += p->b; cnt++;
            }
        }
        buf_tmp[y * width + x].r = sr / cnt;
        buf_tmp[y * width + x].g = sg / cnt;
        buf_tmp[y * width + x].b = sb / cnt;
    }
}
```

**Paso Vertical:**
```c
#pragma omp parallel for collapse(2) schedule(static)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        int sr=0, sg=0, sb=0, cnt=0;
        for (int d = -k; d <= k; d++) {
            int yy = y + d;
            if (yy >= 0 && yy < height) {
                Pixel *p = &buf_tmp[yy * width + x];
                sr += p->r; sg += p->g; sb += p->b; cnt++;
            }
        }
        buf_blur[y * width + x].r = sr / cnt;
        buf_blur[y * width + x].g = sg / cnt;
        buf_blur[y * width + x].b = sb / cnt;
    }
}
```

6. **Escritura de imágenes procesadas:**
   - Se guarda cada versión con un sufijo que representa el efecto aplicado.
```c
writeBMP(img, "gris", buf_gray);
writeBMP(img, "esp_h", buf_hmirror);
writeBMP(img, "esp_v", buf_vmirror);
writeBMP(img, "esp_h_gris", buf_hgray);
writeBMP(img, "esp_v_gris", buf_vgray);
writeBMP(img, "blur", buf_blur);
```

7. **Cálculo de estadísticas:**
   - Se mide el tiempo por imagen y se calculan promedios y rendimiento en MB/s y MIPS.
   - Se registran en el archivo `estadisticas.txt`.


## Tecnologías Implementadas

- **Lenguaje C**: para manipulación directa de archivos e imágenes en formato BMP.  
- **OpenMP**: permite paralelizar ciertas operaciones (como los filtros) para que se procesen más rápido usando múltiples núcleos de la CPU.  
- **BMP**: es un formato de imagen no comprimido, ideal para este tipo de manipulaciones por su estructura simple.

## Resultados Obtenidos

### Laptop 1
**Características:**
- macOS  
- Chip M4 Apple  
- GPU de 8 núcleos  
- CPU de 10 núcleos  
- 16 GB RAM  
- Batería de 53.8 Wh  

**Resultados:**
- Se procesaron 100 imágenes.
- Se generaron 600 imágenes (6 efectos por imagen).
- Tiempo total: ~71.02 segundos.
- Rendimiento: **5.43 MIPS**.

### Laptop 2
**Características:**
- Ubuntu 24.04  
- Intel Core i7  
- CPU de 8 núcleos  
- 8 GB RAM  
- Batería de 70.9 Wh  

**Resultados:**
- Se procesaron 100 imágenes.
- Se generaron 600 imágenes.
- Tiempo total: ~112.97 segundos.
- Rendimiento: **4.78 MIPS**.

### Laptop 3
**Características:**
- Windows 11  
- AMD Ryzen 7 5000 series  
- CPU de 8 núcleos  
- 16 GB RAM  
- Batería de 13.3 Wh  

**Resultados:**
- Se procesaron 100 imágenes.
- Se generaron 600 imágenes.
- Tiempo total: ~545.78 segundos.
- Rendimiento: **0.9894 MIPS**.

## Costo de Operación

**Simulación con instancia AWS con mejor desempeño**  
- AWS EC2 Linux 8vCPUs 16GiB  
- $243.09 USD / mes → **$2917.08 USD / año**

**Consumo eléctrico anual (1.60 MXN/kWh):**

| Laptop | Batería (W) | Tiempo (h) | Tarifa | Costo Anual (MXN) |
|--------|-------------|------------|--------|--------------------|
| 1      | 53.8        | 2080       | 1.5    | $167.85            |
| 2      | 70.9        | 2080       | 1.5    | $221.21            |
| 3      | 13.3        | 2080       | 1.5    | $41.50             |

---

## Análisis de Resultados

Los resultados obtenidos a partir de las tres laptops utilizadas permiten realizar un análisis comparativo desde diferentes dimensiones:

1. **Rendimiento (MIPS)**: La Laptop 1 con chip Apple M4 fue la más rápida con un rendimiento de 5.43 MIPS, seguida por la Laptop 2 con 4.78 MIPS y finalmente la Laptop 3 con 0.9894 MIPS. Esto demuestra que arquitecturas más modernas y eficientes como ARM (M4) ofrecen ventajas claras frente a arquitecturas tradicionales como x86.

2. **Tiempo de procesamiento total**: Mientras la Laptop 1 procesó las 100 imágenes en aproximadamente 71 segundos, la Laptop 3 tardó más de 9 minutos. Esto refuerza la necesidad de contar con hardware potente o estrategias de paralelización cuando se trabaja con grandes volúmenes de datos.

3. **Consumo energético**: Aunque la Laptop 3 fue la más lenta, también fue la más eficiente en consumo energético (41.50 MXN anuales simulados). En cambio, la Laptop 2, aunque con mejor rendimiento, implicó mayor costo anual (221.21 MXN). Esto pone en perspectiva el balance entre rendimiento y eficiencia energética según el uso esperado.

4. **Comparación con AWS**: Simular el procesamiento en una instancia EC2 resultó considerablemente más costoso (casi 3000 USD al año). Esto sugiere que, para tareas de alto volumen pero no constantes, puede ser más rentable utilizar infraestructura local optimizada.

5. **Eficiencia de paralelización**: El uso de OpenMP permitió aprovechar múltiples núcleos, reduciendo el tiempo de ejecución considerablemente en comparación con una ejecución secuencial. Este tipo de paralelización resulta indispensable en procesamiento de imágenes a escala.


## Conclusión

El desarrollo de este programa permitió explorar y aplicar conceptos avanzados en procesamiento de imágenes, programación en C y paralelización con OpenMP. A través de la implementación de filtros como escala de grises, espejos y desenfoque, se logró transformar imágenes de manera eficiente. El uso de OpenMP resultó fundamental para optimizar el rendimiento, especialmente en equipos con múltiples núcleos.

Los resultados obtenidos demuestran que la implementación fue exitosa, procesando grandes volúmenes de imágenes con tiempos razonables. La comparación entre diferentes equipos evidenció la influencia del hardware en el rendimiento, destacando el beneficio del paralelismo. Además, el análisis del costo de operación mostró la viabilidad de este tipo de procesamiento tanto en equipos locales como en servicios en la nube.

En resumen, este reto permitió consolidar habilidades técnicas y analíticas, así como fomentar buenas prácticas de programación estructurada y eficiente.
