# Instituto Tecnológico y de Estudios Superiores de Monterrey  
## Campus Puebla
# Avances del Reto

**Integrantes**  
- Hugo Muñoz Rodríguez - A01736149  
- Rogelio Hernández Cortés - A01735819  
- Hedguhar Domínguez González - A01730640  

---

## Descripción

Este programa fue desarrollado en lenguaje C con la finalidad de procesar imágenes BMP aplicando distintos efectos visuales como escala de grises, reflejos (espejos) tanto vertical como horizontalmente, y desenfoque. Se usa paralelismo con OpenMP para acelerar algunas operaciones que se pueden realizar de forma simultánea.

---

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

---

## Tecnologías Implementadas

- **Lenguaje C**: para manipulación directa de archivos e imágenes en formato BMP.  
- **OpenMP**: permite paralelizar ciertas operaciones (como los filtros) para que se procesen más rápido usando múltiples núcleos de la CPU.  
- **BMP**: es un formato de imagen no comprimido, ideal para este tipo de manipulaciones por su estructura simple.

---


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

---

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

---

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

---

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

---

## Conclusión

El desarrollo de este programa permitió explorar y aplicar conceptos avanzados en procesamiento de imágenes, programación en C y paralelización con OpenMP. A través de la implementación de filtros como escala de grises, espejos y desenfoque, se logró transformar imágenes de manera eficiente. El uso de OpenMP resultó fundamental para optimizar el rendimiento, especialmente en equipos con múltiples núcleos.

Los resultados obtenidos demuestran que la implementación fue exitosa, procesando grandes volúmenes de imágenes con tiempos razonables. La comparación entre diferentes equipos evidenció la influencia del hardware en el rendimiento, destacando el beneficio del paralelismo. Además, el análisis del costo de operación mostró la viabilidad de este tipo de procesamiento tanto en equipos locales como en servicios en la nube.

En resumen, este reto permitió consolidar habilidades técnicas y analíticas, así como fomentar buenas prácticas de programación estructurada y eficiente.
