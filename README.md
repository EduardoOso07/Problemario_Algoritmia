# Experimentos de algoritmos: paralelismo, greedy, TSP y aproximación

Proyecto en Python que implementa los cinco ejercicios solicitados y guarda tablas CSV y gráficas PNG.

## Requisitos
- Python 3.10 o superior
- Dependencias: `pip install -r requirements.txt`

## Ejecución
Desde la carpeta raíz:
```bash
python src/main.py
```
En Windows, ejecutar desde terminal o PowerShell. El bloque `if __name__ == "__main__"` y `mp.freeze_support()` permiten usar multiprocessing correctamente.

## Estructura
```text
Proyecto_Algoritmos/
├── README.md
├── requirements.txt
├── src/
│   └── main.py
├── results/
│   ├── data/       # CSV generados
│   └── figures/    # gráficas PNG generadas
├── report/
│   └── reporte_final.docx
└── video/
    └── README.md
```

## Notas metodológicas
- Los experimentos fijan semillas para facilitar la reproducibilidad.
- El tiempo de paralelismo incluye la creación del pool; se usa el mejor de dos intentos para reducir ruido, por lo que los tiempos variarán según CPU y sistema operativo.
- En TSP, la calidad se mide por la longitud del ciclo; menor es mejor.
- Para 9 ciudades se calcula el óptimo exacto por fuerza bruta (8! permutaciones por instancia).
- El límite de razón MST <= 2 aplica al TSP métrico/Euclidiano y al recorrido de preorden con atajos, sujeto a la implementación estándar del algoritmo.
- El archivo CSV y las gráficas se regeneran al ejecutar el programa.
