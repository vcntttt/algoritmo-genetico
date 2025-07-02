import os
import json
import argparse
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# =========================
#        CONFIG
# =========================

CONFIG = {
    "default_json": "data.json",
    "default_votacion": 0,
    "default_quorum": 216,
    "default_size_poblacion": 38,
    "default_generaciones": 15000,
    "default_p_mutacion": 0.1700019,
    "default_p_seleccion": 0.141,
    "bench_file": "benchmark.csv",
    "output_dir": "cgm",
}

# =========================
#    CuPy or NumPy Setup
# =========================

def get_cp():
    try:
        import cupy as cp
        print(">> ¡Usando CuPy (GPU) para mejorar rendimiento!")
        return cp, True
    except ImportError:
        import numpy as cp
        print(">> CuPy no disponible. Usando NumPy como respaldo.")
        return cp, False

cp, USANDO_CUPY = get_cp()

# =========================
#      Utility Functions
# =========================

def now_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validar_parametros(args, coordenadas):
    if args.quorum > len(coordenadas):
        raise ValueError("El quorum requerido no puede ser mayor que la cantidad total de congresistas.")
    if args.quorum <= 0:
        raise ValueError("El quorum requerido debe ser positivo.")
    if args.size_poblacion > len(coordenadas):
        raise ValueError("El tamaño de la población no puede ser mayor que la cantidad de congresistas.")
    if args.size_poblacion <= 0:
        raise ValueError("El tamaño de población debe ser positiva.")
    if args.generaciones <= 0:
        raise ValueError("La cantidad de generaciones debe ser positiva.")
    if not (0 < args.p_mutacion < 1):
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1.")
    if not (0 < args.p_seleccion < 1):
        raise ValueError("La probabilidad de selección debe estar entre 0 y 1.")

# =========================
#        I/O
# =========================

def cargar_coordenadas_desde_json(ruta_json, indice_votacion=0):
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    votos = datos["rollcalls"][indice_votacion]["votes"]
    filas = [
        {
            "id": v["icpsr"],
            "x": v["x"],
            "y": v["y"],
            "partido": v.get("party", ""),
            "name": v["name"],
        }
        for v in votos
    ]
    return pd.DataFrame(filas)

def guardar_benchmark(resumen, bench_file):
    df_bench = pd.DataFrame([resumen])
    write_header = not os.path.isfile(bench_file)
    df_bench.to_csv(bench_file, mode="a", index=False, header=write_header)

def guardar_coalicion(df, indices_coalicion, ruta):
    coalicion_df = df.loc[indices_coalicion, ["name", "partido", "x", "y"]]
    coalicion_df.to_csv(ruta, index=False, encoding="utf-8")

def crear_directorios_resultados(output_dir, timestamp):
    dir_path = os.path.join(output_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# =========================
#    Genética y Cálculo
# =========================

def poligono_convexo(puntos):
    puntos_ordenados = sorted(puntos, key=lambda p: (p[0], p[1]))

    def producto_cruzado(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    parte_inferior, parte_superior = [], []
    for x, y, idx in puntos_ordenados:
        while len(parte_inferior) >= 2 and producto_cruzado(parte_inferior[-2], parte_inferior[-1], (x, y)) <= 0:
            parte_inferior.pop()
        parte_inferior.append((x, y, idx))
    for x, y, idx in reversed(puntos_ordenados):
        while len(parte_superior) >= 2 and producto_cruzado(parte_superior[-2], parte_superior[-1], (x, y)) <= 0:
            parte_superior.pop()
        parte_superior.append((x, y, idx))
    casco = parte_inferior[:-1] + parte_superior[:-1]
    return [idx for _, _, idx in casco]

def matriz_de_distancias(coordenadas):
    arr = cp.asarray(coordenadas)
    diff = arr[:, None, :] - arr[None, :, :]
    return cp.linalg.norm(diff, axis=2)

def poblacion_inicial(n_congresistas, quorum, size_pob):
    if quorum > n_congresistas:
        raise ValueError("El quorum no puede ser mayor al número total de congresistas.")
    return [random.sample(range(n_congresistas), quorum) for _ in range(size_pob)]

def torneo_de_seleccion(fitness, p):
    n = fitness.shape[0]
    orden = cp.argsort(fitness)
    probs = cp.asarray([p * (1 - p) ** i for i in range(n)])
    probs /= cp.sum(probs)
    dist_acum = cp.cumsum(probs)
    r = cp.random.rand()
    elegido = cp.searchsorted(dist_acum, r)
    return orden[elegido]

def cruce_genetico(padre, madre):
    largo = padre.shape[0]
    punto = cp.random.randint(1, largo)
    hijo1 = cp.concatenate([
        padre[:punto],
        cp.setdiff1d(madre, padre[:punto], assume_unique=True)
    ])[:largo]
    hijo2 = cp.concatenate([
        madre[:largo - punto],
        cp.setdiff1d(padre, madre[:largo - punto], assume_unique=True)
    ])[:largo]
    return hijo1, hijo2

def mutacion(individuo, n_genes):
    todos = cp.arange(n_genes)
    fuera = cp.setdiff1d(todos, individuo, assume_unique=True)
    if fuera.size == 0:
        return individuo
    i = cp.random.randint(0, individuo.shape[0])
    nuevo = fuera[cp.random.randint(0, fuera.size)]
    individuo[i] = nuevo
    return individuo

def evaluar_fitness(poblacion_gpu, matriz_distancias, usando_cupy):
    filas = poblacion_gpu[:, :, None]
    columnas = poblacion_gpu[:, None, :]
    submatriz = matriz_distancias[filas, columnas]
    distancias_gpu = cp.sum(cp.triu(submatriz, k=1), axis=(1,2))
    if usando_cupy:
        return distancias_gpu.get()
    else:
        return distancias_gpu

def algoritmo_genetico(
    coordenadas, quorum, size_poblacion, generaciones, p_mutacion, p_seleccion
):
    n_congresistas = coordenadas.shape[0]
    matriz_dist = matriz_de_distancias(coordenadas)
    poblacion_gpu = cp.asarray(
        [random.sample(range(n_congresistas), quorum) for _ in range(size_poblacion)],
        dtype=cp.int32
    )

    mejor_coalicion, mejor_fitness = None, float("inf")
    historial_fitness, tiempos_generacion = [], []
    tiempo_inicio = time.time()

    for gen in range(1, generaciones + 1):
        t0 = time.time()
        fitness = evaluar_fitness(poblacion_gpu, matriz_dist, USANDO_CUPY)
        t1 = time.time()
        tiempos_generacion.append(t1 - t0)

        idx_best = int(cp.argmin(cp.asarray(fitness)))
        best_fitness = float(fitness[idx_best])
        historial_fitness.append(best_fitness)
        print(f"Generacion {gen}/{generaciones}: tiempo {t1-t0:.6f}s, mejor fitness {best_fitness:.12f}")

        if best_fitness < mejor_fitness:
            mejor_fitness = best_fitness
            mejor_coalicion = poblacion_gpu[idx_best].get().tolist()

        nueva_poblacion = []
        for _ in range(size_poblacion // 2):
            idx_padre = torneo_de_seleccion(cp.asarray(fitness), p_seleccion)
            idx_madre = torneo_de_seleccion(cp.asarray(fitness), p_seleccion)
            padre = poblacion_gpu[idx_padre]
            madre = poblacion_gpu[idx_madre]
            hijo1, hijo2 = cruce_genetico(padre, madre)
            if cp.random.rand() < p_mutacion:
                hijo1 = mutacion(hijo1, n_congresistas)
            if cp.random.rand() < p_mutacion:
                hijo2 = mutacion(hijo2, n_congresistas)
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)

        poblacion_gpu = cp.stack(nueva_poblacion[:size_poblacion])

    total_time = time.time() - tiempo_inicio
    print(f">> Tiempo total algoritmo genético: {total_time:.2f}s")
    return mejor_coalicion, mejor_fitness, historial_fitness, tiempos_generacion, total_time

# =========================
#         Plots
# =========================

def graficar_resultados(coordenadas, partidos, coalicion, titulo=None, salida=None):
    plt.figure()
    partidos_np = np.array(partidos)
    dem = np.where(np.char.startswith(np.char.upper(partidos_np), "D"))[0]
    rep = np.where(np.char.startswith(np.char.upper(partidos_np), "R"))[0]
    otros = [i for i in range(len(partidos)) if i not in dem.tolist() + rep.tolist()]
    plt.scatter(coordenadas[dem, 0], coordenadas[dem, 1], c="blue", s=20, label="Dem")
    plt.scatter(coordenadas[rep, 0], coordenadas[rep, 1], c="red", s=20, label="Rep")
    if otros:
        plt.scatter(coordenadas[otros, 0], coordenadas[otros, 1], c="gray", s=20, label="Otro")
    casco = poligono_convexo([(coordenadas[i][0], coordenadas[i][1], i) for i in coalicion])
    poligono = coordenadas[casco]
    poligono = np.vstack([poligono, poligono[0]])
    plt.plot(poligono[:, 0], poligono[:, 1], "--", label="Pol-conv")
    if titulo:
        plt.title(titulo)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    if salida:
        plt.savefig(f"{salida}.png", dpi=300)
    plt.show()

def graficar_historial_costos(historial, salida="historial_costos.png"):
    plt.figure()
    plt.plot(range(1, len(historial) + 1), historial, linewidth=1)
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness")
    plt.title("Mejores fitness por generación")
    plt.grid(True)
    plt.savefig(salida, dpi=300)
    plt.show()

# =========================
#           MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Buscador de CGM mediante Algoritmo Genético (GPU).")
    grupo = parser.add_mutually_exclusive_group(required=False)
    grupo.add_argument("--json", default=CONFIG["default_json"], help="Archivo JSON con votaciones")
    parser.add_argument("--votacion", type=int, default=CONFIG["default_votacion"], help="Índice de la votación a usar")
    parser.add_argument("--quorum", type=int, default=CONFIG["default_quorum"], help="Tamaño requerido de la coalición (quorum requerido)")
    parser.add_argument("--size_poblacion", type=int, default=CONFIG["default_size_poblacion"], help="Tamaño de la población")
    parser.add_argument("--generaciones", type=int, default=CONFIG["default_generaciones"], help="Cantidad de generaciones")
    parser.add_argument("--p_mutacion", type=float, default=CONFIG["default_p_mutacion"], help="Probabilidad de mutación")
    parser.add_argument("--p_seleccion", type=float, default=CONFIG["default_p_seleccion"], help="Probabilidad de selección")
    parser.add_argument("--sin_graficos", action="store_true", help="Omitir la generación de gráficos")
    args = parser.parse_args()

    df = cargar_coordenadas_desde_json(args.json, args.votacion)
    coordenadas = df[["x", "y"]].values
    partidos = df["partido"].tolist()

    if len(coordenadas) == 0:
        raise ValueError("No hay congresistas disponibles en el archivo proporcionado.")

    validar_parametros(args, coordenadas)

    timestamp = now_timestamp()
    output_dir = crear_directorios_resultados(CONFIG["output_dir"], timestamp)
    coal_path = os.path.join(output_dir, "congresistas_coalicion.csv")
    poligono_path = os.path.join(output_dir, "poligono_convexo")
    fitness_path = os.path.join(output_dir, "mejores_fitness.png")

    cgm, fitness_cgm, historial, tiempos_gen, tiempo_total = algoritmo_genetico(
        coordenadas, args.quorum, args.size_poblacion, args.generaciones,
        args.p_mutacion, args.p_seleccion
    )

    print(f"CGM encontrada (quorum={args.quorum}) con fitness mínimo = {fitness_cgm:.4f}", flush=True)
    resumen = {
        "id": timestamp,
        "quorum_req": args.quorum,
        "size_poblacion": args.size_poblacion,
        "cant_generaciones": args.generaciones,
        "prob_mutacion": args.p_mutacion,
        "prob_seleccion": args.p_seleccion,
        "tiempo_total_ejec_algoritmo": f"{tiempo_total:.4f}",
        "mejor_fitness": f"{fitness_cgm:.8f}",
    }
    guardar_benchmark(resumen, CONFIG["bench_file"])
    print(f"Registro de benchmark guardado en {CONFIG['bench_file']}", flush=True)
    guardar_coalicion(df, cgm, coal_path)

    if not args.sin_graficos:
        graficar_resultados(
            coordenadas, partidos, cgm,
            titulo=f"CGM encontrada (quorum={args.quorum}, best_fitness={fitness_cgm:.4f})",
            salida=poligono_path,
        )
        graficar_historial_costos(historial, salida=fitness_path)


main()
