import os
import json
import argparse
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import cupy as cp

    print(">> ¡Usando CuPy (GPU) para mejorar rendimiento!", flush=True)
except ImportError:
    raise ImportError(">> CuPy no disponible. Comprese una GPU.")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def matriz_distancias_gpu(coords):
    """
    GPU: Convierte coords a CuPy y calcula la matriz n*n de distancias euclídeas.
    - coords: array (n*2) de coordenadas en CPU.
    Retorna: cupy.ndarray de forma (n, n) con distancias por pares.
    """
    coords_gpu = cp.asarray(coords)  # CPU → GPU

    # Broadcasting: genera un array (n, n, 2) con todas las diferencia entre pares de puntos: coords_gpu[i] - coords_gpu[j]
    diff = coords_gpu[:, None, :] - coords_gpu[None, :, :]

    # Norma euclídea a lo largo del eje de las componentes (Δx, Δy)
    return cp.linalg.norm(diff, axis=2)


def tournament_selection(poblacion, p_sel):
    """
    CPU: Torneo ponderado para escoger dos padres de la población, con mayor probabilidad de elegir a los individuos de mejor ranking (menor fitness).
    - poblacion: lista de individuos (ya ordenada de mejor a peor fitness).
    - p_sel: parámetro de presión de selección (float entre 0 y 1).
    Retorna dos individuos (padre, madre).
    """
    n = len(poblacion)
    pesos = [p_sel * ((1 - p_sel) ** i) for i in range(n)]

    # normalizamos
    total = sum(pesos)
    probabilidades = [p / total for p in pesos]  # [0.2, 0.3, 0.4, etc.]

    idx_padre = random.choices(range(n), weights=probabilidades, k=1)[0]
    idx_madre = random.choices(range(n), weights=probabilidades, k=1)[0]

    return poblacion[idx_padre], poblacion[idx_madre]


def crossover(padre, madre, q):
    """
    CPU: Cruce genético de un solo punto sin duplicaciones.
    - padre, madre: listas de genes (enteros).
    - q: longitud del cromosoma.
    Retorna dos hijos (hijo1, hijo2) sin duplicar genes.
    """
    corte = random.randint(1, q - 1)  # evitando primero y ultimo
    prefijo_padre = padre[:corte]
    prefijo_madre = madre[: q - corte]

    # Construir la “cola” (genes que faltan) filtrando duplicados
    #    Primero listamos todos los genes de la madre que no estén
    #    en el prefijo del padre, en el orden en que aparecen.
    complemento1 = [g for g in madre if g not in prefijo_padre]
    #    Y recortamos justo los q-corte que necesitamos.
    cola1 = complemento1[: q - corte]

    # lo mismo para el otro hijo pero al reves
    complemento2 = [g for g in padre if g not in prefijo_madre]
    cola2 = complemento2[:corte]

    # Concatenar prefijo + cola
    hijo1 = prefijo_padre + cola1
    hijo2 = prefijo_madre + cola2
    return hijo1, hijo2


def mutate(individuo, n, p_mut):
    """
    CPU: Mutación de un gen.
    - individuo: lista de genes actuales (longitud q).
    - n: tamaño total del pool de genes válidos (0..n-1).
    - p_mut: probabilidad de que ocurra la mutación (0 < p_mut < 1).
    Retorna el individuo, posiblemente modificado.
    """
    if random.random() < p_mut:
        pos = random.randrange(len(individuo))  # gen a mutar aleatorio
        disponibles = list(set(range(n)) - set(individuo)) # evitar duplicados
        if disponibles:
            individuo[pos] = random.choice(disponibles)

    return individuo


def algoritmo_genetico(coords, q, size_poblacion, max_generaciones, p_mut, p_sel):
    """
    coords            : np.ndarray (n*2)
    q                 : tamaño del cromosoma/quorum
    size_poblacion    : tamaño de la población
    max_generaciones  : número máximo de generaciones
    p_mut             : probabilidad de mutación
    p_sel             : controla la probabilidad de selección en el torneo
    """
    n = coords.shape[0]

    # GPU: matriz de distancias
    D_gpu = matriz_distancias_gpu(coords)

    # CPU: población inicial aleatoria
    poblacion = [random.sample(range(n), q) for _ in range(size_poblacion)]

    best_global, best_fit = None, float("inf")
    historial = []
    t_start = time.time()

    for generacion in range(1, max_generaciones + 1):
        # GPU: calcular fitness de toda la población
        poblacion_gpu = cp.asarray(poblacion, dtype=cp.int32)  # CPU → GPU
        sub = D_gpu[poblacion_gpu[:, :, None], poblacion_gpu[:, None, :]]
        fit_gpu = cp.sum(cp.triu(sub, 1), axis=(1, 2))
        idx_sorted = cp.argsort(fit_gpu).get()
        fit_arr = fit_gpu.get()[idx_sorted]  # GPU → CPU y lo ordena altiro

        # CPU: reordenar población según fitness
        poblacion = [poblacion[i] for i in idx_sorted]
        fit_arr = fit_arr

        # Extraer mejor de esta generación
        current_best = poblacion[0].copy()  # cromosoma
        current_fit = float(fit_arr[0])

        # Actualizar global si hay mejora
        if current_fit < best_fit:
            best_fit = current_fit
            best_global = current_best.copy()
            print(f"> Gen {generacion} → Nuevo best global: {best_fit:.6f}", flush=True)
            historial.append((generacion, best_fit))

        # por si nos pillamos al optimo antes del limite de iteraciones
        if abs(best_fit - 9686.93831) < 0.01:
            print(f">> Óptimo encontrado en {generacion} generaciones.", flush=True)
            break

        # conservar el mejor actual
        nueva_poblacion = [current_best]

        # llenar poblacion
        while len(nueva_poblacion) < size_poblacion:
            padre, madre = tournament_selection(poblacion, p_sel)
            hijo1, hijo2 = crossover(padre, madre, q)
            hijo1 = mutate(hijo1, n, p_mut)
            hijo2 = mutate(hijo2, n, p_mut)
            nueva_poblacion.extend([hijo1, hijo2])

        # Recortar exceso y asignar para la próxima generación
        poblacion = nueva_poblacion[:size_poblacion]

    total_time = time.time() - t_start
    print(
        f">> Total GA: {total_time:.2f}s — Mejor fitness: {best_fit:.6f} — Generaciones: {generacion}",
        flush=True,
    )
    return best_global, best_fit, historial, total_time, generacion


def json_a_posiciones(ruta_json, indice_votacion=0):
    """
    Recibe la ruta de un archivo JSON (string) y un índice de votación (int, opcional);
    abre el archivo, extrae la información de los congresistas para la votación indicada y construye un DataFrame de pandas
    con las columnas: id, x, y, partido y nombre. Retorna ese DataFrame, listo para ser usado como entrada en el resto del algoritmo.
    """
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


def poligono_convexo(puntos):
    puntos_ordenados = sorted(puntos, key=lambda p: (p[0], p[1]))

    def producto_cruzado(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    parte_inferior, parte_superior = [], []
    for x, y, idx in puntos_ordenados:
        while (
            len(parte_inferior) >= 2
            and producto_cruzado(parte_inferior[-2], parte_inferior[-1], (x, y)) <= 0
        ):
            parte_inferior.pop()
        parte_inferior.append((x, y, idx))
    for x, y, idx in reversed(puntos_ordenados):
        while (
            len(parte_superior) >= 2
            and producto_cruzado(parte_superior[-2], parte_superior[-1], (x, y)) <= 0
        ):
            parte_superior.pop()
        parte_superior.append((x, y, idx))
    casco = parte_inferior[:-1] + parte_superior[:-1]
    return [i for _, _, i in casco]


def graficar_resultados(coordenadas, partidos, coalicion, titulo=None, salida=None):
    plt.figure(figsize=(10, 6))
    indices = set(range(len(coordenadas)))
    coalicion_set = set(coalicion)
    fuera_coalicion = list(indices - coalicion_set)
    dentro_coalicion = list(coalicion_set)

    dem = [i for i, p in enumerate(partidos) if p.upper().startswith("D")]
    rep = [i for i, p in enumerate(partidos) if p.upper().startswith("R")]

    dem_fuera = [i for i in dem if i in fuera_coalicion]
    rep_fuera = [i for i in rep if i in fuera_coalicion]

    dem_dentro = [i for i in dem if i in dentro_coalicion]
    rep_dentro = [i for i in rep if i in dentro_coalicion]

    if dem_fuera:
        plt.scatter(
            coordenadas[dem_fuera, 0],
            coordenadas[dem_fuera, 1],
            c="blue",
            marker="x",
            s=12,
            label="Demócrata fuera",
        )
    if rep_fuera:
        plt.scatter(
            coordenadas[rep_fuera, 0],
            coordenadas[rep_fuera, 1],
            c="red",
            marker="x",
            s=12,
            label="Republicano fuera",
        )

    if dem_dentro:
        plt.scatter(
            coordenadas[dem_dentro, 0],
            coordenadas[dem_dentro, 1],
            c="blue",
            marker="o",
            s=8,
            label="Demócrata coalición",
        )
    if rep_dentro:
        plt.scatter(
            coordenadas[rep_dentro, 0],
            coordenadas[rep_dentro, 1],
            c="red",
            marker="o",
            s=8,
            label="Republicano coalición",
        )

    casco = poligono_convexo(
        [(coordenadas[i][0], coordenadas[i][1], i) for i in coalicion]
    )
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
        plt.savefig(salida, dpi=300)

    plt.show()


def graficar_historial_costos(historial_costos, salida="historial_costos.png"):
    plt.figure()
    if len(historial_costos) > 0 and isinstance(historial_costos[0], tuple):
        if len(historial_costos[0]) == 3:
            generaciones, fitness, tiempos = zip(*historial_costos)
        else:
            generaciones, fitness = zip(*historial_costos)
        plt.plot(generaciones, fitness)
    else:
        plt.plot(range(1, len(historial_costos) + 1), historial_costos, linewidth=1)
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness")
    plt.title("Mejores fitness por mejora")
    plt.grid(True)
    plt.savefig(salida, dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description="Buscador de CGM mediante Algoritmo Genético (GPU)."
    )
    grupo = parser.add_mutually_exclusive_group(required=False)
    grupo.add_argument(
        "--json", default="data.json", help="Archivo JSON con votaciones"
    )
    parser.add_argument(
        "--votacion", type=int, default=0, help="Índice de la votación a usar"
    )
    parser.add_argument(
        "--quorum",
        type=int,
        default=216,
        help="Tamaño requerido de la coalición (quorum requerido)",
    )
    parser.add_argument(
        "--size_poblacion", type=int, default=38, help="Tamaño de la población"
    )
    parser.add_argument(
        "--generaciones", type=int, default=20000, help="Cantidad de generaciones"
    )
    parser.add_argument(
        "--p_mutacion", type=float, default=0.1700019, help="Probabilidad de mutación"
    )
    parser.add_argument(
        "--p_seleccion", type=float, default=0.141, help="Probabilidad de selección"
    )
    parser.add_argument(
        "--sin_graficos", action="store_true", help="Omitir la generación de gráficos"
    )
    args = parser.parse_args()

    if args.json:
        df = json_a_posiciones(args.json, args.votacion)
    else:
        parser.error("No se encontró el archivo json.")

    coordenadas = df[["x", "y"]].values
    partidos = df["partido"].tolist()

    if len(coordenadas) == 0:
        parser.error("No hay congresistas disponibles en el archivo proporcionado.")

    if args.quorum > len(coordenadas):
        parser.error(
            "El quorum requerido no puede ser mayor que la cantidad total de congresistas."
        )

    if args.quorum <= 0:
        parser.error("El quorum requerido debe ser positivo.")

    if args.size_poblacion > len(coordenadas):
        parser.error(
            "El tamaño de la población no puede ser mayor que la cantidad de congresistas."
        )

    if args.size_poblacion <= 0:
        parser.error("El tamaño de población debe ser positiva.")

    if args.generaciones <= 0:
        parser.error("La cantidad de generaciones debe ser positiva.")

    if not (0 < args.p_mutacion < 1):
        parser.error("La probabilidad de mutación debe estar entre 0 y 1.")

    if not (0 < args.p_seleccion < 1):
        parser.error("La probabilidad de selección debe estar entre 0 y 1.")

    (
        cgm,
        fitness_cgm,
        historial_distancias,
        tiempo_total,
        gens,
    ) = algoritmo_genetico(
        coordenadas,
        args.quorum,  # q
        args.size_poblacion,  # pop_size
        args.generaciones,  # gens
        args.p_mutacion,  # p_mut
        args.p_seleccion,  # p_sel
    )

    print(
        f"CGM encontrada (quorum={args.quorum}) con fitness mínimo = {fitness_cgm:.6f}",
        flush=True,
    )

    # agregar registro a historial general
    resumen = {
        "id": timestamp,
        "quorum_req": args.quorum,
        "size_poblacion": args.size_poblacion,
        "cant_generaciones": gens,
        "prob_mutacion": args.p_mutacion,
        "prob_seleccion": args.p_seleccion,
        "tiempo_total_ejec_algoritmo": f"{tiempo_total:.4f}",
        "mejor_fitness": f"{fitness_cgm:.6f}",
    }
    bench_file = "benchmark.csv"

    df_bench = pd.DataFrame([resumen])
    write_header = not os.path.isfile(bench_file)
    df_bench.to_csv(bench_file, mode="a", index=False, header=write_header)
    print(f"Registro de benchmark guardado en {bench_file}", flush=True)

    # guardar resultados específicos
    os.makedirs("cgm", exist_ok=True)
    os.makedirs(f"cgm/{timestamp}", exist_ok=True)

    coalicion_df = df.loc[cgm, ["name", "partido", "x", "y"]]
    coalicion_df.to_csv(
        f"cgm/{timestamp}/congresistas_coalicion.csv",
        index=False,
        encoding="utf-8",
    )

    graficar_resultados(
        coordenadas,
        partidos,
        cgm,
        titulo=f"CGM encontrada (quorum={args.quorum}, best_fitness={fitness_cgm:.6f})",
        salida=f"cgm/{timestamp}/poligono_convexo",
    )

    graficar_historial_costos(
        historial_distancias, salida=f"cgm/{timestamp}/mejores_fitness"
    )


main()
