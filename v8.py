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

    usando_cupy = True
    print(">> ¡Usando CuPy (GPU) para mejorar rendimiento!", flush=True)
except ImportError:
    import numpy as cp

    usando_cupy = False
    print(">> CuPy no disponible. Usando NumPy como respaldo.", flush=True)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


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


def matriz_de_distancias(coords):
    arr = cp.asarray(coords)
    diff = arr[:, None, :] - arr[None, :, :]
    return cp.linalg.norm(diff, axis=2)


def poblacion_inicial(n, q, size):
    if q > n:
        raise ValueError("Quorum mayor que congresistas")
    return [random.sample(range(n), q) for _ in range(size)]


def torneo_de_seleccion(pop, fitness, p):
    order = sorted(range(len(pop)), key=lambda i: fitness[i])
    probs = [p * (1 - p) ** i for i in range(len(pop))]
    total = sum(probs)
    probs = [x / total for x in probs]
    choice = random.choices(order, weights=probs, k=1)[0]
    return pop[choice].copy()


def cruce_genetico(p1, p2, q):
    cut = random.randint(1, q - 1)
    c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]][: q - cut]
    c2 = p2[: q - cut] + [g for g in p1 if g not in p2[: q - cut]][:cut]
    return c1, c2


def mutacion(ind, n):
    choices = list(set(range(n)) - set(ind))
    if choices:
        i = random.randrange(len(ind))
        ind[i] = random.choice(choices)
    return ind


def algoritmo_genetico(coords, q, pop_size, gens, p_mut, p_sel):
    n = coords.shape[0]
    D = matriz_de_distancias(coords)
    pop = poblacion_inicial(n, q, pop_size)
    pop_gpu = cp.asarray(pop, dtype=cp.int32)

    best_global, best_fit = None, float("inf")
    history, times = [], []
    limit = 0
    start_all = time.time()

    for gen in range(1, gens + 1):
        t0 = time.time()
        sub = D[pop_gpu[:, :, None], pop_gpu[:, None, :]]
        fit_gpu = cp.sum(cp.triu(sub, 1), axis=(1, 2))
        fit_arr = fit_gpu.get() if usando_cupy else fit_gpu

        order_idx = np.argsort(fit_arr)
        pop = [pop[i] for i in order_idx]
        fit_arr = fit_arr[order_idx]

        current_best, current_fit = pop[0].copy(), float(fit_arr[0])
        history.append(current_fit)

        # si hay mejora global
        if current_fit < best_fit:
            best_fit = current_fit
            best_global = current_best.copy()
            limit = 0
            print(
                f"Gen {gen} -> Nuevo best global: {best_fit:.2f} | t_iter: {time.time() - t0:.4f}s",
                flush=True,
            )
        else:
            limit += 1

        # # si no hay mejora en cierto tiempo, detenemos (minimo llega a 10000)
        # if limit >= 2500 and gens > 10000:
        #     print(f"No hubo mejora en {limit} gen. Deteniendo early.", flush=True)
        #     break

        # elitismo y siguiente gen
        new_pop = [current_best]
        fit_list = fit_arr.tolist()
        while len(new_pop) < pop_size:
            p1 = torneo_de_seleccion(pop, fit_list, p_sel)
            p2 = torneo_de_seleccion(pop, fit_list, p_sel)
            c1, c2 = cruce_genetico(p1, p2, q)
            if random.random() < p_mut:
                c1 = mutacion(c1, n)
            if random.random() < p_mut:
                c2 = mutacion(c2, n)
            new_pop.extend([c1, c2])

        pop = new_pop[:pop_size]
        pop_gpu = cp.asarray(pop, dtype=cp.int32)
        times.append(time.time() - t0)

    total = time.time() - start_all
    print(
        f"Total: {total:.2f}s | Mejor fitness: {best_fit:.2f} | Cantidad de generaciones: {gens}"
    )
    return best_global, best_fit, history, times, total, gens


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


def graficar_resultados(coordenadas, partidos, coalicion, titulo=None, salida=None):
    plt.figure()
    dem = [i for i, p in enumerate(partidos) if p.upper().startswith("D")]
    rep = [i for i, p in enumerate(partidos) if p.upper().startswith("R")]
    otros = [i for i in range(len(partidos)) if i not in dem + rep]

    plt.scatter(coordenadas[dem, 0], coordenadas[dem, 1], c="blue", s=20, label="Dem")
    plt.scatter(coordenadas[rep, 0], coordenadas[rep, 1], c="red", s=20, label="Rep")
    if otros:
        plt.scatter(
            coordenadas[otros, 0], coordenadas[otros, 1], c="gray", s=20, label="Otro"
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
    plt.plot(range(1, len(historial_costos) + 1), historial_costos, linewidth=1)
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness")
    plt.title("Mejores fitness por generación")
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
        "--generaciones", type=int, default=15000, help="Cantidad de generaciones"
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
        tiempos_generacion,
        tiempo_total_alg,
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
        f"CGM encontrada (quorum={args.quorum}) con fitness mínimo = {fitness_cgm:.4f}",
        flush=True,
    )
    # agregar registro a historial general
    resumen = {
        "id": timestamp,
        "quorum_req": args.quorum,
        "size_poblacion": args.size_poblacion,
        "cant_generaciones": args.generaciones,
        "prob_mutacion": args.p_mutacion,
        "prob_seleccion": args.p_seleccion,
        "tiempo_total_ejec_algoritmo": f"{tiempo_total_alg:.4f}",
        "mejor_fitness": f"{fitness_cgm:.8f}",
        "generaciones": gens,
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
        titulo=f"CGM encontrada (quorum={args.quorum}, best_fitness={fitness_cgm:.4f})",
        salida=f"cgm/{timestamp}/poligono_convexo",
    )
    graficar_historial_costos(
        historial_distancias, salida=f"cgm/{timestamp}/mejores_fitness"
    )


main()
