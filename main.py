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
    print(">> ¡Usando CuPy (GPU) para mejorar rendimiento!")
except ImportError:
    import numpy as cp

    usando_cupy = False
    print(">> CuPy no disponible. Usando NumPy como respaldo.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def poligono_convexo(puntos):
    puntos_ordenados = sorted(puntos, key=lambda p: (p[0], p[1]))

    def producto_cruzado(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    parte_inferior, parte_superior = [], []

    for x, y, indice_x in puntos_ordenados:
        while (
            len(parte_inferior) >= 2
            and producto_cruzado(parte_inferior[-2], parte_inferior[-1], (x, y)) <= 0
        ):
            parte_inferior.pop()
        parte_inferior.append((x, y, indice_x))

    for x, y, indice_x in reversed(puntos_ordenados):
        while (
            len(parte_superior) >= 2
            and producto_cruzado(parte_superior[-2], parte_superior[-1], (x, y)) <= 0
        ):
            parte_superior.pop()
        parte_superior.append((x, y, indice_x))

    casco = parte_inferior[:-1] + parte_superior[:-1]

    return [indice_x for _, _, indice_x in casco]


def matriz_de_distancias(coordenadas):
    coordenadas_gpu = cp.asarray(coordenadas)
    diferencias_punto_a_punto = (
        coordenadas_gpu[:, None, :] - coordenadas_gpu[None, :, :]
    )
    return cp.linalg.norm(diferencias_punto_a_punto, axis=2)


def poblacion_inicial(congresistas, quorum_requerido, size_poblacion):
    if quorum_requerido > congresistas:
        raise ValueError(
            "El quorum no puede ser mayor al número total de congresistas."
        )
    return [
        random.sample(range(congresistas), quorum_requerido)
        for _ in range(size_poblacion)
    ]


def torneo_de_seleccion(poblacion, distancias_fitness_poblacion, p):
    indices_ordenados = sorted(
        range(len(poblacion)), key=lambda i: distancias_fitness_poblacion[i]
    )
    n = len(poblacion)
    probabilidades = [p * ((1 - p) ** i) for i in range(n)]
    suma = sum(probabilidades)
    probabilidades = [pi / suma for pi in probabilidades]
    elegido = random.choices(indices_ordenados, weights=probabilidades, k=1)[0]
    return poblacion[elegido].copy()


def cruce_genetico(padre, madre, largo_cromosoma):
    punto_de_cruce = random.randint(1, largo_cromosoma - 1)
    hijo_1 = (
        padre[:punto_de_cruce]
        + [gen for gen in madre if gen not in padre[:punto_de_cruce]][
            : largo_cromosoma - punto_de_cruce
        ]
    )
    hijo_2 = (
        madre[: largo_cromosoma - punto_de_cruce]
        + [
            gen for gen in padre if gen not in madre[: largo_cromosoma - punto_de_cruce]
        ][:punto_de_cruce]
    )
    return hijo_1, hijo_2


def mutacion(individuo, genes_disponibles):
    genes_fuera_del_individuo = list(set(range(genes_disponibles)) - set(individuo))
    if genes_fuera_del_individuo:
        individuo[random.randrange(len(individuo))] = random.choice(
            genes_fuera_del_individuo
        )
    return individuo


def algoritmo_genetico(
    coordenadas_congresistas,
    quorum_requerido,
    size_poblacion,
    cantidad_generaciones,
    probabilidad_de_mutacion,
    probabilidad_de_seleccion,
):
    cantidad_congresistas = coordenadas_congresistas.shape[0]
    matriz_distancias_congresistas = matriz_de_distancias(coordenadas_congresistas)
    poblacion = poblacion_inicial(
        cantidad_congresistas, quorum_requerido, size_poblacion
    )
    poblacion_gpu = cp.asarray(poblacion, dtype=cp.int32)

    coalicion_ganadora_minima, distancia_coalicion_ganadora_minima = None, float("inf")
    (
        mejores_distancias_generacionales,
        tiempos_de_calculo_de_distancias_por_generacion,
    ) = [], []
    tiempo_inicio_algoritmo = time.time()

    for generacion in range(1, cantidad_generaciones + 1):
        tiempo_inicio_calculo_distancias_generacion = time.time()

        filas = poblacion_gpu[:, :, None]  # revisar proposito de esto
        columnas = poblacion_gpu[:, None, :]
        submatriz_distancias_individuos = matriz_distancias_congresistas[
            filas, columnas
        ]

        distancias_individuos_gpu = cp.sum(
            cp.triu(submatriz_distancias_individuos, k=1), axis=(1, 2)
        )
        distancias_individuos = (
            distancias_individuos_gpu.get()
            if usando_cupy
            else distancias_individuos_gpu
        )


        mejor_indice = (
            cp.argmin(distancias_individuos_gpu).get()
            if usando_cupy
            else int(cp.argmin(distancias_individuos_gpu))  # int -> [0]
        )
        mejor_distancia_generacion = float(distancias_individuos[mejor_indice])
        mejores_distancias_generacionales.append(mejor_distancia_generacion)

    

        if mejor_distancia_generacion < distancia_coalicion_ganadora_minima:
            distancia_coalicion_ganadora_minima = mejor_distancia_generacion
            coalicion_ganadora_minima = poblacion[mejor_indice].copy()

        distancias_lista = distancias_individuos.tolist()
        nueva_poblacion = [coalicion_ganadora_minima]
        # agregar el mejor de poblacion anterior paso 9 -> paso 2
        while len(nueva_poblacion) < size_poblacion:
            padre = torneo_de_seleccion(
                poblacion, distancias_lista, p=probabilidad_de_seleccion
            )
            madre = torneo_de_seleccion(
                poblacion, distancias_lista, p=probabilidad_de_seleccion
            )

            individuo_1, individuo_2 = cruce_genetico(padre, madre, quorum_requerido)
            if random.random() < probabilidad_de_mutacion:
                individuo_1 = mutacion(individuo_1, cantidad_congresistas)
            if random.random() < probabilidad_de_mutacion:
                individuo_2 = mutacion(individuo_2, cantidad_congresistas)
            nueva_poblacion.extend([individuo_1, individuo_2])
        poblacion = nueva_poblacion[:size_poblacion]
        poblacion_gpu = cp.asarray(poblacion, dtype=cp.int32)
        tiempo_fin_calculo_distancias_generacion = time.time()
        tiempo_calculo = (
            tiempo_fin_calculo_distancias_generacion
            - tiempo_inicio_calculo_distancias_generacion
        )
        tiempos_de_calculo_de_distancias_por_generacion.append(tiempo_calculo)
        print(
            f"Generacion {generacion}/{cantidad_generaciones}: tiempo calculo de distancias {tiempo_calculo:.6f}s, mejor fitness {mejor_distancia_generacion:.12f}"
        )



    total_tiempo_algoritmo = time.time() - tiempo_inicio_algoritmo
    print(
        f">> Tiempo de demora del algoritmo genético: {total_tiempo_algoritmo:.2f}s",
        flush=True,
    )
    return (
        coalicion_ganadora_minima,
        distancia_coalicion_ganadora_minima,
        mejores_distancias_generacionales,
        tiempos_de_calculo_de_distancias_por_generacion,
        total_tiempo_algoritmo,
    )


def json_a_posiciones(ruta_json, indice_votacion=0):
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


# --- Plotting ---


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

    cgm, fitness_cgm, historial_distancias, tiempos_generacion, tiempo_total_alg = (
        algoritmo_genetico(
            coordenadas,
            quorum_requerido=args.quorum,
            size_poblacion=args.size_poblacion,
            cantidad_generaciones=args.generaciones,
            probabilidad_de_mutacion=args.p_mutacion,
            probabilidad_de_seleccion=args.p_seleccion,
        )
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


"""
- La probabilidad de mutacion
"""
