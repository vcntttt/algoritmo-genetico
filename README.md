
---

# 🧬 **Genetic Algorithm for the Minimum Winning Coalition (MWC) — GPU Accelerated**

### **Universidad Católica de Temuco · Departamento de Ingeniería Informática**

---

## 📖 **Overview**

This repository implements a **Genetic Algorithm (GA)** to solve the *Minimum Winning Coalition (MWC)* problem — an NP-Hard optimization task that seeks the smallest possible coalition of legislators achieving majority consensus while minimizing intra-coalition ideological distance.

Our approach follows the model proposed by **Lincolao-Venegas et al. (2023)** , integrating **GPU acceleration via CuPy** to achieve dramatic runtime improvements over conventional CPU-based heuristics.

---

## ⚙️ **Core Features**

* 🚀 **GPU-Accelerated Distance Computation:** Euclidean pairwise matrices computed entirely on GPU with **CuPy**.
* 🧠 **Deterministic & Reproducible:** Population control, elitism, and tournament selection ensure stable convergence.
* 🧩 **Real-World Political Dataset:** Uses *Voteview*’s 75th U.S. Congress rollcall RH0941234 (1975 – 1977).
* 📊 **Automatic Visualization:** Generates coalition maps, convex-hull plots, and convergence curves.
* 💾 **Benchmark Logging:** Each run appends execution metrics (fitness, time, parameters) to `benchmark.csv`.

---
## 🧩 **Problem Definition**

Given a parliament of *n* representatives positioned in a latent 2-D ideological space (as per **DW-Nominate** dimensions),  
the **Minimum Winning Coalition (MWC)** is defined as the subset **G** of size **q** that minimizes the total pairwise distance among its members:

$$
Z(G) = \sum_{i=1}^{q-1} \sum_{j=i+1}^{q} d(p_i, p_j)
$$

where \( d(p_i, p_j) \) is the Euclidean distance between representatives *i* and *j*.

The goal is therefore to identify:

$$
G^{*} = \arg\min_{G} Z(G) \quad \text{subject to} \quad |G| = q
$$

---

## 🧬 **Genetic Algorithm Implementation**

### **1. Representation**

Each chromosome encodes a candidate coalition — a list of *q* integers representing member indices.

### **2. Operators**

* **Selection:** Weighted tournament favoring lower-fitness individuals.
* **Crossover:** One-point recombination ensuring gene uniqueness.
* **Mutation:** Random substitution of one member (probability *pₘ*).

### **3. Fitness Evaluation**

Implemented on GPU:

```python
sub = D_gpu[poblacion_gpu[:, :, None], poblacion_gpu[:, None, :]]
fit_gpu = cp.sum(cp.triu(sub, 1), axis=(1, 2))
```

### **4. Parameters**

| Parameter          | Default   | Description            |
| ------------------ | --------- | ---------------------- |
| `--quorum`         | 216       | Required majority size |
| `--size_poblacion` | 38        | Population size        |
| `--generaciones`   | 20000     | Max generations        |
| `--p_mutacion`     | 0.1700019 | Mutation probability   |
| `--p_seleccion`    | 0.141     | Selection pressure     |

---

## ⚡ **Performance**

| Metric           | Genetic Algorithm (GPU)  | Ad-hoc Deterministic Algorithm [Lincolao 2023] |
| ---------------- | ------------------------ | ---------------------------------------------- |
| Average Time     | **~0.1 s** (216 members) | ~26.5 s (3 orders slower)                      |
| Best Fitness     | 9686.9383                | 9686.9383                                      |
| Convergence Rate | 91.4 %                   | —                                              |

✅ The GPU-based GA converges to the **same optimal coalition** reported in the IEEE SCCC 2023 paper but in **hundredths of a second**.

---

## 📊 **Outputs**

* **`cgm/<timestamp>/congresistas_coalicion.csv`** → Selected members.
* **`historial_costos.png`** → Fitness convergence curve.
* **`benchmark.csv`** → Run metrics (quorum, population, generations, runtime, fitness).
* **`cgm_plot.png`** → Coalition and convex hull visualization.

Example output visualization:

```
🟦 Demócratas dentro
🟥 Republicanos dentro
×  Fuera de la coalición
--  Envolvente convexa
```

---

## 🧠 **Academic Context**

This implementation corresponds to **Prueba Práctica 3 (INFO1159)** at the Universidad Católica de Temuco .
Students were required to reproduce and compare the MWC results from the IEEE SCCC 2023 paper:

> **Lincolao-Venegas, I. et al. (2023).** *An ad-hoc algorithm to find the minimum winning coalition.*
> 42nd IEEE International Conference of the Chilean Computer Science Society (SCCC 2023).
> DOI [10.1109/SCCC59417.2023.10315747](https://doi.org/10.1109/SCCC59417.2023.10315747)

---

## 💡 **Usage**

```bash
# Run the GA with default parameters
python main.py --json data.json

# Example with custom parameters
python main.py --json data.json --quorum 217 --size_poblacion 50 --generaciones 15000
```

Requires a GPU and **CuPy ≥ 13.0**.

---

## 📈 **Visualization Example**

```python
graficar_resultados(
    coordenadas,
    partidos,
    cgm,
    titulo="CGM found (quorum=216, best_fitness=9686.9383)",
)
```

Generates a 2-D scatter plot separating Democrats (blue) and Republicans (red), with convex hull overlay.

---

## 🧩 **Dependencies**

* **Python ≥ 3.10**
* **CuPy**
* **NumPy**
* **Pandas**
* **Matplotlib**

Install via:

```bash
pip install cupy-cuda12x numpy pandas matplotlib
```

---

## 🧑‍💻 **Authors**

**Equipo INFO1159 – Universidad Católica de Temuco**

Implementation made by:

- **Vicente Rivera**  
- **Vicente Álvarez**  
- **Fernando Valdés**  
- **Juan Muñoz**  
- **Juan Sepúlveda**

