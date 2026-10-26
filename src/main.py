
"""
Proyecto: experimentos de algoritmos
Requisitos: Python 3.10+, matplotlib, pandas, numpy
Ejecutar desde la raíz: python src/main.py
Los experimentos usan semillas fijas para hacer reproducibles los resultados.
"""
from __future__ import annotations
import csv, itertools, math, multiprocessing as mp, os, random, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# ---------------- Ejercicio 1: estimación de pi en paralelo ----------------
def estimate_pi(seed_samples: tuple[int, int]) -> float:
    seed, samples = seed_samples
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

def parallel_speedup(samples_per_process=250_000, repeats=2):
    rows = []
    # El mismo volumen total de trabajo para cada configuración: 8 tareas.
    seeds = [(s + 100, samples_per_process) for s in range(8)]
    for processes in (1, 2, 4, 8):
        times = []
        estimates = []
        for rep in range(repeats):
            t0 = time.perf_counter()
            with mp.Pool(processes=processes) as pool:
                estimates = pool.map(estimate_pi, seeds)
            times.append(time.perf_counter() - t0)
        elapsed = min(times)
        rows.append({"procesos": processes, "tiempo_s": elapsed,
                     "speedup": None, "pi_media": float(np.mean(estimates)),
                     "muestras_totales": samples_per_process * len(seeds)})
    t1 = rows[0]["tiempo_s"]
    for row in rows:
        row["speedup"] = t1 / row["tiempo_s"]
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "ejercicio1_speedup.csv", index=False)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(df["procesos"], df["speedup"], "o-", label="Speedup medido")
    plt.plot(df["procesos"], df["procesos"], "--", label="Speedup ideal")
    plt.xticks([1, 2, 4, 8])
    plt.xlabel("Número de procesos"); plt.ylabel("Speedup (T1 / Tp)")
    plt.title("Ejercicio 1: aceleración de estimate_pi")
    plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "ejercicio1_speedup.png", dpi=160); plt.close()
    return df

# ---------------- Ejercicio 2: mochila voraz ----------------
def greedy_knapsack(items, capacity, criterion):
    # item: dict(name, value, weight)
    if criterion == "ratio":
        ordered = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)
    elif criterion == "lightest":
        ordered = sorted(items, key=lambda x: x["weight"])
    else:
        raise ValueError("criterion debe ser 'ratio' o 'lightest'")
    selected, used, value = [], 0, 0
    for item in ordered:
        if used + item["weight"] <= capacity:
            selected.append(item["name"])
            used += item["weight"]
            value += item["value"]
    return {"metodo": criterion, "seleccionados": ", ".join(selected) or "Ninguno",
            "peso_usado": used, "valor_total": value}

def knapsack_experiment():
    # Instancia diseñada: peso ligero primero toma A y deja capacidad insuficiente para C.
    items = [
        {"name": "A", "weight": 2, "value": 3},
        {"name": "B", "weight": 3, "value": 4},
        {"name": "C", "weight": 5, "value": 10},
    ]
    capacity = 5
    rows = [greedy_knapsack(items, capacity, "lightest"),
            greedy_knapsack(items, capacity, "ratio")]
    df = pd.DataFrame(rows)
    df["capacidad"] = capacity
    df.to_csv(DATA / "ejercicio2_mochila.csv", index=False)
    x = np.arange(len(df)); width = .34
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(x - width/2, df["valor_total"], width, label="Valor total")
    ax.bar(x + width/2, df["peso_usado"], width, label="Peso usado")
    ax.set_xticks(x, ["Menor peso primero", "Mayor valor/peso"])
    ax.set_ylabel("Unidades"); ax.set_title("Ejercicio 2: comparación de heurísticas")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    for container in ax.containers: ax.bar_label(container, padding=2)
    fig.tight_layout(); fig.savefig(FIG / "ejercicio2_mochila.png", dpi=160); plt.close(fig)
    return items, capacity, df

# ---------------- Utilidades TSP ----------------
def dist_matrix(points):
    p = np.asarray(points, dtype=float)
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))

def route_cost(route, D):
    return float(sum(D[route[i], route[(i+1) % len(route)]] for i in range(len(route))))

def two_opt(route, D, first_improvement=True):
    route = list(route)
    improvements = 0
    while True:
        best_route, best_cost = route, route_cost(route, D)
        found = False
        n = len(route)
        for i in range(1, n-1):
            for k in range(i+1, n):
                if i == 1 and k == n-1:
                    continue
                candidate = route[:i] + route[i:k+1][::-1] + route[k+1:]
                cost = route_cost(candidate, D)
                if cost < best_cost - 1e-10:
                    best_route, best_cost, found = candidate, cost, True
                    if first_improvement:
                        break
            if found and first_improvement: break
        if not found: break
        route = best_route
        improvements += 1
    return route, route_cost(route, D), improvements

def nearest_neighbour(D, start):
    n = len(D); route = [start]; remaining = set(range(n)); remaining.remove(start)
    while remaining:
        nxt = min(remaining, key=lambda j: D[route[-1], j])
        route.append(nxt); remaining.remove(nxt)
    return route

def random_route(n, rng):
    route = list(range(n)); rng.shuffle(route); return route

def local_search_experiment(instances=40, n=12):
    rows = []
    rng = random.Random(20261024)
    for inst in range(instances):
        points = [(rng.random()*100, rng.random()*100) for _ in range(n)]
        D = dist_matrix(points)
        random_start = random_route(n, rng)
        nn_start = nearest_neighbour(D, rng.randrange(n))
        for label, start in (("Aleatoria", random_start), ("Vecino más cercano", nn_start)):
            route, cost, improvements = two_opt(start, D)
            rows.append({"instancia": inst+1, "inicio": label, "mejoras": improvements,
                         "costo_final": cost, "costo_inicial": route_cost(start, D)})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "ejercicio3_2opt.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    labels = ["Aleatoria", "Vecino más cercano"]
    axes[0].boxplot([df[df.inicio == l]["mejoras"] for l in labels], tick_labels=labels)
    axes[0].set_ylabel("Número de mejoras aceptadas"); axes[0].set_title("Mejoras hasta óptimo local")
    axes[1].boxplot([df[df.inicio == l]["costo_final"] for l in labels], tick_labels=labels)
    axes[1].set_ylabel("Longitud del tour"); axes[1].set_title("Calidad final (menor es mejor)")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "ejercicio3_2opt.png", dpi=160); plt.close(fig)
    return df

# ---------------- Ejercicio 4: Recocido simulado con movimientos 2-opt ----------------
def simulated_annealing(D, schedule, iterations=2500, seed=1, t0=100.0, alpha=.995):
    rng = random.Random(seed); n = len(D)
    current = random_route(n, rng); current_cost = route_cost(current, D)
    best, best_cost = current[:], current_cost
    history = []
    start_time = time.perf_counter()
    for step in range(iterations):
        if schedule == "exponencial":
            T = t0 * (alpha ** step)
        elif schedule == "lineal":
            T = max(0.01, t0 * (1 - step / iterations))
        elif schedule == "lento":
            T = t0 / (1 + math.log1p(step))
        else:
            raise ValueError(schedule)
        i = rng.randrange(1, n-1); k = rng.randrange(i+1, n)
        candidate = current[:i] + current[i:k+1][::-1] + current[k+1:]
        cand_cost = route_cost(candidate, D)
        delta = cand_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            current, current_cost = candidate, cand_cost
        if current_cost < best_cost:
            best, best_cost = current[:], current_cost
        history.append({"iteracion": step+1, "esquema": schedule,
                        "costo_actual": current_cost, "mejor_costo": best_cost, "temperatura": T})
    elapsed = time.perf_counter() - start_time
    # Aplicar 2-opt al mejor estado devuelto para hacer comparable la mejora local.
    polished, polished_cost, polish_improvements = two_opt(best, D)
    return {"esquema": schedule, "tiempo_s": elapsed, "costo_final": best_cost,
            "costo_con_2opt": polished_cost, "mejoras_2opt_final": polish_improvements}, history

def annealing_experiment(instances=8, n=12, iterations=2500):
    rng = random.Random(404)
    rows, histories = [], []
    schedules = ["lineal", "exponencial", "lento"]
    for inst in range(instances):
        points = [(rng.random()*100, rng.random()*100) for _ in range(n)]
        D = dist_matrix(points)
        for j, schedule in enumerate(schedules):
            result, hist = simulated_annealing(D, schedule, iterations=iterations,
                                               seed=9000 + inst*10 + j)
            result["instancia"] = inst+1; rows.append(result)
            for h in hist:
                h["instancia"] = inst+1; histories.append(h)
    df = pd.DataFrame(rows); hdf = pd.DataFrame(histories)
    df.to_csv(DATA / "ejercicio4_recocido_resumen.csv", index=False)
    hdf.to_csv(DATA / "ejercicio4_recocido_historial.csv", index=False)
    # Curva media del mejor costo por iteración, promediada entre instancias.
    fig, ax = plt.subplots(figsize=(8, 5))
    for schedule in schedules:
        subset = hdf[hdf.esquema == schedule].groupby("iteracion")["mejor_costo"].mean()
        ax.plot(subset.index, subset.values, label=schedule.capitalize())
    ax.set_xlabel("Iteraciones"); ax.set_ylabel("Mejor costo encontrado (promedio)")
    ax.set_title("Ejercicio 4: curvas de convergencia")
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "ejercicio4_convergencia.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    means = df.groupby("esquema")[["tiempo_s", "costo_final"]].mean().reindex(schedules)
    axes[0].bar(means.index, means["tiempo_s"]); axes[0].set_title("Tiempo medio"); axes[0].set_ylabel("Segundos")
    axes[1].bar(means.index, means["costo_final"]); axes[1].set_title("Costo final medio"); axes[1].set_ylabel("Longitud (menor es mejor)")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "ejercicio4_tiempo_calidad.png", dpi=160); plt.close(fig)
    return df, hdf

# ---------------- Ejercicio 5: MST approximation + optimal brute force ----------------
def mst_prim(D):
    n = len(D); in_tree = {0}; parent = {}
    while len(in_tree) < n:
        best = (float("inf"), None, None)
        for u in in_tree:
            for v in range(n):
                if v not in in_tree and D[u, v] < best[0]:
                    best = (D[u, v], u, v)
        _, u, v = best
        parent.setdefault(u, []).append(v); in_tree.add(v)
    return parent

def preorder_mst(tree, root=0):
    result = []
    def visit(u):
        result.append(u)
        for v in tree.get(u, []): visit(v)
    visit(root)
    return result

def optimal_tsp(D):
    n = len(D); best_cost = float("inf"); best_route = None
    for perm in itertools.permutations(range(1, n)):
        route = (0,) + perm
        cost = route_cost(route, D)
        if cost < best_cost:
            best_cost, best_route = cost, route
    return list(best_route), best_cost

def mst_experiment(instances=100, n=9):
    rng = random.Random(777)
    rows = []
    for inst in range(instances):
        points = [(rng.random()*100, rng.random()*100) for _ in range(n)]
        D = dist_matrix(points)
        mst_route = preorder_mst(mst_prim(D))
        mst_cost = route_cost(mst_route, D)
        polished_route, polished_cost, _ = two_opt(mst_route, D)
        opt_route, opt_cost = optimal_tsp(D)
        # Comparación de métodos sobre la misma instancia.
        random_start = random_route(n, rng)
        local_route, local_cost, _ = two_opt(random_start, D)
        nn_route = nearest_neighbour(D, rng.randrange(n))
        nn_route, nn_cost, _ = two_opt(nn_route, D)
        sa_result, _ = simulated_annealing(D, "exponencial", iterations=1200,
                                            seed=10000+inst, t0=100, alpha=.994)
        rows.append({"instancia": inst+1, "mst": mst_cost, "mst_2opt": polished_cost,
                     "busqueda_local_aleatoria": local_cost, "busqueda_local_vecino": nn_cost,
                     "recocido_simulado": sa_result["costo_final"], "optimo": opt_cost,
                     "razon_mst": mst_cost/opt_cost, "razon_mst_2opt": polished_cost/opt_cost})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "ejercicio5_mst_100_instancias.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].hist(df["razon_mst"], bins=14, edgecolor="black")
    axes[0].axvline(2, color="red", linestyle="--", label="Límite teórico = 2")
    axes[0].set_xlabel("Costo MST / costo óptimo"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Razón de aproximación (100 instancias)"); axes[0].legend()
    cols = ["mst", "mst_2opt", "busqueda_local_aleatoria", "busqueda_local_vecino", "recocido_simulado", "optimo"]
    labels = ["MST", "MST + 2-opt", "2-opt aleatoria", "Vecino + 2-opt", "Recocido", "Óptimo"]
    axes[1].boxplot([df[c] for c in cols], tick_labels=labels, showfliers=False)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Costo del tour"); axes[1].set_title("Comparación de métodos")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "ejercicio5_mst_comparacion.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(df["instancia"], df["razon_mst"], alpha=.8, label="MST")
    ax.scatter(df["instancia"], df["razon_mst_2opt"], alpha=.8, label="MST + 2-opt")
    ax.axhline(2, color="red", linestyle="--", label="Límite teórico = 2")
    ax.set_xlabel("Instancia"); ax.set_ylabel("Costo / óptimo")
    ax.set_title("Razón de aproximación por instancia"); ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "ejercicio5_razones.png", dpi=160); plt.close(fig)
    return df

def main():
    print("Ejercicio 1: estimate_pi")
    e1 = parallel_speedup()
    print(e1.to_string(index=False))
    print("\nEjercicio 2: mochila voraz")
    _, cap, e2 = knapsack_experiment(); print(e2.to_string(index=False))
    print("\nEjercicio 3: 2-opt")
    e3 = local_search_experiment()
    print(e3.groupby("inicio")[["mejoras", "costo_inicial", "costo_final"]].mean().round(3))
    print("\nEjercicio 4: recocido simulado")
    e4, _ = annealing_experiment()
    print(e4.groupby("esquema")[["tiempo_s", "costo_final", "costo_con_2opt"]].mean().round(3))
    print("\nEjercicio 5: MST")
    e5 = mst_experiment()
    print(e5[["razon_mst", "razon_mst_2opt"]].describe().round(3))
    print("\nArchivos guardados en results/")

if __name__ == "__main__":
    mp.freeze_support()
    main()
