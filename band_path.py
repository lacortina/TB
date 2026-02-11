#!/usr/bin/env python3
# band_path.py
#
# Genera una ruta de alta simetría con seekpath a partir de:
#   - Red.txt        (3x3, vectores de red en Å, filas = a1,a2,a3)
#   - Posiciones.txt (simbolo  x  y  z   en Å)
#
# Salidas:
#   - kpath_frac.txt
#   - kpath_labels.txt

import numpy as np
import seekpath
from ase.data import atomic_numbers as Z_MAP

# ==========================
# PARÁMETROS
# ==========================

POINTS_PER_SEGMENT = 20

# ==========================
# LEER RED
# ==========================

cell = np.loadtxt("Red.txt")

if cell.shape != (3, 3):
    raise ValueError("Red.txt debe contener exactamente 3 vectores de red (3x3)")

# ==========================
# LEER POSICIONES
# ==========================

symbols = []
positions_cart = []

with open("Posiciones.txt") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        s, x, y, z = line.split()
        symbols.append(s)
        positions_cart.append([float(x), float(y), float(z)])

positions_cart = np.array(positions_cart, dtype=float)

if len(symbols) == 0:
    raise RuntimeError("No se leyeron átomos desde Posiciones.txt")

# ==========================
# MAPA DE ESPECIES → Z
# ==========================

try:
    atomic_numbers = [Z_MAP[s] for s in symbols]
except KeyError:
    missing = sorted(set(symbols) - set(Z_MAP.keys()))
    raise ValueError(f"Símbolos químicos no reconocidos: {missing}")

# ==========================
# CARTESIANAS → FRACCIONARIAS
# ==========================

# Usar solve es más robusto que invertir la matriz
positions_frac = np.linalg.solve(cell.T, positions_cart.T).T

# Normalizar a [0,1)
positions_frac = positions_frac % 1.0

if not np.isfinite(positions_frac).all():
    raise RuntimeError("Coordenadas fraccionarias inválidas")

# ==========================
# ESTRUCTURA PARA SEEKPATH
# ==========================

structure = (
    cell,
    positions_frac,
    atomic_numbers
)

# ==========================
# SEEKPATH
# ==========================

try:
    path_data = seekpath.get_path(
        structure,
        recipe="hpkot",
        with_time_reversal=True
    )
except Exception as e:
    raise RuntimeError("seekpath no pudo determinar la simetría") from e

# ==========================
# CONSTRUIR RUTA DISCRETA
# ==========================

kpoints = []
labels = []
label_positions = []

k_index = 0

for start, end in path_data["path"]:
    k_start = np.array(path_data["point_coords"][start])
    k_end   = np.array(path_data["point_coords"][end])

    segment = np.linspace(
        k_start,
        k_end,
        POINTS_PER_SEGMENT,
        endpoint=False
    )

    kpoints.extend(segment)
    labels.append(start)
    label_positions.append(k_index)

    k_index += len(segment)

# Último punto
final_label = path_data["path"][-1][1]
kpoints.append(path_data["point_coords"][final_label])
labels.append(final_label)
label_positions.append(len(kpoints) - 1)

kpoints = np.array(kpoints)

# ==========================
# SALIDAS
# ==========================

np.savetxt(
    "kpath_frac.txt",
    kpoints,
    fmt="%.10f",
    header="k-points fraccionarios (base recíproca, seekpath)"
)

with open("kpath_labels.txt", "w") as f:
    f.write("# label  index\n")
    for lab, idx in zip(labels, label_positions):
        f.write(f"{lab}  {idx}\n")

print("\nRuta de alta simetría:")
print(" -> ".join(labels))
print(f"Total de k-points: {len(kpoints)}")
print("Hecho.")
