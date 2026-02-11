#!/usr/bin/env python3
"""
plot_bands.py

Uso:
  python plot_bands.py

Asume los ficheros:
  - eigenvalues.txt    (cada fila = un k-point, N columnas = bandas)
  - kpath_frac.txt     (cada fila = k_frac_x k_frac_y k_frac_z, comentarios con '#')
  - kpath_labels.txt   (lineas "LABEL  index" o "# label index")

Opcional:
  - Red.txt            (3x3, vectores directos en Å como filas) para convertir k_frac->k_cart
                       si no se proporciona, se trabaja en espacio fraccional euclidiano.

Salida:
  - muestra la figura y guarda "bands.png"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --------------------------
# Utilidades
# --------------------------
def read_eigenvalues(fn):
    """Lee eigenvalues.txt: cada fila = k, columnas = bandas (ignora líneas que empiecen por '#')."""
    try:
        data = np.loadtxt(fn, comments="#")
    except Exception as e:
        raise RuntimeError(f"No pude leer '{fn}': {e}")
    if data.ndim == 1:
        # solo una banda -> convertir a (Nk,1)
        data = data.reshape(-1, 1)
    return data  # shape (Nk, Nbands)

def read_kpoints_frac(fn):
    kf = []
    with open(fn, "r", encoding="latin-1") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) != 3:
                raise ValueError(f"{fn}:{lineno}: se esperan 3 columnas por línea.")
            kf.append([float(toks[0]), float(toks[1]), float(toks[2])])
    return np.array(kf, dtype=float)

def read_labels(fn):
    """
    Lee kpath_labels.txt, formato (ejemplo):
      # label  index
      G 0
      K 10
    Devuelve lista de (label, index) ordenada por index ascendente.
    """
    labs = []
    with open(fn, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 2:
                continue
            label = toks[0]
            try:
                idx = int(toks[1])
            except ValueError:
                # intentar con último token si hay comentarios extra
                try:
                    idx = int(toks[-1])
                except ValueError:
                    raise ValueError(f"{fn}:{lineno}: índice de etiqueta no es entero.")
            labs.append((label, idx))
    # ordenar por índice (por si vienen desordenadas)
    labs.sort(key=lambda x: x[1])
    return labs

def reciprocal_lattice_from_cell(cell):
    """
    cell: array (3,3) filas = a1,a2,a3
    devuelve B: array (3,3) filas = b1,b2,b3 en la convención b_i·a_j = 2π δ_ij
    """
    a1, a2, a3 = cell
    V = float(np.dot(a1, np.cross(a2, a3)))
    if abs(V) < 1e-12:
        raise RuntimeError("Volumen de celda muy pequeño o nulo.")
    b1 = 2*np.pi * np.cross(a2, a3) / V
    b2 = 2*np.pi * np.cross(a3, a1) / V
    b3 = 2*np.pi * np.cross(a1, a2) / V
    return np.vstack([b1, b2, b3])

def kfrac_to_kcart(kfrac, B):
    """kfrac: (Nk,3) ; B: (3,3) filas=b1,b2,b3 -> devuelve (Nk,3) cartesianas"""
    return np.asarray(kfrac) @ B  # (Nk,3) @ (3,3) -> (Nk,3)

def cumulative_kdist(kcart):
    """Devuelve array (Nk,) de distancias acumuladas a lo largo de la ruta (empezando en 0)."""
    diffs = np.diff(kcart, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    cum = np.zeros(kcart.shape[0], dtype=float)
    cum[1:] = np.cumsum(seglen)
    return cum

# --------------------------
# Plot principal
# --------------------------
def plot_bands(evals_file="eigenvalues.txt",
               kfrac_file="kpath_frac.txt",
               labels_file="kpath_labels.txt",
               cell_file="Red.txt",
               reciprocal_cell=None,
               out_png="bands.png",
               figsize=(8,6),
               lw=1.0):
    # Leer
    E = read_eigenvalues(evals_file)   # (Nk, N)
    kfrac = read_kpoints_frac(kfrac_file)  # (Nk,3)
    labs = read_labels(labels_file) if os.path.exists(labels_file) else []

    Nk, Nbands = E.shape

    if kfrac.shape[0] != Nk:
        raise RuntimeError(f"Número de k en {kfrac_file} ({kfrac.shape[0]}) distinto a filas en {evals_file} ({Nk}).")

    # construir B (recíproca)
    if reciprocal_cell is None:
        if cell_file is not None and os.path.exists(cell_file):
            cell = np.loadtxt(cell_file)
            if cell.shape != (3,3):
                raise RuntimeError(f"{cell_file} debe ser 3x3.")
            B = reciprocal_lattice_from_cell(cell)  # filas b1,b2,b3
        else:
            # no hay cell ni B: usar espacio fraccional directo (advertencia)
            print("AVISO: no se proporcionó Red.txt ni matriz recíproca. Se usará espacio fraccional como métrica euclidiana.")
            B = np.eye(3)
    else:
        B = np.asarray(reciprocal_cell)
        if B.shape != (3,3):
            raise RuntimeError("reciprocal_cell debe ser (3,3).")

    # convertir k
    kcart = kfrac_to_kcart(kfrac, B)  # (Nk,3)
    kdist = cumulative_kdist(kcart)    # (Nk,)

    # preparar figura
    fig, ax = plt.subplots(1,1, figsize=figsize)
    for n in range(Nbands):
        ax.plot(kdist, E[:, n], lw=lw, solid_capstyle='butt')

    # líneas verticales y etiquetas
    if labs:
        tick_pos = []
        tick_labels = []
        for label, idx in labs:
            if idx < 0 or idx >= Nk:
                print(f"Advertencia: etiqueta {label} con índice {idx} fuera de rango (Nk={Nk}). Se ignora.")
                continue
            x = kdist[idx]
            tick_pos.append(x)
            tick_labels.append(label)
            ax.axvline(x=x, color='k', lw=0.6, linestyle='--')
        # colocar ticks en eje x
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks([])

    ax.set_xlabel(r'k (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.grid(False)
    ax.set_xlim(kdist[0], kdist[-1])
    # opcional: ajustar límites y estilo
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Guardado plot en '{out_png}'")
    plt.show()

# --------------------------
# Script ejecutable
# --------------------------
if __name__ == "__main__":
    # rutas por defecto (ajusta si es necesario)
    evals_fn = "eigenvalues.txt"
    kfrac_fn = "kpath_frac.txt"
    labels_fn = "kpath_labels.txt"
    cell_fn = "Red.txt"  # si no existe, se usará métrica fraccional

    if not os.path.exists(evals_fn):
        print(f"ERROR: no encuentro '{evals_fn}' en el directorio actual.")
        sys.exit(1)
    if not os.path.exists(kfrac_fn):
        print(f"ERROR: no encuentro '{kfrac_fn}' en el directorio actual.")
        sys.exit(1)

    plot_bands(evals_file=evals_fn,
               kfrac_file=kfrac_fn,
               labels_file=labels_fn,
               cell_file=cell_fn,
               out_png="bands1.png")