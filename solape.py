#!/usr/bin/env python3
"""
generate_and_filter_sk_improved_solape.py

Versión modificada de generate_and_filter_sk.py con:
 - búsqueda per-pair de la imagen mínima (PBC),
 - salida escrita en 'solape.txt' en lugar de 'sk_params.txt',
 - NO incluye términos on-site (E_s_*, E_p_*, E_d_*) en la salida final.

Mantiene la estructura y ficheros de entrada/salida del script original salvo esos cambios.
"""
import os, sys
import numpy as np
import itertools
from collections import Counter, defaultdict
import math

# ---------------- Usuario puede cambiar esto --------------
TOL = 2            # tolerancia numérica para comparaciones (ajusta según tus unidades)
# -------------------------------------------------------------

# nombres por defecto
RED_FILE = "Red.txt"
POS_FILE = "Posiciones.txt"
ORB_FILE = "orbitales.txt"
SK_FILE = "sk_params.txt"   # archivo original de parámetros (si existe)
OUT_FILE = "solape.txt"     # ahora escribimos solape.txt

# Requisitos de tipos para parámetros Slater-Koster
param_requires = {
    'Vss': ('s','s'),
    'Vsp': ('s','p'),
    'Vsds': ('s','d'),
    'Vpp_sigma': ('p','p'),
    'Vpp_pi': ('p','p'),
    'Vpds': ('p','d'),
    'Vpdp': ('p','d'),
    'Vdds': ('d','d'),
    'Vddp': ('d','d'),
    'Vddd': ('d','d'),
}

# -------------------- Lectura de ficheros ---------------------

def generate_R_list_up_to_cutoff(lattice, cutoff, max_shell=6):
    """
    Genera lista de R = n1*a1 + n2*a2 + n3*a3 con |R| <= cutoff,
    barrido en n1,n2,n3 hasta un número de 'shells' razonable.
    Devuelve (R_list, combos).
    """
    vec_norms = np.linalg.norm(lattice, axis=1)
    min_vec = vec_norms.min()
    N = int(math.ceil(cutoff / min_vec)) + 1
    N = min(N, max_shell)
    R_list = []
    combos = []
    for n1 in range(-N, N+1):
        for n2 in range(-N, N+1):
            for n3 in range(-N, N+1):
                R = n1*lattice[0] + n2*lattice[1] + n3*lattice[2]
                if np.linalg.norm(R) <= cutoff + 1e-12:
                    R_list.append(R)
                    combos.append((n1, n2, n3))
    zero_idx = None
    for idx,R in enumerate(R_list):
        if np.linalg.norm(R) < 1e-12:
            zero_idx = idx
            break
    if zero_idx is None:
        R_list.insert(0, np.zeros(3))
        combos.insert(0, (0,0,0))
    elif zero_idx != 0:
        R0 = R_list.pop(zero_idx)
        c0 = combos.pop(zero_idx)
        R_list.insert(0, R0)
        combos.insert(0, c0)
    return R_list, combos

def read_lattice(fn):
    vecs = []
    with open(fn, 'r') as f:
        for line in f:
            line = line.split('#',1)[0].strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 3: continue
            vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(vecs) < 3:
        raise RuntimeError(f"{fn} debe contener al menos 3 vectores (filas con 3 floats).")
    return np.array(vecs[:3], dtype=float)

def read_positions_cartesian(fn):
    pos = []
    with open(fn,'r') as f:
        for line in f:
            line = line.split('#',1)[0].strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 4:
                sym = parts[0]
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                pos.append((sym, coords))
    if not pos:
        raise RuntimeError(f"No se han leído posiciones desde {fn}")
    return pos

def read_orbitals(fn):
    orb_map = {}
    with open(fn,'r') as f:
        for line in f:
            line = line.split('#',1)[0].strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                s = parts[0]
                orbs = [p.strip() for p in parts[1:] if p.strip()]
                orb_map[s] = orbs
    return orb_map

def read_sk_params(fn):
    params = {}
    if not os.path.exists(fn):
        return params
    with open(fn,'r') as f:
        for line in f:
            s = line.split('#',1)[0].strip()
            if not s: continue
            parts = s.split()
            if len(parts) >= 2:
                key = parts[0]
                val = " ".join(parts[1:])
                params[key] = val
    return params

# ----------------- utilitarios orbitales / parametros ----------

def orbital_types_from_list(orb_list):
    tset = set()
    for orb in orb_list:
        if not orb: continue
        c = orb[0].lower()
        if c in ('s','p','d'):
            tset.add(c)
    return tset

def generate_possible_params(types_list, orb_map):
    """
    Genera todas las claves posibles (incluye onsite y pares).
    """
    params = set()
    params.update(["Delta", "m", "tol"])

    # Onsite
    for t in types_list:
        orbs = orb_map.get(t, [])
        tset = orbital_types_from_list(orbs)
        if 's' in tset:
            params.add(f"E_s_{t}")
        if 'p' in tset:
            params.add(f"E_p_{t}")
        if 'd' in tset:
            params.add(f"E_d_{t}")

    # Pares (con distinción según tipo de orbital requerido)
    for A in types_list:
        for B in types_list:
            la = orbital_types_from_list(orb_map.get(A, []))
            lb = orbital_types_from_list(orb_map.get(B, []))
            if not la or not lb:
                continue

            for pname, (req1, req2) in param_requires.items():
                if req1 == req2:
                    if req1 in la and req2 in lb:
                        pk = "_".join(sorted([A, B]))
                        params.add(f"{pname}_{pk}")
                else:
                    if req1 in la and req2 in lb:
                        params.add(f"{pname}_{A}_{B}")
                    if A != B and req2 in la and req1 in lb:
                        params.add(f"{pname}_{A}_{B}")

    return params

# ----------------- búsqueda por imagen mínima (PBC) ------------
EPS_ZERO = 1e-12  # umbral para considerar una distancia = 0

def generate_R_list_including_zero(lattice, coeffs=(-1,0,1)):
    """Genera lista de vectores R y combinaciones (n1,n2,n3) incluyendo la imagen 0."""
    R_list = []
    combos = []
    R_list.append(np.zeros(3, dtype=float))
    combos.append((0,0,0))
    for n1 in coeffs:
        for n2 in coeffs:
            for n3 in coeffs:
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                R = n1*lattice[0] + n2*lattice[1] + n3*lattice[2]
                R_list.append(R)
                combos.append((n1,n2,n3))
    return R_list, combos

def nearest_image_events(cart_positions, symbols, lattice, coeffs=None,
                         cutoff=None, include_self_images=True):
    """
    Para cada i busca j (incluye j==i si include_self_images=True) y la
    imagen R que minimiza |r_i - (r_j + R)|.
    - Si cutoff no es None, genera R_list hasta esa distancia.
    - include_self_images controla si se consideran pares con j==i:
        * True  -> se permiten imágenes j==i con R != 0 (se ignora R == 0)
        * False -> no se consideran pares j==i en absoluto
    """
    events = []
    n = len(cart_positions)
    if cutoff is not None:
        R_list, combos = generate_R_list_up_to_cutoff(lattice, cutoff)
    else:
        if coeffs is None:
            coeffs = (-1,0,1)
        R_list, combos = generate_R_list_including_zero(lattice, coeffs)

    for i in range(n):
        dmin_i = None
        best_partners = []
        for j in range(n):
            # si no queremos tratar pares i==j en absoluto, saltarlos
            if i == j and not include_self_images:
                continue

            best_d_pair = None
            best_k = []

            for k, R in enumerate(R_list):
                # Si es exactamente la misma celda y la misma imagen R=0, ignorar:
                if i == j and np.linalg.norm(R) <= EPS_ZERO:
                    continue

                dvec = cart_positions[i] - (cart_positions[j] + R)
                d = np.linalg.norm(dvec)

                # ignorar distancias prácticamente cero (evitar contar el mismo átomo)
                if d <= EPS_ZERO:
                    continue

                if (best_d_pair is None) or (d < best_d_pair - TOL):
                    best_d_pair = d
                    best_k = [k]
                elif abs(d - best_d_pair) <= TOL:
                    best_k.append(k)

            if best_d_pair is None:
                continue

            if (dmin_i is None) or (best_d_pair < dmin_i - TOL):
                dmin_i = best_d_pair
                best_partners = [(j, best_k, best_d_pair)]
            elif abs(best_d_pair - dmin_i) <= TOL:
                best_partners.append((j, best_k, best_d_pair))

        for (j, klist, best_d_for_pair) in best_partners:
            for k in klist:
                R = R_list[k]
                combo = combos[k]
                evtype = 'intra' if combo == (0,0,0) else 'inter'
                if i == j and combo != (0,0,0):
                    evtype = 'self-image'
                A = symbols[i]; B = symbols[j]
                events.append({
                    'type': evtype,
                    'i': i,
                    'j': j,
                    'dist': best_d_for_pair,
                    'combo': combo,
                    'R': R,
                    'species': tuple(sorted((A, B)))
                })
    return events

# ----------------- filtrado por FACTOR * d_min_global ----------------
def filter_events_by_factor(events):
    """
    - calcula d_global = min(ev.dist)
    - devuelve lista de eventos cuyo ev.dist <= d_global + TOL (señalando tolerancia)
    - tambien devuelve d_global y la lista completa para reporte
    """
    if not events:
        return [], None, events
    dists = np.array([ev['dist'] for ev in events])
    d_global = float(np.min(dists))
    kept = [ev for ev in events if ev['dist'] <= d_global + TOL]
    return kept, d_global, events

# ----------------- determinar parametros a mantener ----------------
def params_from_kept_events(kept_events, orb_map):
    """
    Determina qué parámetros mantener a partir de los eventos filtrados.
    IMPORTANTE: NO incluirá términos on-site (E_s_*, E_p_*, E_d_*).
    Devuelve (keep_set, species_pairs).
    """
    species_pairs = set(ev['species'] for ev in kept_events)

    # conjunto de especies presentes
    species_flat = set()
    for pair in species_pairs:
        species_flat.update(pair)

    # tipos de orbitales por especie
    orb_types = {s: orbital_types_from_list(orb_map.get(s, [])) for s in species_flat}

    keep = set()
    # siempre conservar claves globales
    keep.update(['Delta', 'm', 'tol'])

    # Pares a mantener (aplicando simetría cuando req1 == req2)
    for (A, B) in species_pairs:
        types_A = orb_types.get(A, set())
        types_B = orb_types.get(B, set())

        for pname, (req1, req2) in param_requires.items():
            # mismo tipo de orbital -> clave simétrica única (si aplica)
            if req1 == req2:
                if req1 in types_A and req2 in types_B:
                    pk = "_".join(sorted([A, B]))
                    keep.add(f"{pname}_{pk}")
            else:
                # direccional A -> B si aplica
                if req1 in types_A and req2 in types_B:
                    keep.add(f"{pname}_{A}_{B}")
                # si A != B considerar la dirección inversa B -> A
                if A != B and req1 in types_B and req2 in types_A:
                    keep.add(f"{pname}_{B}_{A}")

    return keep, species_pairs

# ----------------- escritura fichero final (sin onsite) ----------------
def write_sk_filtered(outfile, keep_keys, sk_original, d_global, kept_events):
    with open(outfile, 'w') as f:
        f.write("# solape (SK) filtered by generate_and_filter_sk_improved_solape.py\n")
        f.write("# d_min_global = %g  (Delta sera igual a esta distancia)\n" % d_global)
        f.write("# Eventos mantenidos (tipo, i, j, species, Rcombo (si aplica), distancia):\n")
        for ev in kept_events:
            if ev['type'] == 'intra':
                f.write("# intra  i=%d  j=%d  species=%s  dist=%g\n" % (ev['i'], ev['j'], str(ev['species']), ev['dist']))
            else:
                f.write("# inter  i=%d  j=%d  species=%s  combo=%s  |R|=%g  dist=%g\n" %
                        (ev['i'], ev['j'], str(ev['species']), str(ev.get('combo')), np.linalg.norm(ev.get('R')), ev['dist']))
        f.write("\n")
        # escribir claves en orden alfabetico. Delta = d_global; mantener valores originales si existian.
        for key in sorted(keep_keys):
            # OMITIR términos on-site que empiecen por E_s_, E_p_, E_d_
            if key.startswith("E_s_") or key.startswith("E_p_") or key.startswith("E_d_"):
                # saltar on-site
                continue
            if key == "Delta":
                f.write(f"Delta {d_global}\n")
            elif key in sk_original:
                f.write(f"{key} {sk_original[key]}\n")
            else:
                if key == "m":
                    f.write("m 1.0\n")
                elif key == "tol":
                    f.write("tol 1e-2\n")
                else:
                    f.write(f"{key} 0.0\n")
    print(f"Wrote filtered solape to {outfile}")

# ----------------- MAIN ----------------
def main(argv):
    global RED_FILE, POS_FILE, ORB_FILE, SK_FILE, OUT_FILE, TOL
    if len(argv) >= 2:
        RED_FILE = argv[1]
    if len(argv) >= 3:
        POS_FILE = argv[2]
    if len(argv) >= 4:
        ORB_FILE = argv[3]
    if len(argv) >= 5:
        SK_FILE = argv[4]
    if len(argv) >= 6:
        OUT_FILE = argv[5]

    # comprobar existencia
    for fn in (RED_FILE, POS_FILE, ORB_FILE):
        if not os.path.exists(fn):
            print(f"Error: no existe {fn}")
            return

    lattice = read_lattice(RED_FILE)   # filas = a1,a2,a3
    pos = read_positions_cartesian(POS_FILE)
    orb_map = read_orbitals(ORB_FILE)
    sk_original = read_sk_params(SK_FILE)

    symbols = [p[0] for p in pos]
    cart = np.vstack([p[1] for p in pos])

    # tipos presentes (para generar potenciales claves)
    species_types = sorted(list({s for s in symbols}))

    # Generar todas las claves posibles (no significa que luego se mantengan todas)
    possible_keys = generate_possible_params(species_types, orb_map)

    # calcular eventos (búsqueda per-pair de la imagen minima)
    # incluir imagenes self (i == j con R != 0) para detectar enlaces i <-> i+R
    all_events = nearest_image_events(cart, symbols, lattice, coeffs=(-1,0,1), include_self_images=True)

    if not all_events:
        print("No se detectaron eventos (distancias). No se genera fichero.")
        return

    # filtrado por FACTOR * d_min_global (aqui FACTOR implícito = 1, usamos d_global + TOL)
    kept_events, d_global, full_events = filter_events_by_factor(all_events)
    if d_global is None:
        print("Error: no hay distancias.")
        return

    print(f"Distancia global minima (d_min_global) = {d_global:.12f}")
    print(f"Eventos totales: {len(all_events)}. Eventos mantenidos tras filtro: {len(kept_events)}")

    # determinar parametros a mantener (sin onsite)
    keep_keys, species_pairs = params_from_kept_events(kept_events, orb_map)

    # Asegurar que Delta/m/tol estén presentes
    keep_keys.update(["Delta","m","tol"])

    # Si no quedo ninguno, avisar
    if not keep_keys:
        print("ATENCION: no quedaron parametros tras el filtrado. Se escribira fichero con Delta y globals solo.")
        keep_keys = set(["Delta","m","tol"])

    # escribir fichero de salida con Delta = d_global y cabecera con las distancias mantenidas
    write_sk_filtered(OUT_FILE, keep_keys, sk_original, d_global, kept_events)

    # imprimir resumen en pantalla
    print("\nResumen de especies participantes (pares):")
    for p in sorted(species_pairs):
        print("  ", p)
    print("\nClaves SK mantenidas (ejemplos):")
    for k in sorted(list(keep_keys))[:50]:
        if k.startswith("E_s_") or k.startswith("E_p_") or k.startswith("E_d_"):
            continue
        print("  ", k)
    print("\nHecho.")

if __name__ == "__main__":
    main(sys.argv)
