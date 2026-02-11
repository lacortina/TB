#!/usr/bin/env python3
"""
select_bands_around_gap.py

(igual que antes, con salida ampliada en eigenvaluesDFT.txt indicando
qué banda es valence / conduction y en qué columna del archivo aparecen)
"""
from pathlib import Path
import re
import numpy as np
import sys
from typing import List, Tuple, Dict, Any

INPUT = Path("bandasDFT.txt")
OUT_ALL = Path("eigenvaluesallDFT.txt")
OUT_SEL = Path("eigenvaluesDFT.txt")

#Seleccionar bandas

# ----------------------------
# Constantes / validaciones
# ----------------------------
ALLOWED_ORBITALS = {
    "s", "px", "py", "pz",
    "dxy", "dxz", "dyz", "dx2_y2", "dr"
}

HEADER_KEYWORDS = {
    "simbol", "simbolo", "symbol", "posiciones", "posicion",
    "vectores", "red", "orbitales"
}

SYMBOL_REGEX = re.compile(r"^[A-Za-z][a-z]?$")  # ej: C, Fe (acepta 1-2 letras)

# ----------------------------
# Utilidades
# ----------------------------
def _is_comment(line: str) -> bool:
    return line.strip().startswith("#")

def _is_header_line(line: str) -> bool:
    """
    Detecta líneas de encabezado incluso si no empiezan por '#'.
    Se considera header si ANY token en minúsculas está en HEADER_KEYWORDS.
    """
    tokens = re.split(r"\s+|,", line.strip())
    tokens = [t.lower() for t in tokens if t]
    return any(tok in HEADER_KEYWORDS for tok in tokens)

def _clean_line(line: str) -> str:
    """Quita fin de línea y comas residuales, pero no quita contenido válido."""
    return line.strip().replace(",", " ")


def read_positions_strict(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Lee Posiciones.txt estrictamente.
    Devuelve (symbols_list, positions_array(N,3))
    """
    symbols = []
    positions = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if _is_comment(raw):
                continue
            if _is_header_line(raw):
                continue
            line = _clean_line(raw)
            if not line:
                continue
            toks = line.split()
            if len(toks) != 4:
                raise ValueError(f"[Posiciones.txt] Línea inválida (se esperan 4 columnas): '{raw.rstrip()}'")
            sym = toks[0]
            if not SYMBOL_REGEX.match(sym):
                raise ValueError(f"[Posiciones.txt] Símbolo atómico inválido: '{sym}' en línea: '{raw.rstrip()}'")
            try:
                x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
            except ValueError:
                raise ValueError(f"[Posiciones.txt] Coordenadas no numéricas en: '{raw.rstrip()}'")
            symbols.append(sym)
            positions.append([x, y, z])
    if len(symbols) == 0:
        raise ValueError("[Posiciones.txt] No se han leído posiciones válidas.")
    return symbols, np.array(positions, dtype=float)

def read_orbitals_strict(path: str) -> List[Dict[str, Any]]:
    """
    Lee Orbitales.txt estrictamente.
    Cada línea de datos: SYMBOL  orb1 [orb2 ...]
    Devuelve lista de dicts: {'symbol':..., 'orbitals':[...] }
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if _is_comment(raw):
                continue
            if _is_header_line(raw):
                continue
            line = _clean_line(raw)
            if not line:
                continue
            toks = line.split()
            if len(toks) < 2:
                raise ValueError(f"[Orbitales.txt] Línea inválida (se esperan >=2 columnas): '{raw.rstrip()}'")
            sym = toks[0]
            if not SYMBOL_REGEX.match(sym):
                raise ValueError(f"[Orbitales.txt] Símbolo atómico inválido: '{sym}' en línea: '{raw.rstrip()}'")
            orbitals = toks[1:]
            # Validar que cada orbital esté en la lista permitida
            for orb in orbitals:
                if orb not in ALLOWED_ORBITALS:
                    raise ValueError(f"[Orbitales.txt] Orbital desconocido '{orb}' en línea: '{raw.rstrip()}'. "
                                     f"Permitidos: {sorted(ALLOWED_ORBITALS)}")
            out.append({"symbol": sym, "orbitals": orbitals})
    if len(out) == 0:
        raise ValueError("[Orbitales.txt] No se han leído líneas válidas.")
    return out

symbols, positions = read_positions_strict("Posiciones.txt")  # list, (N,3) numpy
orbitals = read_orbitals_strict("orbitales.txt")  # list of dicts

#convertimos orbitales para mejor uso
#orbitals = {entry['symbol']: entry['orbitals'] for entry in orbitals}
#N = 0
#for i in range(len(symbols)):
#    N += len(orbitals[symbols[i]])
#Definimos el Hamiltoniano
N = None

# Si quieres forzar N seleccionado, pon un entero par >0; si None -> toma máximo simétrico posible
SELECT_NBANDS = N  # ejemplo: 16  (dejar None para auto)

FLOAT_RE = re.compile(r'[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?')

def parse_bandas_and_header(path):
    """
    Devuelve (bands:list of lists (energies), header:dict con 'Fermi' si existe)
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    bands = []
    current_band = None
    header = {}

    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.rstrip("\n")
            # extraer Fermi si aparece
            mF = re.search(r'Fermi level\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)', line, flags=re.IGNORECASE)
            if mF:
                try:
                    header['Fermi'] = float(mF.group(1))
                except:
                    pass

            # detectar inicio de bloque "# Band n"
            m = re.match(r'^\s*#\s*Band\s*(\d+)\s*', line, flags=re.IGNORECASE)
            if m:
                if current_band is not None:
                    bands.append(current_band)
                current_band = []
                continue

            if current_band is not None:
                nums = FLOAT_RE.findall(line)
                if len(nums) >= 2:
                    # segundo número es energía
                    try:
                        energy = float(nums[1])
                        current_band.append(energy)
                    except ValueError:
                        pass
                # si no hay dos números se ignora la línea
                continue

    if current_band is not None:
        bands.append(current_band)

    return bands, header

def bands_to_array(bands):
    if not bands:
        raise RuntimeError("No se han detectado bandas en el fichero.")
    lens = [len(b) for b in bands]
    if len(set(lens)) != 1:
        raise ValueError(f"Inconsistencia: número de puntos k por banda distinto: {lens}")
    Nk = lens[0]; Nbands = len(bands)
    arr = np.zeros((Nk, Nbands), dtype=float)
    for ib, b in enumerate(bands):
        arr[:, ib] = b
    return arr

def find_vb_cb(energies, EF):
    """
    energies: array (Nk, Nbands)
    EF: float
    devuelve (vb_idx, cb_idx, max_below, min_above) arrays/scalars
    """
    Nk, Nbands = energies.shape
    max_below = np.full(Nbands, -np.inf)
    min_above = np.full(Nbands, np.inf)

    diff = energies - EF  # (Nk, Nbands)

    for b in range(Nbands):
        col = diff[:, b]
        below = col[col <= 0.0]
        above = col[col >= 0.0]
        if below.size > 0:
            max_below[b] = np.max(below)
        if above.size > 0:
            min_above[b] = np.min(above)

    # valence band: largest max_below
    vb = int(np.argmax(max_below))
    # conduction band: smallest min_above
    cb = int(np.argmin(min_above))

    return vb, cb, max_below, min_above

def select_symmetric_bands_around_gap(Nbands, vb, cb, requested=None):
    """
    Decide índices de bandas a seleccionar (lista ordenada).
    - requested: if int >0, try to use that (will be adjusted to allowed max and parity).
    - otherwise choose maximum symmetric: nb = 2*min(vb+1, Nbands-cb)
    Returns list of band indices (ascending) and nb_select.
    """
    n_lower_avail = vb + 1
    n_upper_avail = Nbands - cb
    max_sym = 2 * min(n_lower_avail, n_upper_avail)
    if max_sym == 0:
        # fallback: select all bands
        return list(range(Nbands)), max_sym

    if requested is None:
        nb_select = max_sym
    else:
        nb_select = int(requested)
        # cannot exceed max_sym
        if nb_select > max_sym:
            nb_select = max_sym
        # prefer even (split equally). If odd, allow lower=floor(nb/2), upper=nb - lower
        if nb_select <= 0:
            nb_select = max_sym

    lower_count = nb_select // 2
    upper_count = nb_select - lower_count  # if nb_select even -> equal halves

    start_lower = vb - lower_count + 1
    end_lower = vb  # inclusive
    start_upper = cb
    end_upper = cb + upper_count - 1

    # safety clamps
    start_lower = max(0, start_lower)
    end_upper = min(Nbands - 1, end_upper)

    indices = list(range(start_lower, end_lower + 1)) + list(range(start_upper, end_upper + 1))
    return indices, nb_select

def write_eigenvalues_file(arr, path, header_map: Dict[str, Any]=None, original_band_indices: List[int]=None):
    """
    Escribe arr (Nk, Nb) en path y añade líneas de cabecera explicativas.
    header_map puede contener 'Fermi', 'vb', 'cb' (índices originales), etc.
    original_band_indices: lista de índices originales de bandas correspondiente a cada columna de arr
    """
    Nk, Nb = arr.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# eigenvalues: filas = k-point index, columnas = bandas (N={Nb})\n")
        f.write(f"# columnas (0-based): 0..{Nb-1}\n")
        # Header map básico
        if header_map is None:
            header_map = {}
        if 'Fermi' in header_map:
            f.write(f"# Fermi = {header_map['Fermi']:.12e}\n")
        else:
            f.write(f"# Fermi = (no detectado)\n")

        # si se proporcionan índices originales, escribir el mapeo
        if original_band_indices is not None:
            f.write("# Mapeo: original_band_index -> column_in_file (0-based), column_in_file(1-based)\n")
            for new_col, orig_idx in enumerate(original_band_indices):
                f.write(f"#   original={orig_idx} -> col0={new_col} , col1={new_col+1}\n")

            # indicar explícitamente vb/cb si están en ese listado
            vb = header_map.get('vb', None)
            cb = header_map.get('cb', None)
            if vb is not None:
                if vb in original_band_indices:
                    newcol_vb = original_band_indices.index(vb)
                    f.write(f"# Valence band: original_index={vb} -> appears_as_column0={newcol_vb}, column1={newcol_vb+1}\n")
                else:
                    f.write(f"# Valence band: original_index={vb} -> NOT included in this selection\n")
            if cb is not None:
                if cb in original_band_indices:
                    newcol_cb = original_band_indices.index(cb)
                    f.write(f"# Conduction band: original_index={cb} -> appears_as_column0={newcol_cb}, column1={newcol_cb+1}\n")
                else:
                    f.write(f"# Conduction band: original_index={cb} -> NOT included in this selection\n")

        # separación entre cabecera y datos
        f.write("# --- datos (valores de energía) ---\n")
        for ik in range(Nk):
            row = " ".join(f"{val:.12e}" for val in arr[ik, :])
            f.write(row + "\n")

def main():
    print("Leyendo", INPUT)
    bands, header = parse_bandas_and_header(INPUT)
    if not bands:
        print("No se han encontrado bandas en el fichero.")
        sys.exit(1)

    arr = bands_to_array(bands)   # (Nk, Nbands)
    Nk, Nbands = arr.shape
    print("Detectadas bandas:", Nbands, "puntos k por banda:", Nk)

    EF = header.get('Fermi', None)
    if EF is None:
        print("AVISO: no se detectó 'Fermi' en la cabecera. Se usará EF = 0.0 por defecto.")
        EF = 0.0
    print("Fermi level =", EF)

    # determinar vb y cb
    vb, cb, max_below, min_above = find_vb_cb(arr, EF)
    print(f"Valence band index vb = {vb}, max_below[vb] = {max_below[vb]:.6e}")
    print(f"Conduction band index cb = {cb}, min_above[cb] = {min_above[cb]:.6e}")

    # seleccionar simétricamente
    indices, nb_select = select_symmetric_bands_around_gap(Nbands, vb, cb, requested=SELECT_NBANDS)
    print(f"Se seleccionarán {nb_select} bandas (índices originales): {indices}")

    arr_sel = arr[:, indices]

    # preparar información de cabecera para el archivo de salida seleccionado
    header_map = {
        'Fermi': EF,
        'vb': vb,
        'cb': cb,
        'Nbands_total': Nbands,
        'selected_count': nb_select
    }

    write_eigenvalues_file(arr_sel, OUT_SEL, header_map=header_map, original_band_indices=indices)
    print("Escrito selección en", OUT_SEL, "con shape", arr_sel.shape)

if __name__ == "__main__":
    main()
