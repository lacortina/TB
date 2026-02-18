#!/usr/bin/env python3
#!/usr/bin/env python3
# read_strict.py
"""
Lectura estricta de:
 - Red.txt       : 3 líneas con 3 floats (vectores de red)
 - Posiciones.txt: líneas "SYMBOL  x  y  z" (sin header de datos)
 - Orbitales.txt : líneas "SYMBOL  orb1 orb2 ..." (orbitales entre el conjunto permitido)

Se ignoran líneas que comiencen por '#' o líneas de encabezado que contengan
palabras como 'simbol', 'simbolo', 'posiciones', 'vectores', 'orbitales'.
Si el formato no coincide se lanza ValueError con mensaje explicativo.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import re

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

# ----------------------------
# Lectores estrictos
# ----------------------------
def read_cell_strict(path: str) -> np.ndarray:
    """Lee Red.txt: devuelve (3,3) numpy array. Estricto."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if _is_comment(raw):
                continue
            if _is_header_line(raw):
                # salto explícito de encabezado si existe
                continue
            line = _clean_line(raw)
            if not line:
                continue
            toks = line.split()
            if len(toks) != 3:
                raise ValueError(f"[Red.txt] Línea con {len(toks)} columnas (se esperaban 3): '{raw.rstrip()}'")
            try:
                vals = [float(toks[i]) for i in range(3)]
            except ValueError:
                raise ValueError(f"[Red.txt] Valores no numéricos en: '{raw.rstrip()}'")
            rows.append(vals)
    if len(rows) != 3:
        raise ValueError(f"[Red.txt] Debe contener exactamente 3 líneas de vectores. Encontradas: {len(rows)}")
    return np.array(rows, dtype=float)

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

ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
def parse_param_file(path: str):
    """
    Lee el fichero de parámetros SK y devuelve:

    m, tol, Delta, Onsite, SK

    donde:
      - Onsite[base_name][(X,)] = valor
      - SK[base_name][(X,Y)] = valor
    """

    m = None
    tol = None
    Delta = None

    Onsite: Dict[str, Dict[Tuple[str], float]] = {}
    SK: Dict[str, Dict[Tuple[str, str], float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"[{path}:{lineno}] Formato inválido (se esperan 2 columnas): '{raw.rstrip()}'"
                )

            name, value_str = parts
            try:
                value = float(value_str)
            except ValueError:
                raise ValueError(
                    f"[{path}:{lineno}] Valor no numérico: '{value_str}'"
                )

            # ------------------
            # Variables globales
            # ------------------
            if name == "m":
                m = value
                continue
            if name == "tol":
                tol = value
                continue
            if name == "Delta":
                Delta = value
                continue

            # ------------------
            # Parámetros con especies
            # ------------------
            tokens = name.split("_")
            if len(tokens) < 2:
                raise ValueError(
                    f"[{path}:{lineno}] Nombre de parámetro inválido: '{name}'"
                )

            # detectar especies finales
            species = []
            while tokens and ELEMENT_RE.match(tokens[-1]):
                species.insert(0, tokens.pop())

            if not species:
                raise ValueError(
                    f"[{path}:{lineno}] No se detectaron especies en '{name}'"
                )

            base_name = "_".join(tokens)

            # ------------------
            # On-site (1 átomo)
            # ------------------
            if len(species) == 1:
                key = (species[0],)
                if base_name not in Onsite:
                    Onsite[base_name] = {}
                if key in Onsite[base_name]:
                    raise ValueError(
                        f"[{path}:{lineno}] Duplicado Onsite {base_name}_{species[0]}"
                    )
                Onsite[base_name][key] = value

            # ------------------
            # Slater–Koster (2 átomos)
            # ------------------
            elif len(species) == 2:
                key = (species[0], species[1])
                if base_name not in SK:
                    SK[base_name] = {}
                if key in SK[base_name]:
                    raise ValueError(
                        f"[{path}:{lineno}] Duplicado SK {base_name}_{species[0]}_{species[1]}"
                    )
                SK[base_name][key] = value

            else:
                raise ValueError(
                    f"[{path}:{lineno}] Demasiadas especies en '{name}'"
                )

    # ------------------
    # Comprobaciones finales
    # ------------------
    missing = []
    if m is None:
        missing.append("m")
    if tol is None:
        missing.append("tol")
    if Delta is None:
        missing.append("Delta")
    if missing:
        raise ValueError(f"Faltan parámetros obligatorios: {missing}")

    return m, tol, Delta, Onsite, SK

def generate_R_list_including_zero(lattice, coeffs=(-1,0,1)):
    """Genera lista de vectores R y combinaciones (n1,n2,n3) incluyendo la imagen 0."""
    R_list = []
    combos = []
    # incluir R = (0,0,0) como primera entrada
    R_list.append(np.zeros(3, dtype=float))
    combos.append((0,0,0))
    for n1 in coeffs:
        for n2 in coeffs:
            for n3 in coeffs:
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                R = n1*lattice[0] + n2*lattice[1] + n3*lattice[2]
                # Evitar R que son (prácticamente) el vector nulo
                R_list.append(R)
                combos.append((n1,n2,n3))
    return R_list

def read_kpoints_frac_strict(filename):
    """
    Lee k-points fraccionarios desde fichero:
    3 columnas por línea, comentarios con '#'.
    Devuelve: numpy array shape (Nk, 3)
    """
    kpoints = []

    with open(filename, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = line.split()

            if len(tokens) != 3:
                raise ValueError(
                    f"{filename}:{lineno}: se esperan exactamente 3 columnas"
                )

            try:
                k = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
            except ValueError:
                raise ValueError(
                    f"{filename}:{lineno}: k-point no convertible a float"
                )

            kpoints.append(k)

    if not kpoints:
        raise RuntimeError(f"{filename}: no se leyeron k-points")

    return np.array(kpoints, dtype=float)

def reciprocal_lattice(cell):
    """
    cell: (3,3) array, filas = vectores de red directa a1,a2,a3
    devuelve: (3,3) array, filas = vectores recíprocos b1,b2,b3
    """
    a1, a2, a3 = cell

    V = np.dot(a1, np.cross(a2, a3))

    if abs(V) < 1e-12:
        raise RuntimeError("Volumen de celda nulo")

    b1 = 2*np.pi * np.cross(a2, a3) / V
    b2 = 2*np.pi * np.cross(a3, a1) / V
    b3 = 2*np.pi * np.cross(a1, a2) / V

    return np.vstack([b1, b2, b3])

## -----------------------------------------------
##FUNCION PRINCIPAL SLATER KOSTER
## -----------------------------------------------
#FUNCIONES DE AYUDA PARA SACAR LOS COEFICIENTES SLATERE-KOSTER
def funcionVpps(x, Vpp_sigma, Vpp_pi):
        b=x**2*Vpp_sigma+(1-x**2)*Vpp_pi
        return b
def funcionVppp(x, b, Vpp_sigma, Vpp_pi):
        c=x*b*(Vpp_sigma-Vpp_pi)
        return c


#funcion slater koster
def funcionslaterkoster(vector, orbital1, orbital2, Vss, Vsp, Vsds, Vpp_sigma, Vpp_pi, Vpds, Vpdp, Vdds, Vddp, Vddd):
    #cosenos directores
    r=np.linalg.norm(vector)
    l=vector[0]/r
    m=vector[1]/r
    n=vector[2]/r
    if orbital1 == 's':
        if orbital2 == 's':
            return Vss
        elif orbital2 == 'px':
            return Vsp*l
        elif orbital2 == 'py':
            return Vsp*m
        elif orbital2 == 'pz':
            return Vsp*n
#añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l * m * Vsds
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l * n * Vsds
        elif orbital2 == 'dyz':
            return np.sqrt(3) * n * m * Vsds
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * (l**2 - m**2) * Vsds
        elif orbital2 == 'dr':
            return (n**2 - (l**2 + m**2) / 2 )* Vsds
    
    elif orbital1 == 'px':
        if orbital2 == 's':
            return Vsp*(-l)
        elif orbital2 == 'px':
            return funcionVpps(l, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVppp(l, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVppp(l, n, Vpp_sigma, Vpp_pi)
        #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l**2 * m * Vpds + m * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l**2 * n * Vpds + n * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'dyz':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * l * (l**2 - m**2) * Vpds + l * (1 - l**2 + m**2) * Vpdp
        elif orbital2 == 'dr':
            return l * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * l * n**2 * Vpdp

    elif orbital1 == 'py':
        if orbital2 == 's':
            return Vsp*(-m)
        elif orbital2 == 'px':
            return funcionVppp(l, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVpps(m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVppp(m, n, Vpp_sigma, Vpp_pi)
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * m**2 * l * Vpds + l * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dyz':
            return np.sqrt(3) * m**2 * n * Vpds + n * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * m * (l**2 - m**2) * Vpds - m * (1 + l**2 - m**2) * Vpdp
        elif orbital2 == 'dr':
            return m * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * m * n**2 * Vpdp

    elif orbital1 == 'pz':
        if orbital2 == 's':
            return Vsp*(-n)
        elif orbital2 == 'px':
            return funcionVppp(n, l, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'py':
            return funcionVppp(n, m, Vpp_sigma, Vpp_pi)
        elif orbital2 == 'pz':
            return funcionVpps(n, Vpp_sigma, Vpp_pi)
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l * n*m * Vpds -2* n * m*l * Vpdp
        elif orbital2 == 'dxz':
            return np.sqrt(3) * n**2 * l * Vpds + l * (1 - 2 * n**2) * Vpdp           
        elif orbital2 == 'dyz':
            return np.sqrt(3) * n**2 * m * Vpds + m * (1 - 2 * n**2) * Vpdp
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) / 2 * n * (l**2 - m**2) * Vpds - n * ( l**2 - m**2) * Vpdp
        elif orbital2 == 'dr':
            return n * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * n * (l**2+m**2) * Vpdp

    elif orbital1 == 'dxy':
        if orbital2 == 's':
            return np.sqrt(3) * l * m * Vsds
        elif orbital2 == 'px':
            return np.sqrt(3) * l**2 * (-m) * Vpds + (-m) * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'py':
            return np.sqrt(3) * m**2 * (-l) * Vpds + (-l) * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'pz':
            return np.sqrt(3) * (-l) * n*m * Vpds +2* n * m*l * Vpdp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l**2 * m**2 * Vdds + (l**2 + m**2 - 4 * l**2 * m**2) * Vddp+(n**2+l**2*m**2)*Vddd
        elif orbital2 == 'dxz':
            return 3 * l**2 * m * n * Vdds + m * n * (l**2 - 1) * Vddd  + m*n*(1-4*l**2)*Vddp       
        elif orbital2 == 'dyz':
            return 3 * l * n * m**2 * Vdds + l * n * (1 - 4 * m**2) * Vddp+ l*n*(m**2-1)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * l * m * (l**2 - m**2) * Vdds + 2 * l * m * (m**2 - l**2) * Vddp + (l * m * (l**2 - m**2) / 2) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * l * m * (n**2 - (l**2 + m**2) / 2) * Vdds - 2 * l * m*n**2 * Vddp + (l * m * (1 + n**2) / 2) * Vddd

    elif orbital1 == 'dxz':
        if orbital2 == 's':
            return np.sqrt(3) * l * n * Vsds
        elif orbital2 == 'px':
            return np.sqrt(3) * l**2 * (-n) * Vpds + (-n) * (1 - 2 * l**2) * Vpdp
        elif orbital2 == 'py':
            return np.sqrt(3) * (-l) * n*m * Vpds +2* n * m*l * Vpdp
        elif orbital2 == 'pz':
            return np.sqrt(3) * n**2 * (-l) * Vpds + (-l) * (1 - 2 * n**2) * Vpdp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l**2 * m * n * Vdds + m * n * (l**2 - 1) * Vddd  + m*n*(1-4*l**2)*Vddp
        elif orbital2 == 'dxz':
            return 3 * l**2 * n**2 * Vdds + (l**2 + n**2 - 4 * l**2 * n**2) * Vddp+(m**2+l**2*n**2)*Vddd      
        elif orbital2 == 'dyz':
            return 3 * l * m * n**2 * Vdds + l * m * (1 - 4 * n**2) * Vddp+ l*m*(n**2-1)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * n * l * (l**2 - m**2) * Vdds + n * l * (1 - 2 * (l**2 - m**2)) * Vddp - (n * l *(1- (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * n * l * (n**2 - (l**2 + m**2) / 2) * Vdds + l * n * (l**2 + m**2 - n**2) * Vddp - (l * n * (l**2 + m**2) / 2) * Vddd
   
    elif orbital1 == 'dyz':
        if orbital2 == 's':
            return np.sqrt(3) * n * m * Vsds
        elif orbital2 == 'px':
            return np.sqrt(3) * (-l) * n*m * Vpds +2* n * m*l * Vpdp
        elif orbital2 == 'py':
            return np.sqrt(3) * m**2 *(-n) * Vpds + (-n) * (1 - 2 * m**2) * Vpdp
        elif orbital2 == 'pz':
            return np.sqrt(3) * n**2 *(-m) * Vpds + (-m) * (1 - 2 * n**2) * Vpdp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return 3 * l * n * m**2 * Vdds + l * n * (1 - 4 * m**2) * Vddp+ l*n*(m**2-1)*Vddd
        elif orbital2 == 'dxz':
            return 3 * l * m * n**2 * Vdds + l * m * (1 - 4 * n**2) * Vddp+ l*m*(n**2-1)*Vddd    
        elif orbital2 == 'dyz':
            return 3 * m**2 * n**2 * Vdds + (n**2 + m**2 - 4 * n**2 * m**2) * Vddp+(l**2+n**2*m**2)*Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 2) * n * m * (l**2 - m**2) * Vdds - n * m * (1 + 2 * (l**2 - m**2)) * Vddp + (n * m *(1+ (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * n * m * (n**2 - (l**2 + m**2) / 2) * Vdds + m * n * (l**2 + m**2 - n**2) * Vddp - (n * m * (l**2 + m**2) / 2) * Vddd

    elif orbital1 == 'dx2_y2':
        if orbital2 == 's':
            return np.sqrt(3) / 2 * (l**2 - m**2) * Vsds
        elif orbital2 == 'px':
            return np.sqrt(3) / 2 * (-l) * (l**2 - m**2) * Vpds + (-l) * (1 - l**2 + m**2) * Vpdp
        elif orbital2 == 'py':
            return np.sqrt(3) / 2 * (-m) * (l**2 - m**2) * Vpds - (-m) * (1 + l**2 - m**2) * Vpdp
        elif orbital2 == 'pz':
            return np.sqrt(3) / 2 * (-n) * (l**2 - m**2) * Vpds - (-n) * ( l**2 - m**2) * Vpdp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return (3 / 2) * l * m * (l**2 - m**2) * Vdds + 2 * l * m * (m**2 - l**2) * Vddp + (l * m * (l**2 - m**2) / 2) * Vddd
        elif orbital2 == 'dxz':
            return (3 / 2) * n * l * (l**2 - m**2) * Vdds + n * l * (1 - 2 * (l**2 - m**2)) * Vddp - (n * l *(1- (l**2 - m**2) / 2)) * Vddd   
        elif orbital2 == 'dyz':
            return (3 / 2) * n * m * (l**2 - m**2) * Vdds - n * m * (1 + 2 * (l**2 - m**2)) * Vddp + (n * m *(1+ (l**2 - m**2) / 2)) * Vddd
        elif orbital2 == 'dx2_y2':
            return (3 / 4) * (l**2 - m**2)**2 * Vdds + (l**2 + m**2 - (l**2 - m**2)**2) * Vddp + (n**2+(l**2 - m**2)**2 / 4) * Vddd
        elif orbital2 == 'dr':
            return np.sqrt(3) * (l**2 - m**2) *( n**2 - (l**2 + m**2) / 2) * Vdds/2.0 + (n**2 * (m**2 - l**2)) * Vddp + ((1 + n**2) * (l**2 - m**2) / 4) * Vddd

    elif orbital1 == 'dr':
        if orbital2 == 's':
            return (n**2 - (l**2 + m**2) / 2) * Vsds
        elif orbital2 == 'px':
            return (-l) * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * (-l) * n**2 * Vpdp
        elif orbital2 == 'py':
            return (-m) * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * (-m) * n**2 * Vpdp
        elif orbital2 == 'pz':
            return (-n) * (n**2 - (l**2 + m**2) / 2) * Vpds - np.sqrt(3) * (-n) * (l**2+m**2) * Vpdp
         #añadimos los d (dxy, dxz, dyz, dx2_y2, d3z2_r29)
        elif orbital2 == 'dxy':
            return np.sqrt(3) * l * m * (n**2 - (l**2 + m**2) / 2) * Vdds - 2 * l * m*n**2 * Vddp + (l * m * (1 + n**2) / 2) * Vddd
        elif orbital2 == 'dxz':
            return np.sqrt(3) * n * l * (n**2 - (l**2 + m**2) / 2) * Vdds + l * n * (l**2 + m**2 - n**2) * Vddp - (l * n * (l**2 + m**2) / 2) * Vddd
        elif orbital2 == 'dyz':
            return np.sqrt(3) * n * m * (n**2 - (l**2 + m**2) / 2) * Vdds + m * n * (l**2 + m**2 - n**2) * Vddp - (n * m * (l**2 + m**2) / 2) * Vddd
        elif orbital2 == 'dx2_y2':
            return np.sqrt(3) * (l**2 - m**2) *( n**2 - (l**2 + m**2) / 2) * Vdds/2.0 + (n**2 * (m**2 - l**2)) * Vddp + ((1 + n**2) * (l**2 - m**2) / 4) * Vddd
        elif orbital2 == 'dr':
            return (n**2 - (l**2 + m**2) / 2)**2 * Vdds + 3 * n**2 * (l**2 + m**2) * Vddp + (3 / 4) * (l**2 + m**2)**2 * Vddd

                             
    else:
        return 0

#funcion para asignar los valores
import warnings

def get_SK_param(SKdict, keyname, a, b):
    """Buscar SKdict[keyname][(a,b)] o SKdict[keyname][(b,a)], devolver 0.0 si falta."""
    try:
        return SKdict[keyname][(a, b)]
    except KeyError:
        try:
            return SKdict[keyname][(b, a)]
        except KeyError:
            # opcional: avisar una sola vez por combinación
            warnings.warn(f"SK['{keyname}'] missing for {(a,b)} and {(b,a)} -> using 0.0", stacklevel=3)
            return 0.0

def get_Sol_param(Soldict, keyname, a, b):
    """Análogo para el diccionario de solape."""
    try:
        return Soldict[keyname][(a, b)]
    except KeyError:
        try:
            return Soldict[keyname][(b, a)]
        except KeyError:
            warnings.warn(f"Sol['{keyname}'] missing for {(a,b)} and {(b,a)} -> using 0.0", stacklevel=3)
            return 0.0


# ----------------------------
# Ejemplo de uso
# ----------------------------
#leemos los archivos de entrada
cell = read_cell_strict("Red.txt")                 # (3,3) numpy
symbols, positions = read_positions_strict("Posiciones.txt")  # list, (N,3) numpy
orbitals = read_orbitals_strict("orbitales.txt")  # list of dicts

#convertimos orbitales para mejor uso
orbitals = {entry['symbol']: entry['orbitals'] for entry in orbitals}

m, tol, Delta ,Onshell, SK = parse_param_file("sk_params.txt")  # (m, tol, Delta, species_params, flat_params)

_, _, _ ,_, Sol = parse_param_file("solape.txt")  # (m, tol, Delta, species_params, flat_params)

#print(Sol)

#conjunto de vectores vecinos
Vecinos=generate_R_list_including_zero(cell)


N = 0
for i in range(len(symbols)):
    N += len(orbitals[symbols[i]])


#definos los autovectores y autovalores


#para acceder a los valores se tiene
#SK_final['C','C']['Vsp']
#Onsite['C']['Vddd']
#print(SK_final[symbols[0],symbols[0]]['Vddd'])

#los vectores reciprocos
kpoints_frac = read_kpoints_frac_strict("kpath_frac.txt")
B = reciprocal_lattice(cell)
numk = len(kpoints_frac)
#definimos los autovectores y autovaores
autovaloresA = np.zeros((numk,N), dtype = float)
autovaloresL = np.zeros((numk,N), dtype = float)
autovectoresA = np.zeros((numk,N,N), dtype = complex)
autovectoresL = np.zeros((numk,N,N), dtype = complex)

#abrimos los ficheros
f_eval = open("eigenvalues.txt", "w", encoding="utf-8")
f_evec = open("eigenvectors.txt", "w", encoding="utf-8")

# Cabeceras informativas
f_eval.write(f"# eigenvalues: filas = k-point index, columnas = bandas (N={N})\n")
f_eval.write(f"# columnas: 0..{N-1}\n")
f_evec.write(f"# eigenvectors: bloques por k; cada bloque comienza con '# k <ik>' seguido de N lineas (cada linea es un autovector con N entradas complejas)\n")
f_evec.write(f"# formato complejo por entrada: Re(+-)Imj (ej: 1.234567e-01+2.345678e-03j)\n")


print(orbitals)

for ik, kf in enumerate(kpoints_frac):
    k = kf[0]*B[0] + kf[1]*B[1] + kf[2]*B[2]
    #print(k)
    Ha = np.zeros((N,N), dtype=complex)
    HL = np.zeros((N,N), dtype=complex)
    SA = np.zeros((N,N), dtype=complex)
    SL = np.zeros((N,N), dtype=complex)

    for i in range(len(symbols)):
        sym1 = symbols[i]
        ri = positions[i]
        lenorb1=len(orbitals[sym1])
        #Definimos el Hamiltoniano
        
        for j in range(len(symbols)): #para recorrer solo la parte de arriba range(i,len(symbols))
            sym2 = symbols[j]
            rj = positions[j]
            lenorb2=len(orbitals[sym2])
            
            
            for z in range(len(Vecinos)):
                #Calculamos para cada vecino la distancia entre ellos
                R = rj - ri + Vecinos[z]
                dist = np.linalg.norm(R)
                #aqui se mete la tension 
                if abs(dist) > abs(Delta + tol):
                    continue
                #print(R)
                phase = np.exp(1j * np.dot(k, R)) #version atomic
                phaseL = np.exp(1j * np.dot(k, Vecinos[z])) #version Latiice
                for i_orb, orbital1 in enumerate(orbitals[sym1]):
                    for j_orb, orbital2 in enumerate(orbitals[sym2]):
                        if dist == 0 and orbital1 == orbital2:
                            #hacemos la diferencia entre s, p, d,
                            if orbital1 == 's':
                                Ha[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_s'][(sym1,)]
                                HL[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_s'][(sym1,)]
                            elif orbital1 == 'px' or orbital1 == 'py' or orbital1 == 'pz':
                                Ha[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_p'][(sym1,)]
                                HL[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_p'][(sym1,)]
                            elif orbital1 == 'dxy' or orbital1 == 'dxz' or orbital1 == 'dyz' or orbital1 == 'dx2_y2' or orbital1 == 'dr':
                                Ha[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_d'][(sym1,)]
                                HL[lenorb1*i+i_orb,lenorb1*i+i_orb] = Onshell['E_d'][(sym1,)]
                            #print(1, lenorb1*i+i_orb, lenorb2*j+j_orb)
                            SL[lenorb1*i+i_orb,lenorb1*i+i_orb] = 1
                            SA[lenorb1*i+i_orb,lenorb1*i+i_orb] = 1

                        else:
                            if dist == 0:
                                continue
                            Vss        = get_SK_param(SK,  'Vss',       sym1, sym2)
                            Vpp_sigma  = get_SK_param(SK,  'Vpp_sigma', sym1, sym2)
                            Vpp_pi     = get_SK_param(SK,  'Vpp_pi',    sym1, sym2)
                            Vdds       = get_SK_param(SK,  'Vdds',      sym1, sym2)
                            Vddp       = get_SK_param(SK,  'Vddp',      sym1, sym2)
                            Vddd       = get_SK_param(SK,  'Vddd',      sym1, sym2)
                            Vpds       = get_SK_param(SK,  'Vpds',      sym1, sym2)
                            Vpdp       = get_SK_param(SK,  'Vpdp',      sym1, sym2)
                            Vsp        = get_SK_param(SK,  'Vsp',       sym1, sym2)
                            Vsds       = get_SK_param(SK,  'Vsds',      sym1, sym2)

                            # solape
                            Sss        = get_Sol_param(Sol, 'Vss',       sym1, sym2)
                            Spp_sigma  = get_Sol_param(Sol, 'Vpp_sigma', sym1, sym2)
                            Spp_pi     = get_Sol_param(Sol, 'Vpp_pi',    sym1, sym2)
                            Sdds       = get_Sol_param(Sol, 'Vdds',      sym1, sym2)
                            Sddp       = get_Sol_param(Sol, 'Vddp',      sym1, sym2)
                            Sddd       = get_Sol_param(Sol, 'Vddd',      sym1, sym2)
                            Spds       = get_Sol_param(Sol, 'Vpds',      sym1, sym2)
                            Spdp       = get_Sol_param(Sol, 'Vpdp',      sym1, sym2)
                            Ssp        = get_Sol_param(Sol, 'Vsp',       sym1, sym2)
                            Ssds       = get_Sol_param(Sol, 'Vsds',      sym1, sym2)


                            Ha[lenorb1*i+i_orb,lenorb2*j+j_orb] += phase*funcionslaterkoster(R,orbital1, orbital2, Vss, Vsp, Vsds, Vpp_sigma ,Vpp_pi ,Vpds , Vpdp ,Vdds ,Vddp ,Vddd)
                            HL[lenorb1*i+i_orb,lenorb2*j+j_orb] += phaseL*funcionslaterkoster(R,orbital1, orbital2,Vss, Vsp, Vsds, Vpp_sigma ,Vpp_pi ,Vpds , Vpdp ,Vdds ,Vddp ,Vddd)
                            SA[lenorb1*i+i_orb,lenorb2*j+j_orb] += phase* funcionslaterkoster(R,orbital1, orbital2, Vss, Vsp, Vsds, Spp_sigma ,Spp_pi ,Spds , Spdp ,Sdds ,Sddp ,Sddd)
                            SL[lenorb1*i+i_orb,lenorb2*j+j_orb] += phaseL* funcionslaterkoster(R,orbital1, orbital2, Vss, Vsp, Vsds, Spp_sigma ,Spp_pi ,Spds , Spdp ,Sdds ,Sddp ,Sddd)  #Ha[lenorb2*j+j_orb,lenorb1*i+i_orb] += 
                             

    eigenvaluesL, eigenvectorsL = np.linalg.eigh(HL)
    eigenvaluesA, eigenvectorsA = np.linalg.eigh(Ha)
    #almacenadmos la informacion dentro de las bandas
    autovaloresA[ik] = eigenvaluesA
    autovaloresL[ik] = eigenvaluesL
    autovectoresA[ik] = eigenvectorsA
    autovectoresL[ik] = eigenvectorsL                
    

    # Formateamos una fila: valores separados por espacios
    row = " ".join(f"{val: .12e}" for val in eigenvaluesL)
    f_eval.write(row + "\n")

    # -------------------------
    # Guardar autovectores: bloque NxN para este k
    # -------------------------
    # Cabecera del bloque
    f_evec.write(f"# k {ik}\n")
    # Escribimos N líneas, cada línea corresponde a un autovector v_n (orden: n=0..N-1)
    # cada línea contiene N entradas complejas formateadas como "Re+Imj"
    for n in range(N):
        vec = eigenvectorsA[:, n]  # vector columna de dimensión N
        line = " ".join(f"{v.real:.12e}{v.imag:+.12e}j" for v in vec)
        f_evec.write(line + "\n")
    #print(R)
# Cerrar ficheros al final
f_eval.close()
f_evec.close()

#print(cell)
#print(HL)
#print(k)
#print(eigenvaluesA)
#print(eigenvaluesL)

        



