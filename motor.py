
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import random


# ---------------------------
# Conversión de unidades
# ---------------------------
NM_TO_KM = 1.852
KT_TO_KMPH = 1.852

# ---------------------------
# Parámetros globales del modelo
# ---------------------------
OPERATION_MINUTES = 18 * 60  # 6:00 a 24:00
RUNWAY_SEP_MIN = 4           # separación mínima en minutos
RUNWAY_TARGET_GAP = 5        # buffer de distancia temporal en minutos
REJOIN_REQUIRED_GAP = 10     # gap requerido para reinsertarse tras go-around
GO_AROUND_SPEED_KT = 200     # velocidad durante alejamiento
MIN_REJOIN_DISTANCE_NM = 5   # para reinsertarse, estar > 5 mn de AEP
MAX_GO_AROUND_TIME_MIN = 25  # tiempo máximo tolerado en go-around antes de desviar
DIVERT_DISTANCE_NM = 110     # si supera esta distancia en nm, se desvía
CLOSURE_FORCE_GO_AROUND_DISTANCE_NM = 10  # durante cierre, forzar go-around dentro de este radio

# Bandas de distancia y velocidades (min, max) en nudos
# Usamos los máximos como "velocidad deseada" en camino despejado.
DIST_BANDS = [
    (101, np.inf, 300, 500),  # >100 mn
    (50, 100, 250, 300),
    (15, 50, 200, 250),
    (5, 15, 150, 200),
    (0, 5, 120, 150),
]

def speed_limits_for_distance_nm(d_nm: float) -> Tuple[int, int]:
    for lo, hi, vmin, vmax in DIST_BANDS:
        if lo < d_nm <= hi:
            return vmin, vmax
    # Si está más allá de 100 mn (fuera del radar), usamos el primer rango como sustituto.
    return DIST_BANDS[0][2], DIST_BANDS[0][3]

def desired_speed_max_for_distance_nm(d_nm: float) -> int:
    vmin, vmax = speed_limits_for_distance_nm(d_nm)
    return vmax

@dataclass
class Aircraft:
    id: int
    # Estado cinemático
    distance_nm: float = 100.0   # distancia al umbral (comienza en 100 nm al aparecer)
    speed_kt: float = 500.0      # velocidad actual (se ajusta por banda)
    status: str = "approach"     # approach | go_around | diverted | landed
    # Métricas/registro
    spawn_minute: int = 0
    landing_minute: Optional[int] = None
    go_around_count: int = 0
    diverted: bool = False
    # Flag para interrupciones metereológicas (viento)
    needs_interruption: bool = False
    # Flag para rejoin simple
    rejoining: bool = False
    # Tracking adicional para control de desvíos y reintentos
    go_around_start_minute: Optional[int] = None
    go_around_attempts: int = 0
    # Tiempos continuos (para medir delays no cuantizados)
    landing_time_min: Optional[float] = None           # tiempo fraccional real de aterrizaje
    cross_frac_in_minute: Optional[float] = None       # fracción [0,1] del minuto en que cruzó 0 nm

    def eta_minutes(self) -> float:
        """ETA a la pista mantener speed_kt constante. (Aproximado)"""
        if self.speed_kt <= 0:
            return np.inf
        hours = self.distance_nm / self.speed_kt
        return hours * 60.0 #convertimos a minutos

@dataclass
class SimulationConfig:
    lambda_per_min: float         # probabilidad de arribo por minuto 
    windy_day: bool = False             # si es True, cada avión tiene 0.1 de interrupción (go-around)
    closure_window: Optional[Tuple[int,int]] = None  # (t_start, t_end) minutos donde no se puede aterrizar
    seed: Optional[int] = None
    duration_minutes: int = OPERATION_MINUTES
    # NUEVO: tracing
    trace_all: bool = False
    trace_ids: Optional[List[int]] = None

@dataclass
class SimulationResult:
    landed: int
    diverted: int
    avg_delay_minutes: float
    delays: List[float]
    go_arounds: int
    timeline_landings: List[int]  # minutos de aterrizaje
    aircraft_log: List[Aircraft]  # estado final de cada avión
    congestion_time: int = 0
    # NUEVO: trazas por avión
    traces: Dict[int, List[Dict]] = field(default_factory=dict)

class AEPSimulator:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

        self.aircrafts: List[Aircraft] = []
        self.next_id = 1
        self.last_landing_minute: Optional[int] = None
        self.timeline_landings: List[int] = []
        self.diverted_count = 0
        self.go_around_events = 0
        self.congestion_time = 0

        # NUEVO: estado de tracing
        self._trace_all = bool(getattr(config, "trace_all", False))
        self._trace_ids = set(getattr(config, "trace_ids", []) or [])
        self._traces: Dict[int, List[Dict]] = {}
    # ---------------------------
    # Proceso de arribos (Bernoulli por minuto)
    # ---------------------------
    def maybe_spawn_aircraft(self, minute: int):
        if np.random.rand() < self.cfg.lambda_per_min:
            a = Aircraft(
                id=self.next_id,
                distance_nm=100.0,
                speed_kt=desired_speed_max_for_distance_nm(100.0),
                status="approach",
                spawn_minute=minute,
                needs_interruption=(np.random.rand()<0.1 if self.cfg.windy_day else False)
            )
            self.aircrafts.append(a)
            self.next_id += 1

    # ---------------------------
    # Lógica de separación y control de velocidades
    # ---------------------------
    def enforce_separation(self, minute: int):
        """Control por gap temporal con el líder inmediato."""
        approaching = [a for a in self.aircrafts if a.status == "approach"]
        approaching.sort(key=lambda x: x.distance_nm)

        for i, foll in enumerate(approaching):
            vmin, vmax = speed_limits_for_distance_nm(foll.distance_nm)
            desired = desired_speed_max_for_distance_nm(foll.distance_nm)

            if i == 0:
                # El más cercano va a su velocidad deseada (máxima por banda)
                foll.speed_kt = desired
                continue

            lead = approaching[i-1]
            gap = foll.eta_minutes() - lead.eta_minutes()  # en minutos

            if gap >= RUNWAY_SEP_MIN:
                # Con 4 min o más puede ir a máxima de su banda
                foll.speed_kt = desired
                continue

            # Gap < 4 min: bajar 20 kt respecto a SU velocidad actual (no la del líder)
            target = foll.speed_kt - 20.0
            target = max(target, vmin)        # no bajar del mínimo permitido por banda
            target = min(target, desired)     # no superar la máxima deseada por banda
            foll.speed_kt = target

            # Si con vmin no alcanzamos el buffer de 5 min, hace go-around
            new_gap = foll.eta_minutes() - lead.eta_minutes()
            if new_gap < RUNWAY_TARGET_GAP and foll.speed_kt <= vmin + 1e-6 and foll.status != "go_around":
                foll.status = "go_around"
                foll.speed_kt = GO_AROUND_SPEED_KT
                if getattr(foll, "go_around_start_minute", None) is None:
                    foll.go_around_start_minute = minute
                foll.go_around_attempts += 1
                self.go_around_events += 1

    # ---------------------------
    # Go-around simple y reinserción
    # ---------------------------
    def handle_go_around(self, minute: int):
        """
        Política simple: cuando un avión entra en go-around, se aleja (distance aumenta).
        Nueva política de reinserción:
          - Comparar ETA del que quiere reinsertarse contra TODA la cola que está llegando.
          - Requerimiento: hueco de al menos REJOIN_REQUIRED_GAP tanto con el anterior como el siguiente.
          - Reinsertar solo si la pista está abierta y la distancia es > MIN_REJOIN_DISTANCE_NM.
        Desvío:
          - Si supera DIVERT_DISTANCE_NM o sobrepasa MAX_GO_AROUND_TIME_MIN o excede intentos, se desvía.
        """
        # Precalcular ETAs de la cola actual en approach
        approaching = [b for b in self.aircrafts if b.status == "approach"]
        etas_approach = []
        for b in approaching:
            v_des_b = desired_speed_max_for_distance_nm(b.distance_nm)
            eta_b = minute + (b.distance_nm / max(1e-6, v_des_b) * 60.0)
            etas_approach.append((eta_b, b.id))
        etas_approach.sort(key=lambda x: x[0])

        for a in self.aircrafts:
            if a.status != "go_around":
                continue

            # Criterios de desvío por distancia/tiempo/intentos
            too_far = a.distance_nm > DIVERT_DISTANCE_NM
            too_long = (a.go_around_start_minute is not None) and (minute - a.go_around_start_minute > MAX_GO_AROUND_TIME_MIN)
            too_many = a.go_around_attempts >= 2
            if too_far or too_long or too_many:
                a.status = "diverted"
                a.diverted = True
                self.diverted_count += 1
                continue

            # Condiciones de reinserción segura
            v_des = desired_speed_max_for_distance_nm(a.distance_nm)
            eta_a = minute + (a.distance_nm / max(1e-6, v_des) * 60.0)
            far_enough = a.distance_nm >= MIN_REJOIN_DISTANCE_NM

            ok_gap = True
            
            # Si la pista está cerrada, bloquear reinserción
            if self.runway_closed(minute):
                ok_gap = False
            else:
                if etas_approach:
                    idx = 0
                    while idx < len(etas_approach) and etas_approach[idx][0] < eta_a:
                        idx += 1
                    # vecino anterior
                    if idx - 1 >= 0:
                        prev_eta = etas_approach[idx-1][0]
                        if eta_a - prev_eta < REJOIN_REQUIRED_GAP:
                            ok_gap = False
                    # vecino siguiente
                    if idx < len(etas_approach):
                        next_eta = etas_approach[idx][0]
                        if next_eta - eta_a < REJOIN_REQUIRED_GAP:
                            ok_gap = False

            if ok_gap and far_enough:
                a.status = "approach"  # reinsertado
                a.speed_kt = v_des

    # ---------------------------
    # Cierre del aeropuerto (no se puede aterrizar)
    # ---------------------------
    def runway_closed(self, minute: int) -> bool:
        if self.cfg.closure_window is None:
            return False
        start, end = self.cfg.closure_window
        return (start <= minute < end)

    # ---------------------------
    # Un paso de simulación (1 minuto)
    # ---------------------------
    def step(self, minute: int):
        # 1) Posibles nuevos arribos
        self.maybe_spawn_aircraft(minute)

        # 2) Aplicar reglas de separación a los que están en approach
        self.enforce_separation(minute)

        # 3) Mover cada avión según su estado
        for a in self.aircrafts:
            if a.status in ("approach", "go_around"):
                hours = 1.0/60.0
                delta_nm = a.speed_kt * hours
                if a.status == "approach":
                    remaining = 1.0  # minuto actual
                    # limpiar cruce previo; se recalcula cada minuto
                    a.cross_frac_in_minute = None
                    while remaining > 1e-9 and a.distance_nm > 0.0:
                        # velocidad según banda actual
                        vmin, vmax = speed_limits_for_distance_nm(a.distance_nm)
                        #v = min(a.speed_kt, vmax)
                        v = max(vmin, min(a.speed_kt, vmax))  # clamp explícito
                        nm_per_min = v / 60.0

                        # próximo borde de banda o 0 nm
                        if a.distance_nm > 50: 
                            next_edge = 50.0
                        elif a.distance_nm > 15: 
                            next_edge = 15.0
                        elif a.distance_nm > 5:  
                            next_edge = 5.0
                        else:                    
                            next_edge = 0.0

                        dist_to_edge = max(0.0, a.distance_nm - next_edge)
                        t_to_edge = dist_to_edge / nm_per_min if nm_per_min > 0 else np.inf

                        # Si va a cruzar el umbral dentro del minuto, registrar fracción exacta
                        if next_edge == 0.0 and t_to_edge <= remaining:
                            a.cross_frac_in_minute = 1.0 - remaining + t_to_edge
                            a.distance_nm = 0.0
                            remaining -= t_to_edge
                            break

                        # paso efectivo en este minuto
                        dt = min(remaining, t_to_edge)
                        a.distance_nm = max(0.0, a.distance_nm - nm_per_min * dt)
                        remaining -= dt
                else:
                    # go-around: se aleja
                    #a.distance_nm += delta_nm
                    a.distance_nm += a.speed_kt / 60.0

        # 3b) Si la pista está cerrada, forzar go-around a los que estén cerca del umbral
        if self.runway_closed(minute):
            for a in self.aircrafts:
                if a.status == "approach" and 0.0 < a.distance_nm <= CLOSURE_FORCE_GO_AROUND_DISTANCE_NM:
                    a.status = "go_around"
                    a.speed_kt = GO_AROUND_SPEED_KT
                    if a.go_around_start_minute is None:
                        a.go_around_start_minute = minute
                    a.go_around_attempts += 1
                    self.go_around_events += 1

        # 4) Aterrizajes (si distancia llegó a 0, y pista abierta y separación respetada)
        #    Procesamos en orden de menor distancia (candidatos a aterrizar)
        if not self.runway_closed(minute):
            approaching = [a for a in self.aircrafts if a.status == "approach" and a.distance_nm <= 0.0]
            # Ordenar por spawn para desempate
            approaching.sort(key=lambda x: x.spawn_minute)
            for a in approaching:
                # Chequear separación temporal vs último aterrizaje
                # Ojo: como timbramos aterrizaje en minute+1, comparar contra (minute+1)
                if (self.last_landing_minute is None) or ((minute + 1) - self.last_landing_minute >= RUNWAY_SEP_MIN):
                    # >>> cambio clave: timbrar al minuto siguiente <<<
                    t_land = minute + 1
                    a.status = "landed"
                    a.landing_minute = t_land
                    # tiempo fraccional real del cruce (fallback al fin del minuto)
                    frac = a.cross_frac_in_minute if a.cross_frac_in_minute is not None else 1.0
                    a.landing_time_min = float(minute + frac)
                    self.timeline_landings.append(t_land)
                    self.last_landing_minute = t_land
                else:
                    # No puede aterrizar: vuelve a approach con +dist (simula "round out" corto)
                    a.distance_nm = max(a.distance_nm, 1.0)
                    a.cross_frac_in_minute = None
                    # Dará otra vuelta en el circuito; la separación se resolverá luego

        # 5) Suceso meteorológico: interrupciones al azar (día ventoso)
        if self.cfg.windy_day:
            for a in self.aircrafts:
                if a.status == "approach" and a.needs_interruption and a.distance_nm <= 8.0 and a.distance_nm > 0.0:
                    # forzar un go-around una única vez (simple)
                    a.status = "go_around"
                    a.speed_kt = GO_AROUND_SPEED_KT
                    a.needs_interruption = False
                    if a.go_around_start_minute is None:
                        a.go_around_start_minute = minute
                    a.go_around_attempts += 1
                    self.go_around_events += 1

        # 6) Gestionar go-arounds y posibles reinserciones
        self.handle_go_around(minute)

        # 7) Calcular congestión: al menos un avión en approach volando más lento que su velocidad máxima
        congested = False
        for a in self.aircrafts:
            if a.status == "approach":
                vmax = desired_speed_max_for_distance_nm(a.distance_nm)
                if a.speed_kt < vmax - 1e-6:
                    congested = True
                    break
        if congested:
            self.congestion_time += 1

        # NUEVO: snapshot de trazas al final del minuto
        self._trace_snapshot(minute)

    def _trace_snapshot(self, minute: int):
        if not (self._trace_all or self._trace_ids):
            return
        closed = self.runway_closed(minute)
        for a in self.aircrafts:
            if self._trace_all or (a.id in self._trace_ids):
                self._traces.setdefault(a.id, []).append({
                    "minute": int(minute),
                    "distance_nm": float(a.distance_nm),
                    "speed_kt": float(a.speed_kt),
                    "status": a.status,
                    "eta_min": float(a.eta_minutes()),
                    "runway_closed": bool(closed),
                    "landing_minute": (int(a.landing_minute) if a.landing_minute is not None else None),
                    "landing_time_min": (float(a.landing_time_min) if a.landing_time_min is not None else None),
                })
    # ---------------------------
    # Correr la simulación completa
    # ---------------------------
    def run(self) -> SimulationResult:
        for minute in range(self.cfg.duration_minutes):
            self.step(minute)

        # Métricas
        landed_aircrafts = [a for a in self.aircrafts if a.status == "landed"]
        landed = len(landed_aircrafts)
        diverted = self.diverted_count
        delays = []
        ideal_cont = ideal_time_minutes_continuous()
        for a in landed_aircrafts:
            # Usar tiempo fraccional si está disponible
            t_real = a.landing_time_min if a.landing_time_min is not None else float(a.landing_minute)
            delays.append((t_real - a.spawn_minute) - ideal_cont)

        avg_delay = float(np.mean(delays)) if delays else 0.0
        return SimulationResult(
            landed=landed,
            diverted=diverted,
            avg_delay_minutes=avg_delay,
            delays=delays,
            go_arounds=self.go_around_events,
            timeline_landings=self.timeline_landings,
            aircraft_log=self.aircrafts,
            congestion_time=self.congestion_time,

            traces=self._traces  # <-- AGREGAR ESTA LÍNEA
        )

# ---------------------------
# Utilidades: tiempo ideal desde 100 nm a 0
# ---------------------------
def ideal_time_minutes_continuous() -> float:
    """
    Tiempo ideal continuo (~23.4 min) usando velocidades máximas por banda,
    sin redondear a minutos enteros.
    """
    return 10.0 + 8.4 + 3.0 + 2.0

# Compatibilidad: alias conservando nombre anterior
def ideal_time_minutes() -> float:
    """Compat: retorna el tiempo ideal continuo (23.4)."""
    return ideal_time_minutes_continuous()

# ---------------------------
# Experimentos Monte Carlo por batch
# ---------------------------
def run_batch(lambdas, reps=50, seed=None, windy_day=False, closure_window=None, duration_minutes=OPERATION_MINUTES):
    """
    Corre simulaciones para cada lambda y devuelve métricas promedio y errores estándar.
    """
    results = {}
    rng = np.random.default_rng(seed)
    for lam in lambdas:
        landed, diverted, delays, goas = [], [], [], []
        for r in range(reps):
            cfg = SimulationConfig(lambda_per_min=lam, windy_day=windy_day, closure_window=closure_window, seed=int(rng.integers(0, 1e9)), duration_minutes=int(duration_minutes))
            sim = AEPSimulator(cfg)
            out = sim.run()
            landed.append(out.landed)
            diverted.append(out.diverted)
            goas.append(out.go_arounds)
            if out.delays:
                delays.append(np.mean(out.delays))
            else:
                delays.append(np.nan)

        def mean_nan(x):
            x = np.asarray(x, dtype=float)
            return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float('nan')

        def stderr_nan(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            return float(np.std(x, ddof=1)/np.sqrt(len(x))) if len(x) > 1 else float('nan')

        results[lam] = {
            "landed_mean": float(np.mean(landed)), "landed_se": float(np.std(landed, ddof=1)/np.sqrt(len(landed))) if len(landed)>1 else 0.0,
            "diverted_mean": float(np.mean(diverted)), "diverted_se": float(np.std(diverted, ddof=1)/np.sqrt(len(diverted))) if len(diverted)>1 else 0.0,
            "avg_delay_mean": mean_nan(delays), "avg_delay_se": stderr_nan(delays),
            "go_around_mean": float(np.mean(goas)), "go_around_se": float(np.std(goas, ddof=1)/np.sqrt(len(goas))) if len(goas)>1 else 0.0,
        }
    return results
