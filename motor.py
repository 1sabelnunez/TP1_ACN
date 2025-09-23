
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
    (100, np.inf, 300, 500),  # >100 mn
    (50, 100, 250, 300),
    (15, 50, 200, 250),
    (5, 15, 150, 200),
    (0, 5, 120, 150),
]

def speed_limits_for_distance_nm(d_nm: float) -> Tuple[int, int]:
    for lo, hi, vmin, vmax in DIST_BANDS:
        if lo <= d_nm < hi:
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

    def eta_minutes(self) -> float:
        """ETA al umbral asumiendo mantener speed_kt constante. (Aproximado)"""
        if self.speed_kt <= 0:
            return np.inf
        hours = self.distance_nm / self.speed_kt
        return hours * 60.0

@dataclass
class SimulationConfig:
    lambda_per_min: float         # probabilidad de arribo por minuto 
    windy_day: bool = False             # si es True, cada avión tiene 0.1 de interrupción (go-around)
    closure_window: Optional[Tuple[int,int]] = None  # (t_start, t_end) minutos donde no se puede aterrizar
    seed: Optional[int] = None
    duration_minutes: int = OPERATION_MINUTES

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
        """Ordena por distancia (más cerca primero) y ajusta velocidad de los siguientes aviones."""
        approaching = [a for a in self.aircrafts if a.status == "approach"]
        # Ordenar por distancia al umbral ascendente
        approaching.sort(key=lambda x: x.distance_nm)
        for i in range(1, len(approaching)):
            lead = approaching[i-1]
            foll = approaching[i]

            # Velocidad deseada por banda (máxima)
            desired = desired_speed_max_for_distance_nm(foll.distance_nm)

            # ETA actuales (minutos) si mantuvieran la velocidad
            eta_lead = lead.eta_minutes()
            eta_foll = foll.eta_minutes()

            # Si el siguiente llegaría con menos de RUNWAY_SEP_MIN detrás del líder, debe frenar
            if eta_foll - eta_lead < RUNWAY_SEP_MIN:
                # nueva velocidad = velocidad del líder - 20 kt, respetando buffer de distancia
                vmin, vmax = speed_limits_for_distance_nm(foll.distance_nm)
                target = max(vmin, lead.speed_kt - 20.0)
                foll.speed_kt = min(target, desired)  # no superar desired
                # Recalcular ETA con nueva velocidad
                if foll.eta_minutes() - eta_lead < (RUNWAY_TARGET_GAP):
                    # Sigue muy cerca: irá más abajo del mínimo -> go-around
                    # Regla: si target == vmin y todavía no se logra el buffer, ir a go-around
                    if foll.speed_kt <= vmin + 1e-6 and foll.status != "go_around":
                        foll.status = "go_around"
                        foll.speed_kt = GO_AROUND_SPEED_KT
                        if foll.go_around_start_minute is None:
                            foll.go_around_start_minute = minute
                        foll.go_around_attempts += 1
                        self.go_around_events += 1

            else:
                # Camino despejado: usa desired speed
                foll.speed_kt = desired

        # El líder usa su deseado
        if approaching:
            lead0 = approaching[0]
            lead0.speed_kt = desired_speed_max_for_distance_nm(lead0.distance_nm)

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
                    # se acerca (disminuye distancia)
                    a.distance_nm = max(0.0, a.distance_nm - delta_nm)
                else:
                    # go-around: se aleja (aumenta distancia)
                    pass

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
                if (self.last_landing_minute is None) or (minute - self.last_landing_minute >= RUNWAY_SEP_MIN):
                    a.status = "landed"
                    a.landing_minute = minute
                    self.timeline_landings.append(minute)
                    self.last_landing_minute = minute
                else:
                    # No puede aterrizar: vuelve a approach con +dist (simula "round out" corto)
                    a.distance_nm = max(a.distance_nm, 1.0)
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
        for a in landed_aircrafts:
            # "Retardo" vs. trayectoria despejada: estimamos como (landing_minute - spawn_minute) - T_min_clear
            # T_min_clear ~ 23.4 min (de 100 nm a 0 siguiendo velocidades máximas). Calculamos en función de bandas.
            ideal = ideal_time_minutes()
            delays.append((a.landing_minute - a.spawn_minute) - ideal)

        avg_delay = float(np.mean(delays)) if delays else 0.0
        return SimulationResult(
            landed=landed,
            diverted=diverted,
            avg_delay_minutes=avg_delay,
            delays=delays,
            go_arounds=self.go_around_events,
            timeline_landings=self.timeline_landings,
            aircraft_log=self.aircrafts,
            congestion_time=self.congestion_time
        )

# ---------------------------
# Utilidades: tiempo ideal desde 100 nm a 0
# ---------------------------
def ideal_time_minutes() -> float:
    """
    Tiempo ~23.4 min usando velocidades máximas por banda:
      100-50 nm @ 300 kt -> 50/300 h = 0.1667 h = 10.0 min
      50-15  nm @ 250 kt -> 35/250 h = 0.14   h = 8.4  min
      15-5   nm @ 200 kt -> 10/200 h = 0.05   h = 3.0  min
      5-0    nm @ 150 kt -> 5/150  h = 0.0333 h = 2.0  min
      Total ≈ 23.4 min
    """
    return 10.0 + 8.4 + 3.0 + 2.0

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
            cfg = SimulationConfig(lambda_per_min=lam, windy_day=windy_day, closure_window=closure_window, seed=int(rng.integers(0, 1e9)))
            sim = AEPSimulator(cfg)
            out = sim.run()
            landed.append(out.landed)
            diverted.append(out.diverted)
            goas.append(out.go_arounds)
            if out.delays:
                delays.append(np.mean(out.delays))
            else:
                delays.append(0.0)
        def mean(x): return float(np.mean(x))
        def stderr(x): return float(np.std(x, ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0
        results[lam] = {
            "landed_mean": mean(landed), "landed_se": stderr(landed),
            "diverted_mean": mean(diverted), "diverted_se": stderr(diverted),
            "avg_delay_mean": mean(delays), "avg_delay_se": stderr(delays),
            "go_around_mean": mean(goas), "go_around_se": stderr(goas),
        }
    return results
