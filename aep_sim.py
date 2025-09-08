
# aep_sim.py
# -*- coding: utf-8 -*-

"""
Simulador AEP - TP1 ACN 2025

Se basa en técnicas vistas en clase (numpy.random, Monte Carlo, visualizaciones con matplotlib).
Modelo discreto en pasos de 1 minuto desde las 06:00 a 24:00 (1080 min).

Reglas clave del enunciado (resumen):
- Arribos: en cada minuto aparece un avión a 100 mn con probabilidad lambda_ (Bernoulli)
- Velocidades por banda de distancia (nudos): 
  >100 mn: 300–500 (usamos 500 como "máx despejado")
  100–50: 250–300 (usamos 300 como máx)
  50–15: 200–250 (usamos 250 como máx)
  15–5: 150–200 (usamos 200 como máx)
  5–0: 120–150 (usamos 150 como máx)
- Separación mínima entre aterrizajes: 4 min; objetivo operativo: 5 min de gap (buffer).
- Si un avión queda a <4 min del anterior, reduce su velocidad en 20 kt vs el de adelante hasta lograr ≥5 min de separación.
- Si la reducción empuja por debajo del mínimo permitido de su banda, hace go-around: vira 180° y se aleja a 200 kt;
  idealmente, debe reinsertarse cuando encuentre un hueco ≥10 min y esté a >5 mn del umbral.
  (En esta versión inicial implementamos una estrategia simple de rejoin; puede refinarse.)

Extensiones (hooks ya previstos en el código):
- Día ventoso: cada avión tiene prob 0.1 de necesitar interrupción (go-around). 
- Cierre sorpresivo de 30 min: los aterrizajes quedan bloqueados durante la ventana.
"""

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
RUNWAY_TARGET_GAP = 5        # objetivo operativo (buffer)
REJOIN_REQUIRED_GAP = 10     # gap requerido para reinsertarse tras go-around
GO_AROUND_SPEED_KT = 200     # velocidad durante alejamiento
MIN_REJOIN_DISTANCE_NM = 5   # para reinsertarse, estar > 5 mn de AEP

# Bandas de distancia y velocidades (min, max) en nudos
# Usamos los máximos como "desired speed" en camino despejado.
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
    # Si está más allá de 100 mn (fuera del radar), usamos el primer rango como proxy
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
    # Bandera para interrupciones meteo (viento)
    needs_interruption: bool = False
    # Para rejoin simple
    rejoining: bool = False

    def eta_minutes(self) -> float:
        """ETA al umbral asumiendo mantener speed_kt constante. (Aproximado)"""
        if self.speed_kt <= 0:
            return np.inf
        hours = self.distance_nm / self.speed_kt
        return hours * 60.0

@dataclass
class SimulationConfig:
    lambda_per_min: float = 0.1         # probabilidad de arribo por minuto
    windy_day: bool = False             # si True, cada avión tiene 0.1 de interrupción (go-around)
    closure_window: Optional[Tuple[int,int]] = None  # (t_start, t_end) minutos donde no se puede aterrizar
    seed: Optional[int] = None

@dataclass
class SimulationResult:
    landed: int
    diverted: int
    avg_delay_minutes: float
    delays: List[float]
    go_arounds: int
    timeline_landings: List[int]  # minutos de aterrizaje
    aircraft_log: List[Aircraft]  # estado final de cada avión

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
        """Ordena por distancia (más cerca primero) y ajusta velocidad de followers."""
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

            # Si el follower llegaría con menos de RUNWAY_SEP_MIN detrás del líder, debe frenar
            if eta_foll - eta_lead < RUNWAY_SEP_MIN:
                # nueva velocidad = velocidad del líder - 20 kt, respetando mínimo de banda
                vmin, vmax = speed_limits_for_distance_nm(foll.distance_nm)
                target = max(vmin, lead.speed_kt - 20.0)
                foll.speed_kt = min(target, desired)  # no superar desired
                # Recalcular ETA con nueva velocidad
                if foll.eta_minutes() - eta_lead < (RUNWAY_TARGET_GAP):
                    # Sigue muy cerca: empujará más abajo del mínimo -> go-around
                    # Regla: si target == vmin y aún no se logra el buffer, ir a go-around
                    if foll.speed_kt <= vmin + 1e-6:
                        foll.status = "go_around"
                        foll.speed_kt = GO_AROUND_SPEED_KT
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
        Podrá reinsertarse cuando:
          - su ETA calculada con velocidad deseada daría un aterrizaje ≥ last_landing + 10
          - y su distancia sea > 5 nm
        Si no logra reinserción antes de salirse del radar (>100 nm), se desvía a Montevideo.
        """
        for a in self.aircrafts:
            if a.status == "go_around":
                a.go_around_count += 0  # contadores si se desean por evento único
                # Condición de rejoin
                desired = desired_speed_max_for_distance_nm(a.distance_nm)
                eta = a.distance_nm / max(1e-6, desired) * 60.0
                ok_gap = True
                if self.last_landing_minute is not None:
                    ok_gap = (minute + eta) >= (self.last_landing_minute + REJOIN_REQUIRED_GAP)
                far_enough = a.distance_nm >= MIN_REJOIN_DISTANCE_NM
                if ok_gap and far_enough:
                    a.status = "approach"  # reinsertado
                    a.speed_kt = desired
                    continue

                # Si se fue más allá de 120 nm (salió del área), lo consideramos desvío a Montevideo
                if a.distance_nm > 120.0:
                    a.status = "diverted"
                    a.diverted = True
                    self.diverted_count += 1

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
                    a.distance_nm = a.distance_nm + delta_nm

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

        # 5) Meteo: interrupciones al azar (día ventoso)
        if self.cfg.windy_day:
            for a in self.aircrafts:
                if a.status == "approach" and a.needs_interruption and a.distance_nm <= 8.0 and a.distance_nm > 0.0:
                    # forzar un go-around una única vez (simple)
                    a.status = "go_around"
                    a.speed_kt = GO_AROUND_SPEED_KT
                    a.needs_interruption = False
                    self.go_around_events += 1

        # 6) Gestionar go-arounds y posibles reinserciones
        self.handle_go_around(minute)

    # ---------------------------
    # Correr la simulación completa
    # ---------------------------
    def run(self) -> SimulationResult:
        for minute in range(OPERATION_MINUTES):
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
            aircraft_log=self.aircrafts
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
def run_batch(lambdas, reps=50, seed=None, windy_day=False, closure_window=None):
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
