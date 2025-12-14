# sumo_env.py

import os, sys
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os, sys
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import traci


class SumoIntersectionEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"], "render_fps": 10}


    def __init__(
        self,
        sumo_cfg: str = "simple_intersection/probabilistic.sumocfg",
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 1,          # SUMO seconds per env step
        gui_delay: float = 0.0,       # slows down gui, only needed if you want to watch it train
        reward_mode: str = "hybrid",  # "queue" and "delay" were not great, so now there is only "hybrid"
				      # without min and max below, model will cheat by cycling between the lights super fast
        min_green: float = 10.0,      # minimum green duration before switch allowed
        max_green: float = 60.0,      # maximum on green
        yellow_min: float = 3.0,      # minimum yellow duration
        yellow_max: float = 7.0,      # maximum yellow duration
    ):
        super().__init__()

        if "SUMO_HOME" not in os.environ:
            raise RuntimeError(
                "SUMO_HOME not set. Add 'export SUMO_HOME=/usr/share/sumo' (or your path) to ~/.bashrc."
            )

        assert reward_mode in ("queue", "delay", "hybrid") # "queue" and "delay" were removed
        self.reward_mode = reward_mode

        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.max_steps = int(max_steps)
        self.delta_time = int(delta_time)
        self.gui_delay = float(gui_delay)

        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_min = float(yellow_min)
        self.yellow_max = float(yellow_max)

        binary_name = "sumo-gui" if use_gui else "sumo"
        self.sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", binary_name)

        # 4 queues + 1 phase_group
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(5,), dtype=np.float32
        )

        # 0 = stay on current group, 1 = request switch to the other group
        self.action_space = spaces.Discrete(2)

        self._incoming_edges = ["north_in", "south_in", "east_in", "west_in"]
        self._tls_id: Optional[str] = None
        self._n_phases: int = 0

        # Mapping NS/EW + GREEN/YELLOW to SUMO phase indices
        self._ns_green_phase: int = 0
        self._ns_yellow_phase: int = 0
        self._ew_green_phase: int = 0
        self._ew_yellow_phase: int = 0

        # 0 = NS group, 1 = EW group
        self._current_group: int = 0
        self._mode: str = "GREEN"  # "GREEN" or "YELLOW"
        self._time_in_phase: float = 0.0
        self._yellow_remaining: float = 0.0

        # Episode counters
        self._step_count: int = 0
        self._episode_total_queue: float = 0.0
        self._episode_total_delay: float = 0.0  # in linear seconds
        self._episode_steps: int = 0

    # ------------------------------------------------------------------
    # SUMO management stuff
    # ------------------------------------------------------------------

    def _start_sumo(self) -> None:
        """Start or restart a SUMO instance under TraCI and infer TLS phase mapping."""
        if traci.isLoaded():
            traci.close()

        seed = int(np.random.randint(0, 1_000_000))

        traci.start(
            [
                self.sumo_binary,
                "-c",
                self.sumo_cfg,
                "--start",
                "--no-step-log",
                "true",
                "--seed",
                str(seed),
            ]
        )

        tls_ids = traci.trafficlight.getIDList()
        if len(tls_ids) == 0:
            raise RuntimeError("Cant find traffic light in SUMO")
        self._tls_id = tls_ids[0]

        # Get controlled lanes and TLS logic
        controlled_lanes = traci.trafficlight.getControlledLanes(self._tls_id)

        # Indices of lanes belonging to NS vs EW incoming edges
        ns_lane_indices = [
            i
            for i, lid in enumerate(controlled_lanes)
            if "north_in" in lid or "south_in" in lid
        ]
        ew_lane_indices = [
            i
            for i, lid in enumerate(controlled_lanes)
            if "east_in" in lid or "west_in" in lid
        ]

        if not ns_lane_indices or not ew_lane_indices:
            print(
                "[WARN] Couldnt label NS/EW lanes for some reason",
                controlled_lanes,
            )

        # Get TLS data
        try:
            logic_list = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                self._tls_id
            )
            if logic_list:
                logic = logic_list[0]
                phases = logic.phases
                self._n_phases = len(phases)
            else:
                phases = []
                self._n_phases = 0
        except Exception as e:
            print("[WARN] Failed to get TLS data", e)
            phases = []
            self._n_phases = 0

        # Defaults 
        self._ns_green_phase = 0
        self._ns_yellow_phase = 0
        self._ew_green_phase = 0
        self._ew_yellow_phase = 0


        def is_group_green(state: str, lane_indices) -> bool:
            # Consider G and g as green
            return lane_indices and all(state[i] in "Gg" for i in lane_indices)

        def is_group_red(state: str, lane_indices) -> bool:
            # Consider r and R as red
            return lane_indices and all(state[i] in "rR" for i in lane_indices)

        # Try to find NS-green and EW-green phases by looking at the states
        ns_green_idx = None
        ew_green_idx = None

        for idx, ph in enumerate(phases):
            s = ph.state
            # Lngth of state == number of controlled lanes
            if len(s) != len(controlled_lanes):
                continue

            # NS green, EW red
            if is_group_green(s, ns_lane_indices) and is_group_red(s, ew_lane_indices):
                ns_green_idx = idx

            # EW green, NS red
            if is_group_green(s, ew_lane_indices) and is_group_red(s, ns_lane_indices):
                ew_green_idx = idx


        # Assume yellow is the next phase
        self._ns_green_phase = ns_green_idx
        self._ns_yellow_phase = (
            (ns_green_idx + 1) % self._n_phases if self._n_phases > 1 else ns_green_idx
        )

        self._ew_green_phase = ew_green_idx
        self._ew_yellow_phase = (
            (ew_green_idx + 1) % self._n_phases if self._n_phases > 1 else ew_green_idx
        )
	# Tracking
        print(
            f"[INFO] TLS mapping: ns_green={self._ns_green_phase}, "
            f"ns_yellow={self._ns_yellow_phase}, "
            f"ew_green={self._ew_green_phase}, "
            f"ew_yellow={self._ew_yellow_phase}"
        )

    def _do_sim_steps(self, n_steps: int) -> None:
        """Advance SUMO by n_steps simulation steps."""
        for _ in range(n_steps):
            traci.simulationStep()
            if self.use_gui and self.gui_delay > 0:
                time.sleep(self.gui_delay)

    # ------------------------------------------------------------------
    # Observation & -PUNISHMENT-
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        queues = [
            traci.edge.getLastStepHaltingNumber(e) for e in self._incoming_edges
        ]
        phase_group = self._current_group  # 0 = NS, 1 = EW
        obs = np.array(queues + [phase_group], dtype=np.float32)
        return obs

    def _compute_reward(self, obs: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns The Following
        reward: float
        total_queue: float
        total_delay_seconds: float
        """
        total_queue = float(np.sum(obs[:-1]))

        # Linear delay (in seconds) for metrics
        total_delay_seconds = 0.0

        # Exponential-style penalty to make long waits more painful
        exp_delay_penalty = 0.0

        # Hyperparameters for exponential
        delay_scale = 20.0   # How many seconds before penalty really ramps
        max_scaled = 5.0     # Limtis exponent size to keep penalty from getting too extreme and confusing the model

        for e in self._incoming_edges:
            veh_ids = traci.edge.getLastStepVehicleIDs(e)
            for vid in veh_ids:
                w = traci.vehicle.getWaitingTime(vid)
                total_delay_seconds += w
                scaled = w / delay_scale

                if scaled > max_scaled:
                    scaled = max_scaled

                exp_delay_penalty += float(np.exp(scaled) - 1.0)

        # Create punishment based on mode
        if self.reward_mode == "queue":
            reward = -total_queue

        elif self.reward_mode == "delay":
            reward = -exp_delay_penalty

        else:  # "hybrid"
            reward = -(0.5 * total_queue + 0.5 * exp_delay_penalty)

        return reward, total_queue, total_delay_seconds


    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_total_queue = 0.0
        self._episode_total_delay = 0.0
        self._episode_steps = 0

        self._current_group = 0  # start with NS
        self._mode = "GREEN"
        self._time_in_phase = 0.0
        self._yellow_remaining = 0.0

        #  Fresh SUMO 
        self._start_sumo()
        self._do_sim_steps(1)
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._tls_id is None or not traci.isLoaded():
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        # Actions:
        # 0 = keep current direction
        # 1 = switch to the other direction
        want_switch = int(action) == 1

        # ----------------------------
        # GREEN / YELLOW state machine
        # ----------------------------
        if self._mode == "GREEN":
            if self._current_group == 0:
                target_phase = self._ns_green_phase
            else:
                target_phase = self._ew_green_phase

            traci.trafficlight.setPhase(self._tls_id, target_phase)
            self._time_in_phase += self.delta_time

            # Start yellow if:
            # agent wants switch && min_green satisfied, OR max_green exceeded
            if (
                (want_switch and self._time_in_phase >= self.min_green)
                or (self._time_in_phase >= self.max_green)
            ):
                self._mode = "YELLOW"
                self._time_in_phase = 0.0
                self._yellow_remaining = float(
                    np.random.uniform(self.yellow_min, self.yellow_max)
                )

        elif self._mode == "YELLOW":
            if self._current_group == 0:
                target_phase = self._ns_yellow_phase
            else:
                target_phase = self._ew_yellow_phase

            traci.trafficlight.setPhase(self._tls_id, target_phase)

            self._time_in_phase += self.delta_time
            self._yellow_remaining -= self.delta_time

            if self._yellow_remaining <= 0.0:
                self._current_group = 1 - self._current_group
                self._mode = "GREEN"
                self._time_in_phase = 0.0

        # Advance SUMO
        self._do_sim_steps(self.delta_time)
        self._step_count += 1

        obs = self._get_obs()
        reward, total_queue, total_delay = self._compute_reward(obs)

        # Track episode stats
        self._episode_total_queue += total_queue
        self._episode_total_delay += total_delay
        self._episode_steps += 1

        # Termination conditions
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            print(
                f"[DEBUG] Episode end: terminated={terminated}, "
                f"truncated={truncated}, "
                f"minExpected={traci.simulation.getMinExpectedNumber()}, "
                f"steps={self._step_count}"
            )


        info: Dict[str, Any] = {}
        if terminated or truncated:
            if self._episode_steps > 0:
                info["episode_length"] = self._episode_steps
                info["episode_avg_queue"] = (
                    self._episode_total_queue / self._episode_steps
                )
                info["episode_avg_delay"] = (
                    self._episode_total_delay / self._episode_steps
                )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if traci.isLoaded():
            traci.close()
