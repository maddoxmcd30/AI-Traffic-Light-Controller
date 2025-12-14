from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Reads episode-level metrics from env.info and logs them:
    (avg_queue avg_delay length)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            if "episode_avg_queue" in info:
                self.logger.record("episode/avg_queue", info["episode_avg_queue"])
            if "episode_avg_delay" in info:
                self.logger.record("episode/avg_delay", info["episode_avg_delay"])
            if "episode_length" in info:
                self.logger.record("episode/length", info["episode_length"])

        return True
