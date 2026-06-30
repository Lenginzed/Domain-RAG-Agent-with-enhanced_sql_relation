class EventDrivenReward:
    """Synthetic example reward class for mini demo."""

    def get_reward(self, event):
        if event == "missile_hit":
            return 200
        if event == "crash":
            return -200
        return 0
