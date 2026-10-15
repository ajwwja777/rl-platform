"""Add monitoring outside the checkpoint-signed learner implementation."""
import logging
import online_learner
from monitoring import LearnerHistory


online_learner_base_infer = online_learner.Learner.infer


class ObservedLearner(online_learner.Learner):
    def infer(self, route, data):
        import sys
        from monitoring import install_selection_observer, infer_with_q
        install_selection_observer(sys.modules[type(self.agent).__module__])
        return infer_with_q(self, online_learner_base_infer, route, data)

    def save_status(self):
        value = super().save_status()
        try:
            if not hasattr(self, '_monitor_history'):
                self._monitor_history = LearnerHistory(self.root, self.meta.get('critic_updates', 0))
            self._monitor_history.observe(value, self.metrics)
            # The runtime consumes this bounded history and removes it before ROS gate status.
            value['monitoring'] = self._monitor_history.payload()
            online_learner.atomic_json(self.root / 'status.json', value)
        except Exception:
            logging.exception('EXPO monitoring write failed; learner state unaffected')
        return value


if __name__ == '__main__':
    online_learner.Learner = ObservedLearner
    online_learner.main()
