import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = "out/conv1d-1/lightning_logs/version_0"

event_acc = EventAccumulator(path)
event_acc.Reload()
print(event_acc.Scalars("validation_accuracy"))