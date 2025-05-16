from elliot.run import run_experiment
import tensorflow as tf
from elliot.evaluation.metrics.settings import Settings
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
#run_experiment("thesis_sources/configs/tuning/tuning.yml")
#run_experiment("thesis_sources/configs/test/train_test_config.yml")
# RERUN
Settings.current_file = 0
for file in Settings.files:
    #if file not in ['item_knn', 'bprmf', 'multivae', 'mf2020']:
    #    continue
    run_experiment(f"thesis_sources/configs/rerun/{file}.yml")
    Settings.current_file += 1