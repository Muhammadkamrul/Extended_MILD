
import os, random, numpy as np, tensorflow as tf

def set_global_determinism(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
