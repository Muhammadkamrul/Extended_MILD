
import tensorflow as tf
from tensorflow.keras import losses

def distillation_focal_loss(alpha=0.1, gamma=2.0, horizon_min=120, pos_weight=3.0, min_time_weight=0.3):
    """
    A combined loss function that balances a student's learning between ground-truth and a teacher's soft labels.
    - Uses Focal Loss for the ground-truth component.
    - Uses KL Divergence for the distillation component.
    """
    base_focal_loss_fn = lead_time_focal_bce_with_margin(
        horizon_min=horizon_min, pos_weight=pos_weight, min_time_weight=min_time_weight, gamma=gamma
    )
    kld = losses.KLDivergence()

    def loss(y_true_and_soft, y_pred):
        # Unpack the concatenated targets. Shape is (batch_size, 2)
        # Column 0: y_ttf (ground truth time-to-failure)
        # Column 1: y_soft (teacher's probability)
        y_ttf = tf.slice(y_true_and_soft, [0, 0], [-1, 1])
        y_soft = tf.slice(y_true_and_soft, [0, 1], [-1, 1])

        # 1. Calculate the standard loss against the ground truth
        ground_truth_loss = base_focal_loss_fn(y_ttf, y_pred)

        # 2. Calculate the distillation loss against the teacher's soft labels
        # KL Divergence expects probability distributions. Convert single probabilities [p] to [p, 1-p].
        y_soft_dist = tf.concat([y_soft, 1.0 - y_soft], axis=1)
        y_pred_dist = tf.concat([y_pred, 1.0 - y_pred], axis=1)
        distillation_loss = kld(y_soft_dist, y_pred_dist)

        # 3. Return the weighted average of the two losses
        return alpha * ground_truth_loss + (1.0 - alpha) * distillation_loss

    return loss



def lead_time_focal_bce_with_margin(horizon_min: int, pos_weight: float = 3.0, min_time_weight: float = 0.3,
                                    neg_margin: float = 0.2, lambda_neg: float = 0.2, gamma: float = 2.0):
    """Combines lead-time weighting, class balance, and a focal loss component."""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    eps = tf.constant(1e-6, tf.float32)

    def loss(y_ttf, y_pred):
        is_pos = tf.cast(y_ttf > 0.0, tf.float32)      # y_bin
        ttf = tf.maximum(y_ttf, eps)
        
        # --- Existing time and class weighting ---
        time_w = min_time_weight + (1.0 - min_time_weight) * tf.clip_by_value(ttf / float(horizon_min), 0.0, 1.0)
        cls_w = 1.0 + (pos_weight - 1.0) * is_pos
        w = cls_w * time_w

        # --- Base BCE loss and negative penalty ---
        per_bce = bce(is_pos, y_pred)
        neg_pen = tf.nn.relu(y_pred - neg_margin) * (1.0 - is_pos)

        # --- NEW: Focal Loss Modulating Factor ---
        p_t = y_pred * is_pos + (1.0 - y_pred) * (1.0 - is_pos)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # --- Combine everything ---
        focal_bce_loss = modulating_factor * per_bce
        
        return tf.reduce_mean(focal_bce_loss * tf.stop_gradient(tf.squeeze(w, axis=-1))) + lambda_neg * tf.reduce_mean(neg_pen)

    return loss

def lead_time_cb_bce_with_margin(horizon_min: int, pos_weight: float = 3.0, min_time_weight: float = 0.3,
                                 neg_margin: float = 0.2, lambda_neg: float = 0.2):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    eps = tf.constant(1e-6, tf.float32)
    def loss(y_ttf, y_pred):
        is_pos = tf.cast(y_ttf > 0.0, tf.float32)          # y_bin
        ttf = tf.maximum(y_ttf, eps)
        time_w = min_time_weight + (1.0 - min_time_weight) * tf.clip_by_value(ttf / float(horizon_min), 0.0, 1.0)
        cls_w = 1.0 + (pos_weight - 1.0) * is_pos
        w = cls_w * time_w
        per = bce(is_pos, y_pred)
        neg_pen = tf.nn.relu(y_pred - neg_margin) * (1.0 - is_pos)
        return tf.reduce_mean(per * tf.stop_gradient(tf.squeeze(w, axis=-1))) + lambda_neg * tf.reduce_mean(neg_pen)
    return loss

def gate_kldiv_with_sparsity(supervise_weight: float = 0.3, sparsity_weight: float = 0.005):
    kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_gate_target, p_gate):
        y_gate_target = tf.cast(y_gate_target, tf.float32)
        p_gate = tf.cast(p_gate, tf.float32)
        s = tf.reduce_sum(y_gate_target, axis=-1, keepdims=True)
        y_dist = tf.where(s > 0, y_gate_target / s, y_gate_target)  # uniform over active
        base = kld(y_dist, p_gate)                                  # (B,)
        mask = tf.cast(s > 0, tf.float32)                           # (B,1)
        kl_term = supervise_weight * tf.reduce_mean(base * tf.squeeze(mask, -1))
        sparsity_term = sparsity_weight * tf.reduce_mean(tf.reduce_sum(p_gate * (1.0 - p_gate), axis=-1))
        return kl_term + sparsity_term
    return loss

def head_decorrelation(lambda_d=1e-4):
    def loss(_, h_concat):
        x = h_concat - tf.reduce_mean(h_concat, axis=0, keepdims=True)
        cov = tf.matmul(tf.transpose(x), x) / (tf.cast(tf.shape(x)[0], tf.float32) + 1e-6)
        diag = tf.linalg.tensor_diag(tf.linalg.diag_part(cov))
        off = cov - diag
        return lambda_d * tf.reduce_mean(tf.square(off))
    return loss
