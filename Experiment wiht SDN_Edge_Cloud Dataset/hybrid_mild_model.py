
"""
hybrid_mild_model.py
--------------------
MILD‑MoE variant that accepts additional *teacher* features (e.g., Logistic (OvR) predictions)
and conditions the gating on them. Heads and experts remain the same.
"""
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

def create_mild_moe_with_teacher(num_features: int, num_teacher: int, intent_ids,
                                 l2=1e-5, dropout=0.2, gate_units=48,
                                 enc_units=(96,96), use_teacher_gate=True):
    # Inputs
    x_in = layers.Input(shape=(num_features,), name='x')
    inputs = [x_in]

    if use_teacher_gate:
        t_in = layers.Input(shape=(num_teacher,), name='teacher_in')
        inputs.append(t_in)

    # Shared encoder from raw features
    x = x_in
    for u in enc_units:
        x = layers.Dense(u, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
    shared = x

    # Gate
    if use_teacher_gate:
        gx = layers.Concatenate(name='gate_fuse')([shared, t_in])
    else:
        gx = shared

    g = layers.Dense(gate_units, activation='relu')(gx)
    g = layers.Dropout(dropout)(g)
    gate_probs = layers.Dense(len(intent_ids), activation='softmax', name='gate')(g)

    # Experts & heads
    head_hiddens = []
    outputs = []
    for i, intent in enumerate(intent_ids):
        e = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2), name=f'expert_{intent}_1')(shared)
        e = layers.Dropout(dropout)(e)
        e = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), name=f'expert_{intent}_2')(e)
        gi = layers.Lambda(lambda t: tf.expand_dims(t[:, i], -1), name=f'gate_pick_{intent}')(gate_probs)
        e = layers.Multiply(name=f'gate_mul_{intent}')([e, gi])
        h = layers.Dense(16, activation='relu', name=f'head_{intent}')(e)
        head_hiddens.append(h)
        outputs.append(layers.Dense(1, activation='sigmoid', name=f'{intent}_out')(h))

    concat_hidden = layers.Concatenate(name='head_concat')(head_hiddens)
    model = models.Model(inputs=inputs, outputs=outputs + [gate_probs, concat_hidden], name='MILD-MoE-HYBRID')
    return model

def create_mild_moe_with_teacher_old(num_features: int, num_teacher: int, intent_ids,
                                 l2=1e-5, dropout=0.2, gate_units=48, enc_units=(96,96), use_teacher_gate=True):
    # Inputs
    x_in = layers.Input(shape=(num_features,), name='x')
    t_in = layers.Input(shape=(num_teacher,), name='teacher_in')  # per‑intent teacher probs/dist (K,)

    # Shared encoder from raw features
    x = x_in
    for u in enc_units:
        x = layers.Dense(u, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
    shared = x

    # Gating conditions on both shared features and teacher distribution
    if use_teacher_gate:
        gx = layers.Concatenate(name='gate_fuse')([shared, t_in])
    else:
        gx = shared
    #gx = layers.Concatenate(name='gate_fuse')([shared, t_in])
    g = layers.Dense(gate_units, activation='relu')(gx)
    g = layers.Dropout(dropout)(g)
    gate_probs = layers.Dense(len(intent_ids), activation='softmax', name='gate')(g)

    # Experts & heads (same as MILD)
    head_hiddens = []
    outputs = []
    for i, intent in enumerate(intent_ids):
        e = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2), name=f'expert_{intent}_1')(shared)
        e = layers.Dropout(dropout)(e)
        e = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2), name=f'expert_{intent}_2')(e)
        gi = layers.Lambda(lambda t: tf.expand_dims(t[:, i], -1), name=f'gate_pick_{intent}')(gate_probs)
        e = layers.Multiply(name=f'gate_mul_{intent}')([e, gi])
        h = layers.Dense(16, activation='relu', name=f'head_{intent}')(e)
        head_hiddens.append(h)
        out = layers.Dense(1, activation='sigmoid', name=f'{intent}_out')(h)
        outputs.append(out)

    concat_hidden = layers.Concatenate(name='head_concat')(head_hiddens)

    model = models.Model(inputs=[x_in, t_in], outputs=outputs + [gate_probs, concat_hidden], name='MILD-MoE-HYBRID')
    return model
