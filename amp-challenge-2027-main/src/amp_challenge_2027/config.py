"""HydrAMP model configuration — ported from Keras/TF to PyTorch."""

VOCAB_SIZE = 20
VOCAB_PAD_SIZE = VOCAB_SIZE + 1  # 20 amino acids + 1 padding token (0)
MIN_LENGTH = 0
MAX_LENGTH = 25

HIDDEN_DIM = 128
RCL_WEIGHT = 64
LATENT_DIM = 64
MIN_KL = 1e-4
MAX_KL = 1e-2
KL_ANNEALRATE = 0.01
TAU_ANNEALRATE = 0.01
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0

# Standard amino acid alphabet used for encoding/decoding
STD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Loss-weight presets (16 terms)
hydra = [
    0.1,   # classifier output
    0.1,   # regressor output
    1.0,   # reconstruction
    0.10,  # regressor_mean_grad_input
    0.10,  # classifier_mean_grad_input
    0.05,  # unconstrained_sleep_regressor_output_grad_input
    0.05,  # unconstrained_sleep_classifier_output_grad_input
    0.05,  # correction_sleep_regressor_output_grad_input
    0.05,  # correction_sleep_classifier_output_grad_input
    0.05,  # correction sleep classifier output
    0.05,  # correction sleep regressor output
    0.05,  # unconstrained sleep classifier output
    0.05,  # unconstrained sleep regressor output
    0.10,  # z cond reconstructed error
    0.05,  # correction sleep cond reconstructed error
    0.05,  # unconstrained sleep cond reconstructed error
]

pepcvae = [
    0.1, 0.1, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.05, 0.05, 0.1, 0.0, 0.05,
]

basic = [
    0.1, 0.1, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
]
