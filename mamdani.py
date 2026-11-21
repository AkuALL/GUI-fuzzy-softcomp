"""Logika fuzzy Mamdani kualitas air (disalin dari notebook, tanpa perubahan rumus)."""
import numpy as np

FEATURES = ["Temperature (°C)", "pH", "Dissolved oxygen (mg/L)"]
STATES = ["Critical", "Caution", "Optimal"]
CLASSES = ["Excellent", "Good", "Poor"]
PRIORITY = np.array(["Poor", "Good", "Excellent"])
SCORE_GRID = np.linspace(0, 100, 1001)
OUTPUT_CENTERS = np.array([20, 50, 80])
OUTPUT_HALF_WIDTH = 20
TIE_TOLERANCE = 1e-10

# Tiap input: batas awal/akhir transisi, diikuti status zona dari kiri ke kanan.
INPUT_MEMBERSHIPS = {
    "Temperature (°C)": (
        [(15.5, 16.5), (24, 25), (30, 31), (31.5, 32.5)],
        ["Critical", "Caution", "Optimal", "Caution", "Critical"],
    ),
    "pH": (
        [(3.95, 4.05), (6.4, 6.5), (8.5, 8.6), (10.95, 11.05)],
        ["Critical", "Caution", "Optimal", "Caution", "Critical"],
    ),
    "Dissolved oxygen (mg/L)": (
        [(0.45, 0.55), (1.65, 1.75)],
        ["Critical", "Caution", "Optimal"],
    ),
}


def membership(values, feature):
    '''Hitung derajat Critical, Caution, dan Optimal untuk satu input.'''
    transitions, zones = INPUT_MEMBERSHIPS[feature]
    start, end = np.array(transitions).T
    ramps = np.clip((values[:, None] - start) / (end - start), 0, 1)
    boundaries = np.column_stack([np.ones(len(values)), ramps, np.zeros(len(values))])
    zone_degrees = boundaries[:, :-1] - boundaries[:, 1:]
    return np.column_stack([
        zone_degrees[:, np.array(zones) == state].sum(axis=1) for state in STATES
    ])


output_sets = np.maximum(
    0, 1 - np.abs(SCORE_GRID - OUTPUT_CENTERS[:, None]) / OUTPUT_HALF_WIDTH
)

def input_domain(samples):
    '''Periksa kelayakan tiga kolom input sebelum inferensi.'''
    temperature, ph, oxygen = samples.T
    return (
        np.isfinite(samples).all(axis=1)
        & (temperature >= 0) & (temperature <= 50)
        & (ph >= 0) & (ph <= 14)
        & (oxygen >= 0)
    )


def rule_activation(degrees):
    '''Gabungkan derajat input sesuai tiga aturan.'''
    critical, caution, optimal = degrees.transpose(2, 0, 1)
    poor = critical.max(axis=1)
    good = np.minimum(caution.max(axis=1), np.maximum(caution, optimal).min(axis=1))
    excellent = optimal.min(axis=1)
    return np.column_stack([poor, good, excellent])


def aggregate_output(activation):
    '''Potong tiap output set lalu ambil MAX.'''
    clipped = np.minimum(activation[:, :, None], output_sets[None, :, :])
    return clipped.max(axis=1)


def predict(samples):
    '''Terima baris suhu, pH, DO. Kembalikan kelas, skor, dan aktivasi.'''
    values = np.atleast_2d(np.asarray(samples, dtype=float))
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Masukkan tiga kolom: suhu (°C), pH, dan DO (mg/L).")
    if not input_domain(values).all():
        raise ValueError("Ada nilai input yang tidak berhingga atau di luar domain.")

    degrees = np.stack([membership(values[:, i], feature) for i, feature in enumerate(FEATURES)], axis=1)
    activation = rule_activation(degrees)
    aggregated = aggregate_output(activation)
    score = (aggregated * SCORE_GRID).sum(axis=1) / aggregated.sum(axis=1)
    strongest = activation.max(axis=1, keepdims=True)
    winners = activation >= strongest - TIE_TOLERANCE
    return PRIORITY[winners.argmax(axis=1)], score, activation
