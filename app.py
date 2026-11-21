import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mamdani import (FEATURES, PRIORITY, SCORE_GRID, STATES, aggregate_output,
                     membership, output_sets, predict)

st.set_page_config(page_title="Mamdani Kualitas Air", layout="wide")
st.title("Penilaian Kualitas Air Budidaya Ikan (Fuzzy Mamdani)")
st.caption("Model berbasis literatur: suhu, pH, dan DO → Excellent / Good / Poor.")

# 1. Input di sidebar. Setiap slider digeser, seluruh skrip dijalankan ulang.
def sync(src, dst):
    """Salin nilai widget `src` ke `dst` agar slider dan kotak angka selalu sama."""
    st.session_state[dst] = st.session_state[src]


def dual_input(label, key, lo, hi, default, step, fmt):
    """Slider + number_input yang tersinkron lewat session_state."""
    st.session_state.setdefault(f"{key}_s", default)
    st.session_state.setdefault(f"{key}_n", default)
    st.markdown(f"**{label}**")
    c1, c2 = st.columns([3, 2])
    c1.slider(label, lo, hi, step=step, key=f"{key}_s", label_visibility="collapsed",
              on_change=sync, args=(f"{key}_s", f"{key}_n"))
    c2.number_input(label, lo, hi, step=step, format=fmt, key=f"{key}_n",
                    label_visibility="collapsed",
                    on_change=sync, args=(f"{key}_n", f"{key}_s"))
    return st.session_state[f"{key}_s"]


with st.sidebar:
    st.header("Input")
    st.caption("Geser slider atau ketik angka langsung.")
    suhu = dual_input("Suhu (°C)", "suhu", 0.0, 50.0, 27.0, 0.1, "%.2f")
    ph = dual_input("pH", "ph", 0.0, 14.0, 7.5, 0.1, "%.2f")
    do = dual_input("DO (mg/L)", "do", 0.0, 15.0, 3.0, 0.05, "%.2f")

# 2. Inferensi memakai fungsi dari mamdani.py
sample = [[suhu, ph, do]]
try:
    labels, scores, activation = predict(sample)
except ValueError as e:
    st.error(str(e))
    st.stop()

# 3. Hasil utama
c1, c2 = st.columns(2)
c1.metric("Kategori", labels[0])
c2.metric("Skor centroid (0–100)", f"{scores[0]:.2f}")

tab1, tab2, tab3 = st.tabs(["Fuzzifikasi", "Aktivasi aturan", "Agregasi & centroid"])

with tab1:
    degrees = [membership(np.array([sample[0][i]]), feature)[0]
               for i, feature in enumerate(FEATURES)]
    st.dataframe(pd.DataFrame(degrees, index=FEATURES, columns=STATES).round(3))

with tab2:
    st.bar_chart(pd.Series(activation[0], index=PRIORITY, name="Aktivasi"))

with tab3:
    agg = aggregate_output(activation)[0]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(SCORE_GRID, agg, color="#d6e1e3", label="Agregasi")
    for name, a, curve in zip(PRIORITY, activation[0], output_sets):
        ax.plot(SCORE_GRID, [min(a, v) for v in curve], label=name)
    ax.axvline(scores[0], color="k", label=f"Centroid = {scores[0]:.2f}")
    ax.set(xlabel="Skor kualitas air (0–100)", ylabel="Derajat keanggotaan", xlim=(0, 100), ylim=(0, 1.05))
    ax.legend()
    st.pyplot(fig)

