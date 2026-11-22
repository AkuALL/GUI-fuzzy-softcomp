import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mamdani import (
    FEATURES,
    INPUT_MEMBERSHIPS,
    OUTPUT_CENTERS,
    PRIORITY,
    SCORE_GRID,
    STATES,
    aggregate_output,
    membership,
    output_sets,
    predict,
)


st.set_page_config(page_title="Penilaian Awal Kualitas Air", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 2rem;}
    h1 {font-size: clamp(1.7rem, 3.4vw, 2.4rem) !important; line-height: 1.12 !important;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e9e6;
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] {color: #58665f;}
    .eyebrow {color: #087f78; font-size: .76rem; font-weight: 700; letter-spacing: .12em;}
    </style>
    """,
    unsafe_allow_html=True,
)


CLASS_COLORS = {"Poor": "#b44c4c", "Good": "#d09324", "Excellent": "#167d68"}
STATE_COLORS = {"Critical": "#b44c4c", "Caution": "#d09324", "Optimal": "#167d68"}
DISPLAY_FEATURES = ["Suhu air (°C)", "pH", "DO (mg/L)"]
DEFAULT_SCENARIO = "Contoh hitung (Good)"
SCENARIOS = {
    DEFAULT_SCENARIO: (27.0, 7.5, 1.68),
    "Semua Optimal (Excellent)": (27.0, 7.5, 3.0),
    "Satu input Critical (Poor)": (15.5, 7.5, 1.68),
}
RULES = [
    {"ID": "R1", "Jika": "Suhu, pH, dan DO Optimal", "Maka": "Excellent"},
    {
        "ID": "R2",
        "Jika": "Semua input Caution/Optimal dan minimal satu Caution",
        "Maka": "Good",
    },
    {"ID": "R3", "Jika": "Minimal satu input Critical", "Maka": "Poor"},
]


def sync_widget(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]
    st.session_state["scenario_choice"] = "Atur manual"


def apply_scenario():
    values = SCENARIOS.get(st.session_state["scenario_choice"])
    if values is None:
        return
    for key, value in zip(("temperature", "ph", "oxygen"), values):
        st.session_state[f"{key}_slider"] = value
        st.session_state[f"{key}_number"] = value


def reset_scenario():
    st.session_state["scenario_choice"] = DEFAULT_SCENARIO
    apply_scenario()


def dual_input(label, key, minimum, maximum, default, step, number_format, open_upper=False):
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"
    st.session_state.setdefault(slider_key, default)
    st.session_state.setdefault(number_key, default)

    slider_maximum = maximum
    if open_upper:
        slider_maximum = max(
            float(maximum),
            float(st.session_state[slider_key]),
            float(st.session_state[number_key]),
        )

    st.markdown(f"**{label}**")
    slider_col, number_col = st.columns([3, 2])
    slider_col.slider(
        label,
        min_value=float(minimum),
        max_value=float(slider_maximum),
        step=float(step),
        key=slider_key,
        label_visibility="collapsed",
        on_change=sync_widget,
        args=(slider_key, number_key),
    )
    number_col.number_input(
        label,
        min_value=float(minimum),
        max_value=None if open_upper else float(maximum),
        step=float(step),
        format=number_format,
        key=number_key,
        label_visibility="collapsed",
        on_change=sync_widget,
        args=(number_key, slider_key),
    )
    return float(st.session_state[number_key])


def chart_theme(chart):
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            gridColor="#edf1ee",
            labelColor="#64716b",
            titleColor="#4d5b55",
            domainColor="#d6dfda",
            tickColor="#d6dfda",
            labelFontSize=10,
            titleFontSize=10,
        )
        .configure_legend(
            orient="bottom",
            labelColor="#58665f",
            labelFontSize=10,
            symbolSize=80,
        )
        .configure_title(fontSize=12, color="#33413b", anchor="start")
    )


def membership_chart(feature, display_name, value):
    maximum = {FEATURES[0]: 50.0, FEATURES[1]: 14.0}.get(feature, 15.0)
    if feature == FEATURES[2]:
        maximum = max(maximum, value * 1.1, value + 0.5)

    grid = np.linspace(0, maximum, 301)
    degrees = membership(grid, feature)
    curve_data = pd.DataFrame(
        {
            "Nilai": np.tile(grid, len(STATES)),
            "Derajat": np.concatenate([degrees[:, index] for index in range(len(STATES))]),
            "Zona": np.repeat(STATES, len(grid)),
        }
    )
    selected_degrees = membership(np.array([value]), feature)[0]
    point_data = pd.DataFrame(
        {"Nilai": [value] * len(STATES), "Derajat": selected_degrees, "Zona": STATES}
    )
    cursor_data = pd.DataFrame({"Nilai": [value]})

    colors = alt.Scale(domain=STATES, range=[STATE_COLORS[state] for state in STATES])
    visibility = alt.selection_point(name="zona_aktif", fields=["Zona"], bind="legend")
    zoom = alt.selection_interval(name="zoom_x", bind="scales", encodings=["x"])

    curves = (
        alt.Chart(curve_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Nilai:Q", title=display_name, scale=alt.Scale(domain=[0, maximum], nice=False)),
            y=alt.Y("Derajat:Q", title="Derajat keanggotaan", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Zona:N", title=None, scale=colors),
            opacity=alt.condition(visibility, alt.value(1), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("Zona:N", title="Zona"),
                alt.Tooltip("Nilai:Q", title="Nilai", format=".2f"),
                alt.Tooltip("Derajat:Q", title="Derajat", format=".3f"),
            ],
        )
        .add_params(visibility, zoom)
    )
    selected_points = alt.Chart(point_data).mark_point(
        filled=True, size=60, stroke="#ffffff", strokeWidth=1
    ).encode(
        x=alt.X("Nilai:Q", scale=alt.Scale(domain=[0, maximum], nice=False)),
        y=alt.Y("Derajat:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Zona:N", scale=colors),
        opacity=alt.condition(visibility, alt.value(1), alt.value(0.12)),
        tooltip=[
            alt.Tooltip("Zona:N", title="Zona"),
            alt.Tooltip("Nilai:Q", title="Nilai dipilih", format=".2f"),
            alt.Tooltip("Derajat:Q", title="Derajat", format=".3f"),
        ],
    )
    cursor = alt.Chart(cursor_data).mark_rule(
        color="#27343b", strokeDash=[5, 4], strokeWidth=1.2
    ).encode(x=alt.X("Nilai:Q", scale=alt.Scale(domain=[0, maximum], nice=False)))

    return chart_theme(
        alt.layer(curves, selected_points, cursor).properties(
            width="container",
            height=240,
            title=f"{display_name}: {value:.2f}",
        )
    )


def output_chart(activation, score):
    aggregate = aggregate_output(activation)[0]
    aggregate_data = pd.DataFrame({"Skor": SCORE_GRID, "Derajat": aggregate})

    curve_rows = []
    for index, category in enumerate(PRIORITY):
        clipped = np.minimum(activation[0, index], output_sets[index])
        curve_rows.extend(
            {"Skor": x, "Derajat": y, "Kategori": category}
            for x, y in zip(SCORE_GRID, clipped)
        )
    curve_data = pd.DataFrame(curve_rows)
    centroid_data = pd.DataFrame({"Skor": [score]})

    colors = alt.Scale(
        domain=list(PRIORITY),
        range=[CLASS_COLORS[str(category)] for category in PRIORITY],
    )
    visibility = alt.selection_point(name="kelas_aktif", fields=["Kategori"], bind="legend")
    zoom = alt.selection_interval(name="zoom_skor", bind="scales", encodings=["x"])

    aggregate_area = alt.Chart(aggregate_data).mark_area(
        color="#73958b", opacity=0.20
    ).encode(
        x=alt.X("Skor:Q", title="Skor keluaran fuzzy", scale=alt.Scale(domain=[0, 100], nice=False)),
        y=alt.Y("Derajat:Q", title="Derajat keanggotaan", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("Skor:Q", title="Skor", format=".1f"),
            alt.Tooltip("Derajat:Q", title="Agregasi MAX", format=".3f"),
        ],
    )
    output_curves = (
        alt.Chart(curve_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Skor:Q", title="Skor keluaran fuzzy", scale=alt.Scale(domain=[0, 100], nice=False)),
            y=alt.Y("Derajat:Q", title="Derajat keanggotaan", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Kategori:N", title=None, scale=colors),
            opacity=alt.condition(visibility, alt.value(1), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("Kategori:N", title="Keluaran"),
                alt.Tooltip("Skor:Q", title="Skor", format=".1f"),
                alt.Tooltip("Derajat:Q", title="Derajat", format=".3f"),
            ],
        )
        .add_params(visibility, zoom)
    )
    centroid_rule = alt.Chart(centroid_data).mark_rule(
        color="#27343b", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(x=alt.X("Skor:Q", scale=alt.Scale(domain=[0, 100], nice=False)))

    return chart_theme(
        alt.layer(aggregate_area, output_curves, centroid_rule).properties(
            width="container",
            height=230,
        )
    )


with st.sidebar:
    st.header("Masukan skenario")
    st.caption("Geser atau ketik nilai.")
    st.selectbox(
        "Preset skenario",
        [DEFAULT_SCENARIO, *[name for name in SCENARIOS if name != DEFAULT_SCENARIO], "Atur manual"],
        key="scenario_choice",
        on_change=apply_scenario,
    )
    st.button("Reset ke contoh hitung", on_click=reset_scenario, use_container_width=True)
    st.caption("Preset adalah simulasi aturan, bukan pengukuran lapangan.")
    temperature = dual_input("Suhu air (°C)", "temperature", 0, 50, 27.0, 0.1, "%.2f")
    ph = dual_input("pH", "ph", 0, 14, 7.5, 0.1, "%.2f")
    oxygen = dual_input("DO (mg/L)", "oxygen", 0, 15, 1.68, 0.01, "%.2f", open_upper=True)


st.markdown('<div class="eyebrow">SIMULASI FUZZY MAMDANI</div>', unsafe_allow_html=True)
st.title("Penilaian Awal Kualitas Air Budidaya Ikan")
st.markdown("Suhu, pH, dan DO → **Excellent · Good · Poor**")
with st.container(border=True):
    st.markdown(
        "Rentang acuan dari studi kolam lele bioflok; model ini belum divalidasi di lapangan. "
        "[Sumber](https://jtsiskom.undip.ac.id/article/download/13209/12555)."
    )
st.caption("Aturan dari literatur; notebook membandingkan hasilnya dengan label WQD.")


sample = [[temperature, ph, oxygen]]
try:
    labels, scores, activation = predict(sample)
except ValueError as error:
    st.error(str(error))
    st.stop()

category = str(labels[0])
score = float(scores[0])
activation_by_class = dict(zip(PRIORITY, activation[0]))
strongest_rule = next(rule for rule in RULES if rule["Maka"] == category)

if oxygen > 15:
    with st.container(border=True):
        st.markdown("**Peringatan rentang DO**")
        st.caption(
            "15 mg/L hanya batas tampilan awal slider, bukan ambang mutu dari sumber. "
            "Membership DO memakai shoulder kanan tanpa batas atas, sehingga nilai ini tetap Optimal; "
            "hasil di atas 15 mg/L merupakan ekstrapolasi model."
        )

st.subheader("Hasil skenario")
result_col, score_col = st.columns(2)
result_col.metric("Kategori model", category)
score_col.metric("Skor centroid (0–100)", f"{score:.2f}")
st.caption(
    f"Aturan terkuat: {strongest_rule['ID']} ({category}, α = {activation_by_class[category]:.3f}). "
    "Skor centroid bukan probabilitas."
)

tab_summary, tab_fuzzification, tab_rules, tab_output = st.tabs(
    ["Ringkasan", "Fuzzifikasi", "Basis aturan", "Agregasi & centroid"]
)

degrees = np.stack(
    [membership(np.array([sample[0][index]]), feature)[0] for index, feature in enumerate(FEATURES)]
)

with tab_summary:
    st.markdown("#### Nilai yang dinilai")
    value_columns = st.columns(3)
    for column, name, value in zip(
        value_columns,
        ["Suhu air (°C)", "pH", "DO (mg/L)"],
        sample[0],
    ):
        column.metric(name, f"{value:.2f}")

    st.markdown("#### Derajat keanggotaan")
    degree_table = pd.DataFrame(degrees, index=DISPLAY_FEATURES, columns=STATES)
    degree_table.index.name = "Indikator"
    st.dataframe(degree_table.style.format("{:.3f}"), width="stretch")

with tab_fuzzification:
    selected_feature = st.radio(
        "Indikator",
        DISPLAY_FEATURES,
        horizontal=True,
        label_visibility="collapsed",
    )
    feature_index = DISPLAY_FEATURES.index(selected_feature)
    selected_memberships = degrees[feature_index]
    degree_summary = " · ".join(
        f"{state} {degree:.3f}" for state, degree in zip(STATES, selected_memberships)
    )
    st.caption(f"Derajat terpilih: {degree_summary}")
    st.altair_chart(
        membership_chart(FEATURES[feature_index], selected_feature, sample[0][feature_index]),
        width="stretch",
        height=310,
    )
    st.caption("Hover untuk nilai tepat; klik legenda untuk sorot zona; drag sumbu x untuk zoom.")
    feature = FEATURES[feature_index]
    transitions, zones = INPUT_MEMBERSHIPS[feature]
    unit = {FEATURES[0]: "°C", FEATURES[1]: "", FEATURES[2]: "mg/L"}[feature]
    boundary_rows = [
        {
            "Peralihan zona": f"{zones[index]} → {zones[index + 1]}",
            "Rentang transisi": f"{start:g}–{end:g} {unit}".strip(),
        }
        for index, (start, end) in enumerate(transitions)
    ]
    if feature == FEATURES[2]:
        boundary_rows.append(
            {
                "Peralihan zona": "Optimal (shoulder kanan)",
                "Rentang transisi": f"≥ {transitions[-1][1]:g} mg/L; tanpa batas atas",
            }
        )
    with st.expander("Batas transisi model"):
        st.caption("Nilai yang dipakai untuk membentuk fungsi keanggotaan; sumber acuan tercantum di atas.")
        st.dataframe(pd.DataFrame(boundary_rows), hide_index=True, width="stretch")

with tab_rules:
    for rule in RULES:
        alpha = float(activation_by_class[rule["Maka"]])
        with st.container(border=True):
            st.markdown(f"**{rule['ID']} → {rule['Maka']}**")
            st.caption(f"Jika {rule['Jika'].lower()}.")
            st.progress(alpha, text=f"Aktivasi α = {alpha:.3f}")
    st.caption("Prioritas jika aktivasi sama: Poor → Good → Excellent.")

with tab_output:
    centers_by_class = dict(zip(PRIORITY, OUTPUT_CENTERS))
    center_text = " · ".join(
        f"{class_name} {center:g}" for class_name, center in centers_by_class.items()
    )
    st.caption(f"Pusat skor: {center_text}")
    st.altair_chart(output_chart(activation, score), width="stretch", height=300)
    st.caption("Hover untuk nilai tepat; klik legenda untuk sorot kelas; drag sumbu x untuk zoom.")

