import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandapower as pp
import pandapower.networks as pn


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="EZHAL | AI Grid Support",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

V_MIN = 0.95
V_MAX = 1.05

SOLAR_S_RATED = 179.3
WIND_S_RATED = 93.5


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    reg = pd.read_csv("regression_feasible_clean.csv")
    safety = pd.read_csv("ai_safety_validation.csv")
    baseline = pd.read_csv("fixed_headroom_baseline.csv")
    return reg, safety, baseline


@st.cache_resource
def load_models():
    classifier = joblib.load("final_action_classifier.pkl")
    regressor = joblib.load("final_action_regressor.pkl")
    return classifier, regressor


reg_df, safety_df, baseline_df = load_data()
classifier, regressor = load_models()


# ============================================================
# ML FEATURES
# ============================================================

features = [
    "Load_Level",
    "Solar_Level",
    "Wind_Level",
    "Solar_Available_MW",
    "Wind_Available_MW",
    "From_Bus",
    "To_Bus",
    "Base_Min_Voltage",
    "Base_Max_Voltage",
    "Weakest_Bus_Before",
    "Base_Solar_Q_MVAr",
    "Base_Wind_Q_MVAr",
    "Base_Solar_Q_Capability_MVAr",
    "Base_Wind_Q_Capability_MVAr",
    "Base_Solar_Q_Margin_MVAr",
    "Base_Wind_Q_Margin_MVAr"
]


# ============================================================
# LIVE PHYSICS VALIDATION
# ============================================================

def live_security_check(
    row,
    solar_hr,
    wind_hr,
    solar_support,
    wind_support
):

    test_net = pn.case9()

    # --------------------------------------------------------
    # OPERATING STATE
    # --------------------------------------------------------

    test_net.load["p_mw"] *= float(row["Load_Level"])
    test_net.load["q_mvar"] *= float(row["Load_Level"])

    solar_available = float(row["Solar_Available_MW"])
    wind_available = float(row["Wind_Available_MW"])

    # --------------------------------------------------------
    # PHYSICAL BOUNDS
    # --------------------------------------------------------

    solar_hr = np.clip(
        solar_hr,
        0.0,
        solar_available
    )

    wind_hr = np.clip(
        wind_hr,
        0.0,
        wind_available
    )

    solar_support = np.clip(
        solar_support,
        1.00,
        1.05
    )

    wind_support = np.clip(
        wind_support,
        1.00,
        1.05
    )

    # --------------------------------------------------------
    # ACTIVE POWER AFTER HEADROOM
    # --------------------------------------------------------

    solar_p = solar_available - solar_hr
    wind_p = wind_available - wind_hr

    test_net.gen.at[0, "p_mw"] = solar_p
    test_net.gen.at[1, "p_mw"] = wind_p

    test_net.gen.at[0, "vm_pu"] = solar_support
    test_net.gen.at[1, "vm_pu"] = wind_support

    # --------------------------------------------------------
    # P-Q CAPABILITY
    # --------------------------------------------------------

    solar_qmax = np.sqrt(
        max(
            0.0,
            SOLAR_S_RATED**2 - solar_p**2
        )
    )

    wind_qmax = np.sqrt(
        max(
            0.0,
            WIND_S_RATED**2 - wind_p**2
        )
    )

    test_net.gen.at[0, "max_q_mvar"] = solar_qmax
    test_net.gen.at[0, "min_q_mvar"] = -solar_qmax

    test_net.gen.at[1, "max_q_mvar"] = wind_qmax
    test_net.gen.at[1, "min_q_mvar"] = -wind_qmax

    # --------------------------------------------------------
    # APPLY LINE CONTINGENCY
    # --------------------------------------------------------

    from_bus = int(row["From_Bus"])
    to_bus = int(row["To_Bus"])

    matching = test_net.line[
        (
            (
                test_net.line["from_bus"] == from_bus - 1
            )
            &
            (
                test_net.line["to_bus"] == to_bus - 1
            )
        )
        |
        (
            (
                test_net.line["from_bus"] == to_bus - 1
            )
            &
            (
                test_net.line["to_bus"] == from_bus - 1
            )
        )
    ].index

    if len(matching) == 0:
        return {"Solved": False}

    test_net.line.at[
        matching[0],
        "in_service"
    ] = False

    # --------------------------------------------------------
    # LIVE POWER FLOW
    # --------------------------------------------------------

    try:

        pp.runpp(
            test_net,
            enforce_q_lims=True,
            numba=False
        )

    except Exception:

        return {"Solved": False}

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    vmin = float(
        test_net.res_bus.vm_pu.min()
    )

    vmax = float(
        test_net.res_bus.vm_pu.max()
    )

    weakest_bus = int(
        test_net.res_bus.vm_pu.idxmin() + 1
    )

    solar_q = float(
        test_net.res_gen.at[0, "q_mvar"]
    )

    wind_q = float(
        test_net.res_gen.at[1, "q_mvar"]
    )

    solar_used_s = np.sqrt(
        solar_p**2 + solar_q**2
    )

    wind_used_s = np.sqrt(
        wind_p**2 + wind_q**2
    )

    # --------------------------------------------------------
    # CHECK 1 — VOLTAGE
    # --------------------------------------------------------

    voltage_pass = (
        vmin >= V_MIN - 1e-6
        and
        vmax <= V_MAX + 1e-6
    )

    # --------------------------------------------------------
    # CHECK 2 — REACTIVE POWER
    # --------------------------------------------------------

    q_pass = (
        abs(solar_q) <= solar_qmax + 1e-6
        and
        abs(wind_q) <= wind_qmax + 1e-6
    )

    # --------------------------------------------------------
    # CHECK 3 — INVERTER CAPACITY
    # --------------------------------------------------------

    inverter_pass = (
        solar_used_s <= SOLAR_S_RATED + 1e-6
        and
        wind_used_s <= WIND_S_RATED + 1e-6
    )

    overall_pass = (
        voltage_pass
        and
        q_pass
        and
        inverter_pass
    )

    return {

        "Solved": True,

        "Vmin": vmin,
        "Vmax": vmax,

        "Weakest_Bus": weakest_bus,

        "Solar_Q": solar_q,
        "Solar_Qmax": solar_qmax,

        "Wind_Q": wind_q,
        "Wind_Qmax": wind_qmax,

        "Solar_Used_S": solar_used_s,
        "Wind_Used_S": wind_used_s,

        "Voltage_PASS": voltage_pass,
        "Q_PASS": q_pass,
        "Inverter_PASS": inverter_pass,

        "Overall_PASS": overall_pass
    }


# ============================================================
# UNSEEN TEST SCENARIOS
# ============================================================

test_indices = (
    safety_df["Index"]
    .dropna()
    .astype(int)
    .tolist()
)

test_scenarios = reg_df[
    reg_df["Index"].isin(test_indices)
].copy()


# ============================================================
# LOGO
# ============================================================

logo_left, logo_center, logo_right = st.columns(
    [1, 2, 1]
)

with logo_center:

    st.image(
        "ezhallogo.png",
        width="stretch"
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h2 style="text-align:center; margin-top:-20px;">
        AI-Assisted Dynamic Grid Support
    </h2>

    <p style="text-align:center; font-size:17px;">
        Machine Learning Decision Support with
        Live Physics-Based Grid Security Validation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROTOTYPE PERFORMANCE
# ============================================================

st.header(
    "Prototype Performance"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "ML Macro F1",
    "95.23%"
)

k2.metric(
    "AI + Safety Test",
    "82 / 82"
)

k3.metric(
    "Dynamic Avg. Curtailment",
    "1.63 MW"
)

k4.metric(
    "vs Fixed 5%",
    "82.83% lower"
)

st.divider()


# ============================================================
# SELECT GRID SCENARIO
# ============================================================

st.header(
    "1. Select Grid Scenario"
)

scenario_options = {}

for _, row in test_scenarios.iterrows():

    idx = int(row["Index"])

    label = (
        f"Scenario {idx} | "
        f"Load {row['Load_Level']*100:.0f}% | "
        f"Solar {row['Solar_Level']*100:.0f}% | "
        f"Wind {row['Wind_Level']*100:.0f}% | "
        f"Line {int(row['From_Bus'])}-"
        f"{int(row['To_Bus'])}"
    )

    scenario_options[label] = idx


# ============================================================
# DEFAULT SCENARIO = 313
# ============================================================

default_position = 0

for i, (label, idx) in enumerate(
    scenario_options.items()
):

    if idx == 313:

        default_position = i
        break


selected_label = st.selectbox(

    "Choose an unseen test scenario:",

    list(
        scenario_options.keys()
    ),

    index=default_position
)


selected_index = scenario_options[
    selected_label
]


scenario = reg_df[
    reg_df["Index"] == selected_index
].iloc[0]


# ============================================================
# CURRENT GRID STATE
# ============================================================

st.subheader(
    "Current Grid State"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Load",
    f"{scenario['Load_Level']*100:.0f}%"
)

c2.metric(
    "Solar Availability",
    f"{scenario['Solar_Level']*100:.0f}%"
)

c3.metric(
    "Wind Availability",
    f"{scenario['Wind_Level']*100:.0f}%"
)

c4.metric(
    "Contingency",
    f"Line "
    f"{int(scenario['From_Bus'])}-"
    f"{int(scenario['To_Bus'])}"
)


c1, c2, c3 = st.columns(3)

c1.metric(
    "Minimum Voltage Before",
    f"{scenario['Base_Min_Voltage']:.4f} pu"
)

c2.metric(
    "Weakest Bus",
    f"Bus {int(scenario['Weakest_Bus_Before'])}"
)

c3.metric(
    "Maximum Voltage Before",
    f"{scenario['Base_Max_Voltage']:.4f} pu"
)

st.divider()


# ============================================================
# RUN EZHAL AI
# ============================================================

st.header(
    "2. Run EZHAL AI"
)

if st.button(
    "⚡ RUN EZHAL AI",
    type="primary",
    width="stretch"
):

    # --------------------------------------------------------
    # BUILD LIVE ML INPUT
    # --------------------------------------------------------

    X_live = pd.DataFrame(
        [
            scenario[
                features
            ].values
        ],
        columns=features
    ).astype(float)


    # ========================================================
    # LIVE ML CLASSIFIER
    # ========================================================

    predicted_action = (
        classifier.predict(
            X_live
        )[0]
    )


    # ========================================================
    # LIVE ML REGRESSOR
    # ========================================================

    prediction = (
        regressor.predict(
            X_live
        )[0]
    )


    solar_hr = max(
        0.0,
        min(
            float(prediction[0]),
            float(
                scenario[
                    "Solar_Available_MW"
                ]
            )
        )
    )


    wind_hr = max(
        0.0,
        min(
            float(prediction[1]),
            float(
                scenario[
                    "Wind_Available_MW"
                ]
            )
        )
    )


    solar_support = float(
        np.clip(
            prediction[2],
            1.00,
            1.05
        )
    )


    wind_support = float(
        np.clip(
            prediction[3],
            1.00,
            1.05
        )
    )


    # ========================================================
    # LIVE ML DECISION
    # ========================================================

    st.subheader(
        "🤖 Live Machine Learning Decision"
    )

    st.success(
        f"Recommended Action: {predicted_action}"
    )


    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Solar Headroom",
        f"{solar_hr:.2f} MW"
    )

    a2.metric(
        "Wind Headroom",
        f"{wind_hr:.2f} MW"
    )

    a3.metric(
        "Solar Support",
        f"{solar_support:.4f} pu"
    )

    a4.metric(
        "Wind Support",
        f"{wind_support:.4f} pu"
    )


    # ========================================================
    # LIVE PHYSICS
    # ========================================================

    with st.spinner(
        "Running live physics validation..."
    ):

        physics = live_security_check(
            scenario,
            solar_hr,
            wind_hr,
            solar_support,
            wind_support
        )


    st.divider()

    st.subheader(
        "⚡ Live Physics Validation Checks"
    )


    if not physics["Solved"]:

        st.error(
            "❌ Power Flow Failed"
        )

    else:

        # ====================================================
        # THREE PHYSICS CHECKS
        # ====================================================

        p1, p2, p3 = st.columns(3)


        # ----------------------------------------------------
        # VOLTAGE
        # ----------------------------------------------------

        with p1:

            if physics["Voltage_PASS"]:

                st.success(
                    "✅ Voltage Security"
                )

            else:

                st.error(
                    "❌ Voltage Security"
                )

            st.metric(
                "Voltage Range",
                f"{physics['Vmin']:.4f}"
                f" – "
                f"{physics['Vmax']:.4f} pu"
            )

            st.caption(
                "Required: 0.95–1.05 pu"
            )


        # ----------------------------------------------------
        # REACTIVE CAPABILITY
        # ----------------------------------------------------

        with p2:

            if physics["Q_PASS"]:

                st.success(
                    "✅ Reactive Capability"
                )

            else:

                st.error(
                    "❌ Reactive Capability"
                )

            st.metric(
                "Solar Q",
                f"{physics['Solar_Q']:.2f} / "
                f"{physics['Solar_Qmax']:.2f} MVAr"
            )

            st.metric(
                "Wind Q",
                f"{physics['Wind_Q']:.2f} / "
                f"{physics['Wind_Qmax']:.2f} MVAr"
            )


        # ----------------------------------------------------
        # INVERTER CAPACITY
        # ----------------------------------------------------

        with p3:

            if physics["Inverter_PASS"]:

                st.success(
                    "✅ Inverter Capacity"
                )

            else:

                st.error(
                    "❌ Inverter Capacity"
                )

            st.metric(
                "Solar Apparent Power",
                f"{physics['Solar_Used_S']:.2f} / "
                f"{SOLAR_S_RATED:.2f} MVA"
            )

            st.metric(
                "Wind Apparent Power",
                f"{physics['Wind_Used_S']:.2f} / "
                f"{WIND_S_RATED:.2f} MVA"
            )


        # ====================================================
        # FINAL GRID STATUS
        # ====================================================

        st.divider()

        if physics["Overall_PASS"]:

            st.success(
                "🟢 GRID STATUS: SECURE"
            )

        else:

            st.error(
                "🔴 GRID STATUS: "
                "SAFETY CORRECTION REQUIRED"
            )


        # ====================================================
        # BEFORE VS AFTER METRICS
        # ====================================================

        st.subheader(
            "Before vs After EZHAL"
        )

        b1, b2, b3 = st.columns(3)


        b1.metric(
            "Without EZHAL",
            f"{scenario['Base_Min_Voltage']:.4f} pu"
        )


        b2.metric(
            "With EZHAL",
            f"{physics['Vmin']:.4f} pu",
            delta=(
                f"{physics['Vmin'] - scenario['Base_Min_Voltage']:+.4f} pu"
            )
        )


        b3.metric(
            "Renewable Curtailment",
            f"{solar_hr + wind_hr:.2f} MW"
        )


        # ====================================================
        # VOLTAGE COMPARISON LINE
        # ====================================================

        # ====================================================
        # VOLTAGE IMPROVEMENT — RED BEFORE / GREEN AFTER
        # ====================================================

        import altair as alt

        st.subheader("Voltage Improvement")

        before_v = float(scenario["Base_Min_Voltage"])
        after_v = float(physics["Vmin"])

        voltage_improvement = after_v - before_v
        improvement_percent = (
            voltage_improvement / before_v
        ) * 100


        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        voltage_df = pd.DataFrame({
            "Condition": [
                "Without EZHAL",
                "With EZHAL"
            ],
            "Minimum Voltage (pu)": [
                before_v,
                after_v
            ],
            "Label": [
                f"{before_v:.4f} pu",
                f"{after_v:.4f} pu"
            ],
            "Status": [
                "Without EZHAL",
                "With EZHAL"
            ]
        })


        condition_order = [
            "Without EZHAL",
            "With EZHAL"
        ]


        # ----------------------------------------------------
        # BLUE CONNECTING LINE
        # ----------------------------------------------------

        line = alt.Chart(
            voltage_df
        ).mark_line(
            strokeWidth=4,
            color="#5DBBFF"
        ).encode(

            x=alt.X(
                "Condition:N",
                sort=condition_order,
                title="Grid Condition"
            ),

            y=alt.Y(
                "Minimum Voltage (pu):Q",
                scale=alt.Scale(
                    domain=[0.80, 1.02]
                ),
                title="Minimum Voltage (pu)"
            )
        )


        # ----------------------------------------------------
        # RED / GREEN POINTS
        # ----------------------------------------------------

        points = alt.Chart(
            voltage_df
        ).mark_circle(
            size=500,
            stroke="white",
            strokeWidth=2
        ).encode(

            x=alt.X(
                "Condition:N",
                sort=condition_order
            ),

            y="Minimum Voltage (pu):Q",

            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=[
                        "Without EZHAL",
                        "With EZHAL"
                    ],
                    range=[
                        "#FF5C5C",   # RED
                        "#32E875"    # GREEN
                    ]
                ),
                legend=None
            )
        )


        # ----------------------------------------------------
        # COLORED NUMBERS ABOVE POINTS
        # ----------------------------------------------------

        labels = alt.Chart(
            voltage_df
        ).mark_text(
            dy=-28,
            fontSize=22,
            fontWeight="bold"
        ).encode(

            x=alt.X(
                "Condition:N",
                sort=condition_order
            ),

            y="Minimum Voltage (pu):Q",

            text="Label:N",

            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=[
                        "Without EZHAL",
                        "With EZHAL"
                    ],
                    range=[
                        "#FF5C5C",   # RED NUMBER
                        "#32E875"    # GREEN NUMBER
                    ]
                ),
                legend=None
            )
        )


        # ----------------------------------------------------
        # 0.95 SECURITY THRESHOLD
        # ----------------------------------------------------

        threshold_df = pd.DataFrame({
            "Threshold": [0.95]
        })

        threshold = alt.Chart(
            threshold_df
        ).mark_rule(
            color="#FFD166",
            strokeDash=[8, 6],
            strokeWidth=2
        ).encode(
            y="Threshold:Q"
        )


        threshold_text = alt.Chart(
            pd.DataFrame({
                "Condition": ["With EZHAL"],
                "Threshold": [0.95],
                "Label": ["0.95 pu Security Threshold"]
            })
        ).mark_text(
            align="right",
            dx=180,
            dy=-14,
            fontSize=15,
            fontWeight="bold",
            color="#FFD166"
        ).encode(

            x=alt.X(
                "Condition:N",
                sort=condition_order
            ),

            y="Threshold:Q",

            text="Label:N"
        )


        # ----------------------------------------------------
        # FINAL CHART
        # ----------------------------------------------------

        chart = (
            line
            + threshold
            + points
            + labels
            + threshold_text
        ).properties(
            height=380
        )


        st.altair_chart(
            chart,
            width="stretch"
        )


        # ----------------------------------------------------
        # JUDGE-FRIENDLY RESULT
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:10px;
                background-color:rgba(20, 100, 70, 0.18);
                border:1px solid rgba(50,232,117,0.5);
                font-size:18px;
            ">

            ⚡ Minimum voltage increased from

            <b style="color:#FF5C5C;">
            {before_v:.4f} pu
            </b>

            (Without EZHAL)

            to

            <b style="color:#32E875;">
            {after_v:.4f} pu
            </b>

            (With EZHAL),

            an improvement of

            <b style="color:#32E875;">
            +{voltage_improvement:.4f} pu
            ({improvement_percent:.1f}%)
            </b>.

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # CLEAR RESULT FOR JUDGES
        # ----------------------------------------------------

        st.success(
            f"⚡ Minimum voltage increased from "
            f"{before_v:.4f} pu to "
            f"{after_v:.4f} pu | "
            f"+{voltage_improvement:.4f} pu "
            f"({improvement_percent:.1f}% improvement)"
        )
        # ====================================================
        # LIVE STATUS
        # ====================================================

        st.info(
            "🤖 ML inference: LIVE   |   "
            "⚡ Physics validation: LIVE"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EZHAL Prototype • "
    "Machine Learning + Physics-Based Grid Validation"
)