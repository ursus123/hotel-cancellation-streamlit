import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "hotel_cancellation_tuned_gbm.pkl"
COLUMNS_PATH = "model_columns_tuned_gbm.pkl"
DATA_PATH = "hotel_bookings.csv"

HIGH_SEASON_MONTHS = ["June", "July", "August", "September"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


st.set_page_config(
    page_title="Hotel Cancellation Risk",
    layout="wide",
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, model_columns


def cap_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df["children"] = df["children"].fillna(df["children"].mean())
    df = df.drop_duplicates()
    df = df.dropna(subset=["country"])

    cols_to_winsorize = [
        "lead_time",
        "adr",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "days_in_waiting_list",
        "previous_cancellations",
        "booking_changes",
    ]

    for col in cols_to_winsorize:
        df[col] = cap_outliers(df[col])

    adr_quantiles = df["adr"].quantile([0.25, 0.5, 0.75]).to_dict()

    options = {
        "hotel": sorted(df["hotel"].dropna().unique().tolist()),
        "country": sorted(df["country"].dropna().unique().tolist()),
        "meal": sorted(df["meal"].dropna().unique().tolist()),
        "market_segment": sorted(df["market_segment"].dropna().unique().tolist()),
        "distribution_channel": sorted(df["distribution_channel"].dropna().unique().tolist()),
        "reserved_room_type": sorted(df["reserved_room_type"].dropna().unique().tolist()),
        "assigned_room_type": sorted(df["assigned_room_type"].dropna().unique().tolist()),
        "deposit_type": sorted(df["deposit_type"].dropna().unique().tolist()),
        "customer_type": sorted(df["customer_type"].dropna().unique().tolist()),
    }

    summary = {
        "bookings": len(df),
        "cancellation_rate": float(df["is_canceled"].mean()),
        "adr_quantiles": adr_quantiles,
        "median_adr": float(df["adr"].median()),
        "median_lead_time": int(df["lead_time"].median()),
    }

    return options, summary


def adr_to_bin(adr, quantiles):
    if adr <= quantiles[0.25]:
        return "Low"
    if adr <= quantiles[0.5]:
        return "Medium-Low"
    if adr <= quantiles[0.75]:
        return "Medium-High"
    return "High"


def lead_time_to_group(lead_time):
    if lead_time <= 7:
        return "0-7 days"
    if lead_time <= 30:
        return "8-30 days"
    if lead_time <= 90:
        return "31-90 days"
    if lead_time <= 180:
        return "91-180 days"
    if lead_time <= 365:
        return "181-365 days"
    return "365+ days"


def build_model_row(values, model_columns, adr_quantiles):
    total_nights = values["stays_in_weekend_nights"] + values["stays_in_week_nights"]
    total_guests = values["adults"] + values["children"] + values["babies"]
    room_match = int(values["reserved_room_type"] == values["assigned_room_type"])
    has_special_requests = int(values["total_of_special_requests"] > 0)
    guest_loyalty = values["is_repeated_guest"] + values["previous_bookings_not_canceled"]
    weekend_ratio = values["stays_in_weekend_nights"] / total_nights if total_nights > 0 else 0

    row = {
        **values,
        "total_nights": total_nights,
        "total_guests": total_guests,
        "room_match": room_match,
        "is_high_season": int(values["arrival_date_month"] in HIGH_SEASON_MONTHS),
        "cancel_risk_score": (
            values["lead_time"]
            + values["previous_cancellations"] * 10
            - values["booking_changes"]
        ),
        "adr_bin": adr_to_bin(values["adr"], adr_quantiles),
        "has_special_requests": has_special_requests,
        "guest_loyalty": guest_loyalty,
        "weekend_ratio": weekend_ratio,
        "booked_via_agent": values.pop("booked_via_agent"),
        "lead_time_group": lead_time_to_group(values["lead_time"]),
    }

    input_df = pd.DataFrame([row])
    encoded = pd.get_dummies(input_df, drop_first=True)
    encoded = encoded.reindex(columns=model_columns, fill_value=0)
    return encoded, row


def risk_label(probability):
    if probability >= 0.70:
        return "High", "#b42318"
    if probability >= 0.40:
        return "Medium", "#b54708"
    return "Low", "#067647"


def risk_actions(label):
    if label == "High":
        return [
            "Confirm the reservation closer to arrival.",
            "Review deposit or prepayment policy.",
            "Avoid relying on the booking for occupancy planning without backup demand.",
        ]
    if label == "Medium":
        return [
            "Monitor the booking as arrival approaches.",
            "Send a reminder or confirmation message.",
            "Check whether added services could increase commitment.",
        ]
    return [
        "Treat as a relatively stable booking.",
        "Maintain normal communication.",
        "Use standard booking follow-up.",
    ]


def format_percent(value):
    return f"{value * 100:.1f}%"


model, model_columns = load_model_assets()
options, summary = load_reference_data()

st.title("Hotel Cancellation Risk")

top_metrics = st.columns(3)
top_metrics[0].metric("Reference Bookings", f"{summary['bookings']:,}")
top_metrics[1].metric("Dataset Cancellation Rate", format_percent(summary["cancellation_rate"]))
top_metrics[2].metric("Model", "Tuned GBM")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Booking Details")

    with st.form("booking_form"):
        c1, c2, c3 = st.columns(3)
        hotel = c1.selectbox("Hotel", options["hotel"])
        country_default = options["country"].index("PRT") if "PRT" in options["country"] else 0
        country = c2.selectbox("Guest Country", options["country"], index=country_default)
        arrival_date_year = c3.number_input("Arrival Year", 2015, 2030, 2017)

        c1, c2, c3 = st.columns(3)
        arrival_date_month = c1.selectbox("Arrival Month", MONTHS, index=7)
        arrival_date_week_number = c2.number_input("Arrival Week Number", 1, 53, 32)
        arrival_date_day_of_month = c3.number_input("Arrival Day", 1, 31, 15)

        c1, c2, c3 = st.columns(3)
        lead_time = c1.number_input("Lead Time", 0, 737, summary["median_lead_time"])
        stays_in_weekend_nights = c2.number_input("Weekend Nights", 0, 20, 1)
        stays_in_week_nights = c3.number_input("Week Nights", 0, 50, 2)

        c1, c2, c3 = st.columns(3)
        adults = c1.number_input("Adults", 0, 10, 2)
        children = c2.number_input("Children", 0.0, 10.0, 0.0, step=1.0)
        babies = c3.number_input("Babies", 0, 10, 0)

        c1, c2, c3 = st.columns(3)
        meal = c1.selectbox("Meal", options["meal"])
        market_segment_default = (
            options["market_segment"].index("Online TA")
            if "Online TA" in options["market_segment"]
            else 0
        )
        market_segment = c2.selectbox(
            "Market Segment",
            options["market_segment"],
            index=market_segment_default,
        )
        distribution_channel = c3.selectbox(
            "Distribution Channel",
            options["distribution_channel"],
        )

        c1, c2, c3 = st.columns(3)
        reserved_room_type = c1.selectbox("Reserved Room Type", options["reserved_room_type"])
        assigned_room_type = c2.selectbox("Assigned Room Type", options["assigned_room_type"])
        deposit_type = c3.selectbox("Deposit Type", options["deposit_type"])

        c1, c2, c3 = st.columns(3)
        customer_type = c1.selectbox("Customer Type", options["customer_type"])
        adr = c2.number_input("ADR", 0.0, 1000.0, summary["median_adr"], step=5.0)
        days_in_waiting_list = c3.number_input("Days in Waiting List", 0, 400, 0)

        c1, c2, c3 = st.columns(3)
        previous_cancellations = c1.number_input("Previous Cancellations", 0, 30, 0)
        previous_bookings_not_canceled = c2.number_input(
            "Previous Non-Canceled Bookings",
            0,
            100,
            0,
        )
        booking_changes = c3.number_input("Booking Changes", 0, 30, 0)

        c1, c2, c3 = st.columns(3)
        required_car_parking_spaces = c1.number_input("Parking Spaces", 0, 8, 0)
        total_of_special_requests = c2.number_input("Special Requests", 0, 5, 0)
        is_repeated_guest = c3.toggle("Repeated Guest", value=False)

        booked_via_agent = st.toggle("Booked via Agent", value=False)
        submitted = st.form_submit_button("Predict Cancellation Risk", type="primary")

with right:
    st.subheader("Prediction")

    if submitted:
        values = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_date_year,
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": arrival_date_week_number,
            "arrival_date_day_of_month": arrival_date_day_of_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": int(is_repeated_guest),
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "assigned_room_type": assigned_room_type,
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "days_in_waiting_list": days_in_waiting_list,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "booked_via_agent": int(booked_via_agent),
        }

        model_row, engineered_row = build_model_row(
            values.copy(),
            model_columns,
            summary["adr_quantiles"],
        )
        probability = float(model.predict_proba(model_row)[0, 1])
        label, color = risk_label(probability)
        prediction = "Canceled" if probability >= 0.5 else "Not Canceled"

        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 1rem 1.2rem; background: #f8fafc;">
                <div style="font-size: 0.9rem; color: #475467;">Cancellation Probability</div>
                <div style="font-size: 2.4rem; font-weight: 700; color: {color};">{probability * 100:.1f}%</div>
                <div style="font-size: 1.05rem; color: #101828;">Risk Level: <strong>{label}</strong></div>
                <div style="font-size: 1.05rem; color: #101828;">Predicted Outcome: <strong>{prediction}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(probability, 0), 1))

        st.subheader("Operational Actions")
        for action in risk_actions(label):
            st.write(f"- {action}")

        st.subheader("Signals Used")
        signal_cols = st.columns(2)
        signal_cols[0].metric("Total Nights", engineered_row["total_nights"])
        signal_cols[1].metric("Total Guests", engineered_row["total_guests"])
        signal_cols[0].metric("Room Match", "Yes" if engineered_row["room_match"] else "No")
        signal_cols[1].metric("High Season", "Yes" if engineered_row["is_high_season"] else "No")
        signal_cols[0].metric("ADR Tier", engineered_row["adr_bin"])
        signal_cols[1].metric("Weekend Ratio", f"{engineered_row['weekend_ratio']:.2f}")
    else:
        st.info("Enter booking details and run the prediction.")

st.divider()

importance = pd.DataFrame(
    {
        "Feature": [
            "country_PRT",
            "market_segment_Online TA",
            "required_car_parking_spaces",
            "arrival_date_week_number",
            "arrival_date_year",
            "lead_time",
            "room_match",
            "cancel_risk_score",
            "has_special_requests",
            "adr",
        ],
        "Importance": [
            0.152255,
            0.119848,
            0.112654,
            0.094666,
            0.089310,
            0.084463,
            0.082508,
            0.073948,
            0.037077,
            0.030282,
        ],
    }
)

st.subheader("Top Tuned GBM Features")
st.bar_chart(importance.set_index("Feature"))
