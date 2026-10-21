
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EQUIPIX — Yard Equipment Booking System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CORPORATE CSS — Deep Navy / Teal / Gold Theme
# ============================================================
st.markdown("""
<style>
    /* ---- Global ---- */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0C2340;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #5A6F8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ---- Brand Identity ---- */
    .brand-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0C2340 0%, #007C8A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 4px;
        margin-bottom: 0;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #5A6F8A;
        text-align: center;
        margin-top: 2px;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* ---- Divider ---- */
    .corp-divider {
        height: 3px;
        background: linear-gradient(90deg, #0C2340 0%, #007C8A 50%, #00A3B5 100%);
        border: none;
        border-radius: 2px;
        margin: 0.5rem 0 1.5rem 0;
    }

    /* ---- Score / Decision Cards ---- */
    .score-win {
        background: linear-gradient(135deg, #0C2340 0%, #007C8A 100%);
        padding: 18px;
        border-radius: 8px;
        color: white;
        text-align: center;
        border: 1px solid #00A3B5;
    }
    .score-lose {
        background: linear-gradient(135deg, #8B1A1A 0%, #D63031 100%);
        padding: 18px;
        border-radius: 8px;
        color: white;
        text-align: center;
        border: 1px solid #D63031;
    }

    /* ---- Equipment Status Cards ---- */
    .status-card {
        padding: 18px 14px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .status-available {
        background: linear-gradient(135deg, #006B5A 0%, #00B27A 100%);
        color: white;
    }
    .status-booked {
        background: linear-gradient(135deg, #0C2340 0%, #007C8A 100%);
        color: white;
    }
    .status-conflict {
        background: linear-gradient(135deg, #8B1A1A 0%, #D63031 100%);
        color: white;
    }

    /* ---- KPI Metric Overrides ---- */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D6DDE6;
        border-left: 4px solid #007C8A;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(12,35,64,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #5A6F8A !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0C2340 !important;
        font-weight: 700 !important;
    }

    /* ============================================
       SIDEBAR — Corporate Dark Navy
       ============================================ */
    section[data-testid="stSidebar"] {
        background-color: #0A1628;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #8B9DC3 !important;
    }

    /* Nav buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 11px 18px;
        margin: 3px 0;
        border: none;
        border-radius: 6px;
        background-color: transparent;
        color: #8B9DC3 !important;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: pointer;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #0F2A47;
        color: #FFFFFF !important;
    }

    /* Active nav button — teal accent */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0C2340 0%, #007C8A 100%);
        color: #FFFFFF !important;
        border-left: 3px solid #00A3B5;
        font-weight: 600;
    }

    /* Sidebar info cards */
    section[data-testid="stSidebar"] .stAlert {
        background-color: #0F2A47;
        border: 1px solid #1A4B6E;
        color: #8B9DC3;
        border-radius: 6px;
    }

    /* ---- Dataframe / Table Styling ---- */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
    }

    /* ---- Buttons (main content) ---- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0C2340 0%, #007C8A 100%);
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #007C8A 0%, #00A3B5 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

DURATION_MAP = {
    'Half Day (AM)': 0.5,
    'Half Day (PM)': 0.5,
    'Full Day': 1,
    '2 Days': 2,
    '3 Days': 3,
    '5 Days': 5,
    '7 Days': 7,
    '10 Days': 10,
    '14 Days': 14,
}

PRIORITY_SCORES = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}

PHASE_SCORES = {
    'Commissioning': 5,
    'Testing': 4,
    'Assembly': 3,
    'Fabrication': 2,
    'Planning': 1,
    'Outfitting': 1,
}


@st.cache_data
def load_data():
    raw_df = pd.read_excel('Equipment_Bookings_AI_Training.xlsx', engine='openpyxl')
    raw_df = raw_df.copy()
    raw_df['Booking_Date_dt'] = pd.to_datetime(raw_df['Booking_Date'])
    raw_df['Booking_Month'] = raw_df['Booking_Date_dt'].dt.to_period('M').astype(str)
    return raw_df


def calculate_end_date(booking_date, duration):
    days = DURATION_MAP.get(duration, 1)
    if days >= 1:
        return booking_date + timedelta(days=int(days) - 1)
    return booking_date


def suggest_equipment(df, category, load_weight):
    available = df[df['Equipment_Category'] == category].copy()
    if 'Ton_Capacity' in available.columns and load_weight > 0:
        suitable = available[available['Ton_Capacity'] >= load_weight * 1.2]
        if suitable.empty:
            suitable = available[available['Ton_Capacity'] >= load_weight]
        if suitable.empty:
            suitable = available
    else:
        suitable = available
    suggested_types = sorted(suitable['Equipment_Type'].unique().tolist())
    return suitable, suggested_types


def detect_conflicts(df, new_booking):
    new_start = pd.to_datetime(new_booking['Booking_Date'])
    new_end = calculate_end_date(new_start, new_booking['Duration'])
    new_equipment = new_booking['Equipment_ID']
    new_shift = new_booking['Shift']

    same_equipment = df[
        (df['Equipment_ID'] == new_equipment) &
        (df['Shift'] == new_shift) &
        (df['Status'].isin(['Confirmed', 'Pending', 'Completed']))
    ].copy()

    if same_equipment.empty:
        return pd.DataFrame(), False

    same_equipment['End_Date_dt'] = same_equipment.apply(
        lambda row: calculate_end_date(row['Booking_Date_dt'], row['Duration']), axis=1
    )

    conflicts = same_equipment[
        (same_equipment['Booking_Date_dt'] <= new_end) &
        (same_equipment['End_Date_dt'] >= new_start)
    ]

    return conflicts, len(conflicts) > 0


# ============================================================
# AI PRIORITY SCORE SYSTEM
# ============================================================

def calculate_ai_priority_score(booking, df, equipment_info):
    score_breakdown = {}

    # 1. PROJECT CRITICALITY (1-5)
    phase = booking.get('Project_Phase', 'Fabrication')
    project_criticality = PHASE_SCORES.get(phase, 2)
    score_breakdown['Project Criticality'] = {
        'score': project_criticality, 'max': 5, 'reason': f"{phase} phase"
    }

    # 2. SCHEDULE IMPACT (1-5)
    lead_time = (pd.to_datetime(booking['Booking_Date']) - pd.to_datetime(booking['Request_Date'])).days
    if lead_time <= 1:
        schedule_impact = 5
    elif lead_time <= 3:
        schedule_impact = 4
    elif lead_time <= 5:
        schedule_impact = 3
    elif lead_time <= 9:
        schedule_impact = 2
    else:
        schedule_impact = 1
    if booking.get('Rescheduled', False):
        schedule_impact = min(5, schedule_impact + 1)
    score_breakdown['Schedule Impact'] = {
        'score': schedule_impact, 'max': 5, 'reason': f"Lead time: {lead_time} days"
    }

    # 3. URGENCY (1-5)
    priority = booking.get('Priority', 'Medium')
    urgency_map = {'Urgent': 5, 'High': 4, 'Medium': 3, 'Low': 2}
    urgency = urgency_map.get(priority, 3)
    if priority == 'Low' and lead_time > 10:
        urgency = 1
    score_breakdown['Urgency'] = {
        'score': urgency, 'max': 5, 'reason': f"{priority} priority"
    }

    # 4. EQUIPMENT SUITABILITY (1-5)
    load_weight = booking.get('Load_Weight', 0)
    equipment_capacity = 100
    if isinstance(equipment_info, pd.Series) and 'Ton_Capacity' in equipment_info.index:
        equipment_capacity = equipment_info.get('Ton_Capacity', 100)
    elif isinstance(equipment_info, dict):
        equipment_capacity = equipment_info.get('Ton_Capacity', 100)

    equipment_type = booking.get('Equipment_Type', '')
    available_units = df[df['Equipment_Type'] == equipment_type]['Equipment_ID'].nunique()

    if available_units <= 1:
        equipment_suitability = 5
    elif equipment_capacity > 0 and load_weight > 0:
        utilization_ratio = load_weight / equipment_capacity
        if utilization_ratio >= 0.8:
            equipment_suitability = 5
        elif utilization_ratio >= 0.6:
            equipment_suitability = 4
        elif utilization_ratio >= 0.4:
            equipment_suitability = 3
        elif utilization_ratio >= 0.2:
            equipment_suitability = 2
        else:
            equipment_suitability = 1
    else:
        equipment_suitability = 3

    pct = (load_weight / max(equipment_capacity, 1)) * 100
    score_breakdown['Equipment Suitability'] = {
        'score': equipment_suitability, 'max': 5,
        'reason': f"{load_weight}T on {equipment_capacity}T capacity ({pct:.0f}%)"
    }

    # 5. LOCATION EFFICIENCY (1-5)
    booking_location = booking.get('Location', '')
    equipment_id = booking.get('Equipment_ID', '')

    recent_bookings = df[
        (df['Equipment_ID'] == equipment_id) &
        (df['Booking_Date_dt'] < pd.to_datetime(booking['Booking_Date']))
    ].sort_values('Booking_Date_dt', ascending=False)

    if not recent_bookings.empty:
        last_location = recent_bookings.iloc[0]['Location']
        if last_location == booking_location:
            location_efficiency = 5
            loc_reason = f"Equipment already at {booking_location}"
        elif str(last_location).split(' ')[0] == str(booking_location).split(' ')[0]:
            location_efficiency = 4
            loc_reason = f"Equipment nearby ({last_location} → {booking_location})"
        else:
            location_efficiency = 2
            loc_reason = f"Relocation needed ({last_location} → {booking_location})"
    else:
        location_efficiency = 3
        loc_reason = "No prior location data"

    score_breakdown['Location Efficiency'] = {
        'score': location_efficiency, 'max': 5, 'reason': loc_reason
    }

    total_score = (project_criticality + schedule_impact + urgency +
                   equipment_suitability + location_efficiency)

    return total_score, score_breakdown


def resolve_conflict_ai(new_booking, conflicts, df):
    eq_rows = df[df['Equipment_ID'] == new_booking['Equipment_ID']]
    eq_info_new = eq_rows.iloc[0] if len(eq_rows) > 0 else {}
    new_score, new_breakdown = calculate_ai_priority_score(new_booking, df, eq_info_new)

    results = []
    for _, conflict in conflicts.iterrows():
        conflict_booking = {
            'Request_Date': conflict['Request_Date'],
            'Booking_Date': conflict['Booking_Date'],
            'Priority': conflict['Priority'],
            'Equipment_Type': conflict['Equipment_Type'],
            'Equipment_ID': conflict['Equipment_ID'],
            'Location': conflict['Location'],
            'Load_Weight': conflict.get('Load_Weight', 10),
            'Project_Phase': conflict.get('Project_Phase', 'Fabrication'),
            'Rescheduled': conflict.get('Rescheduled', False),
        }
        eq_rows_c = df[df['Equipment_ID'] == conflict['Equipment_ID']]
        eq_info_conflict = eq_rows_c.iloc[0] if len(eq_rows_c) > 0 else {}
        conflict_score, conflict_breakdown = calculate_ai_priority_score(conflict_booking, df, eq_info_conflict)

        if new_score > conflict_score:
            resolution = f"Your booking ({new_score}/25) beats existing ({conflict_score}/25)"
            new_wins = True
        elif new_score < conflict_score:
            resolution = f"Existing ({conflict_score}/25) beats your booking ({new_score}/25)"
            new_wins = False
        else:
            if pd.to_datetime(new_booking['Request_Date']) <= pd.to_datetime(conflict['Request_Date']):
                resolution = f"TIED ({new_score}/25) — You requested first"
                new_wins = True
            else:
                resolution = f"TIED ({conflict_score}/25) — Existing requested first"
                new_wins = False

        results.append({
            'Conflicting_Booking': conflict['Booking_ID'],
            'Your_Score': new_score,
            'Existing_Score': conflict_score,
            'Conflict_Priority': conflict['Priority'],
            'Conflict_Project': conflict.get('Project', 'N/A'),
            'Conflict_Lift_Item': conflict.get('Lift_Item', 'N/A'),
            'Resolution': resolution,
            'New_Booking_Wins': new_wins,
            'Conflict_Breakdown': conflict_breakdown,
        })

    return pd.DataFrame(results), new_score, new_breakdown


def display_score_breakdown(label, score, breakdown):
    st.markdown(f"**{label} — Total: {score}/25**")
    breakdown_data = []
    for factor, details in breakdown.items():
        bar = "█" * details['score'] + "░" * (details['max'] - details['score'])
        breakdown_data.append({
            'Factor': factor,
            'Score': f"{details['score']}/{details['max']}",
            'Bar': bar,
            'Reason': details['reason']
        })
    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)


@st.cache_resource
def train_model(df):
    df_model = df.copy()
    df_model['Priority_Score'] = df_model['Priority'].map(PRIORITY_SCORES)
    df_model['Duration_Days'] = df_model['Duration'].map(DURATION_MAP)
    df_model['Request_Date_dt'] = pd.to_datetime(df_model['Request_Date'])
    df_model['Lead_Time'] = (df_model['Booking_Date_dt'] - df_model['Request_Date_dt']).dt.days

    le_equipment_cat = LabelEncoder()
    le_equipment_type = LabelEncoder()
    le_shift = LabelEncoder()
    le_project = LabelEncoder()
    le_weather = LabelEncoder()

    df_model['Equipment_Category_Enc'] = le_equipment_cat.fit_transform(df_model['Equipment_Category'].astype(str))
    df_model['Equipment_Type_Enc'] = le_equipment_type.fit_transform(df_model['Equipment_Type'].astype(str))
    df_model['Shift_Enc'] = le_shift.fit_transform(df_model['Shift'].astype(str))
    df_model['Project_Enc'] = le_project.fit_transform(df_model['Project'].astype(str))
    df_model['Weather_Enc'] = le_weather.fit_transform(df_model['Weather_Condition'].astype(str))

    if 'Wins_Equipment' in df_model.columns:
        df_model['Target'] = (df_model['Wins_Equipment'] == 'Yes').astype(int)
    else:
        df_model['Target'] = 1

    features = ['Priority_Score', 'Duration_Days', 'Lead_Time',
                'Equipment_Category_Enc', 'Equipment_Type_Enc',
                'Shift_Enc', 'Project_Enc', 'Weather_Enc',
                'Wind_Speed_Knots', 'Crane_Utilization_Pct']

    X = df_model[features].fillna(0)
    y = df_model['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    encoders = {
        'equipment_cat': le_equipment_cat,
        'equipment_type': le_equipment_type,
        'shift': le_shift,
        'project': le_project,
        'weather': le_weather,
    }

    return model, encoders, accuracy, features, classification_report(y_test, y_pred, output_dict=True)


# ============================================================
# SIDEBAR — CORPORATE NAVIGATION
# ============================================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

with st.sidebar:
    st.markdown("## ⚙️ EQUIPIX")
    st.markdown("##### Yard Equipment Booking System")
    st.markdown("---")

    nav_items = {
        'Dashboard': '📊  Dashboard',
        'Booking Manager': '📋  Booking Manager',
        'Booking': '🔧  Booking',
        'Analytics': '📈  Analytics',
    }

    for key, label in nav_items.items():
        is_active = st.session_state.current_page == key
        if st.button(
            label,
            key=f"nav_{key}",
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            st.session_state.current_page = key
            st.rerun()

    st.markdown("---")

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.info("Please ensure 'Equipment_Bookings_AI_Training.xlsx' is in the repo root.")
    st.stop()

# Load AI model
model = None
encoders = None
accuracy = 0
features = []
report = {}
try:
    model, encoders, accuracy, features, report = train_model(df)
except Exception:
    pass

# Sidebar system info
with st.sidebar:
    st.markdown("##### System Status")
    st.markdown(f"""
    <div style="background-color:#0F2A47; padding:10px 14px; border-radius:6px; margin:5px 0; border-left:3px solid #007C8A;">
        <span style="color:#5A7A9A; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Date</span><br>
        <span style="color:#FFFFFF; font-size:13px; font-weight:600;">{datetime.now().strftime('%d %b %Y')}</span>
    </div>
    <div style="background-color:#0F2A47; padding:10px 14px; border-radius:6px; margin:5px 0; border-left:3px solid #007C8A;">
        <span style="color:#5A7A9A; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Records</span><br>
        <span style="color:#FFFFFF; font-size:13px; font-weight:600;">{len(df):,} bookings</span>
    </div>
    <div style="background-color:#0F2A47; padding:10px 14px; border-radius:6px; margin:5px 0; border-left:3px solid #007C8A;">
        <span style="color:#5A7A9A; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Equipment</span><br>
        <span style="color:#FFFFFF; font-size:13px; font-weight:600;">{df['Equipment_ID'].nunique()} units</span>
    </div>
    <div style="background-color:#0F2A47; padding:10px 14px; border-radius:6px; margin:5px 0; border-left:3px solid {'#00B27A' if model else '#D63031'};">
        <span style="color:#5A7A9A; font-size:11px; text-transform:uppercase; letter-spacing:1px;">AI Model</span><br>
        <span style="color:{'#00B27A' if model else '#D63031'}; font-size:13px; font-weight:600;">{'● Active' if model else '● Inactive'} — {accuracy*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#5A7A9A; font-size:10px; letter-spacing:1px; text-transform:uppercase;">'
        'EQUIPIX v2.0 — AI Priority Engine</div>',
        unsafe_allow_html=True
    )

page = st.session_state.current_page
today = pd.to_datetime(datetime.now().date())

# ============================================================
# PAGE: DASHBOARD — LIVE OPERATIONS HUB
# ============================================================
if page == "Dashboard":
    st.markdown('<div class="brand-header">EQUIPIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Yard Equipment Booking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="corp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">📊 Live Operations Hub</div>', unsafe_allow_html=True)
    st.markdown("")

    # ---- KPI ROW ----
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Bookings", len(df))
    with col2:
        if 'Has_Conflict' in df.columns:
            conflict_count = len(df[df['Has_Conflict'] == 'Yes'])
        else:
            conflict_count = "N/A"
        st.metric("Conflicts", conflict_count)
    with col3:
        st.metric("Equipment Units", df['Equipment_ID'].nunique())
    with col4:
        st.metric("Projects", df['Project'].nunique())
    with col5:
        completion_rate = len(df[df['Status'] == 'Completed']) / len(df) * 100
        st.metric("Completion Rate", f"{completion_rate:.1f}%")

    st.markdown("---")

    # ---- EQUIPMENT AVAILABILITY STATUS ----
    st.subheader("🟢 Equipment Availability Status")

    all_equipment_ids = df['Equipment_ID'].unique()
    booked_today = df[
        (df['Booking_Date_dt'] <= today) &
        (df['Status'].isin(['Confirmed', 'Pending']))
    ]['Equipment_ID'].unique()

    if 'Has_Conflict' in df.columns:
        conflict_equipment = df[
            (df['Has_Conflict'] == 'Yes') &
            (df['Booking_Date_dt'] <= today) &
            (df['Status'].isin(['Confirmed', 'Pending']))
        ]['Equipment_ID'].unique()
    else:
        conflict_equipment = np.array([])

    available_count = len(set(all_equipment_ids) - set(booked_today))
    booked_count = len(set(booked_today) - set(conflict_equipment))
    conflict_eq_count = len(conflict_equipment)

    eq_col1, eq_col2, eq_col3 = st.columns(3)
    with eq_col1:
        st.markdown(
            f'<div class="status-card status-available">'
            f'<h2 style="margin:0;">{available_count}</h2>'
            f'<p style="margin:4px 0 0 0; font-size:0.85rem; opacity:0.9;">Available</p></div>',
            unsafe_allow_html=True
        )
    with eq_col2:
        st.markdown(
            f'<div class="status-card status-booked">'
            f'<h2 style="margin:0;">{booked_count}</h2>'
            f'<p style="margin:4px 0 0 0; font-size:0.85rem; opacity:0.9;">Booked</p></div>',
            unsafe_allow_html=True
        )
    with eq_col3:
        st.markdown(
            f'<div class="status-card status-conflict">'
            f'<h2 style="margin:0;">{conflict_eq_count}</h2>'
            f'<p style="margin:4px 0 0 0; font-size:0.85rem; opacity:0.9;">In Conflict</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---- ACTIVE CONFLICTS + UPCOMING BOOKINGS ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⚠️ Active Conflicts Needing Resolution")
        if 'Has_Conflict' in df.columns:
            active_conflicts = df[
                (df['Has_Conflict'] == 'Yes') &
                (df['Status'].isin(['Confirmed', 'Pending']))
            ]
            if not active_conflicts.empty:
                conflict_cols = ['Booking_ID', 'Booking_Date', 'Equipment_ID',
                                 'Equipment_Type', 'Project', 'Priority']
                if 'Conflict_Resolution' in active_conflicts.columns:
                    conflict_cols.append('Conflict_Resolution')
                available_cols = [c for c in conflict_cols if c in active_conflicts.columns]
                st.dataframe(
                    active_conflicts[available_cols].sort_values('Booking_Date'),
                    use_container_width=True, height=280, hide_index=True
                )
            else:
                st.success("✅ No active conflicts — all clear!")
        else:
            st.info("Conflict data not available in dataset.")

    with col_right:
        st.subheader("📅 Upcoming Bookings (Next 7 Days)")
        upcoming = df[
            (df['Booking_Date_dt'] >= today) &
            (df['Booking_Date_dt'] <= today + timedelta(days=7)) &
            (df['Status'].isin(['Confirmed', 'Pending']))
        ].sort_values('Booking_Date_dt')

        if not upcoming.empty:
            upcoming_cols = ['Booking_ID', 'Booking_Date', 'Equipment_Type',
                             'Equipment_ID', 'Project', 'Priority', 'Status']
            available_cols = [c for c in upcoming_cols if c in upcoming.columns]
            st.dataframe(upcoming[available_cols], use_container_width=True, height=280, hide_index=True)
        else:
            st.info("📭 No bookings scheduled in the next 7 days.")

    st.markdown("---")

    # ---- RECENT AI DECISIONS LOG ----
    st.subheader("🤖 Recent AI Conflict Resolutions")
    if 'Conflict_Resolution' in df.columns and 'Has_Conflict' in df.columns:
        resolved = df[
            (df['Has_Conflict'] == 'Yes') &
            (df['Conflict_Resolution'] != 'No Conflict')
        ]
        if not resolved.empty:
            resolved_sorted = resolved.sort_values('Booking_Date', ascending=False).head(10)
            log_cols = ['Booking_ID', 'Booking_Date', 'Equipment_Type', 'Equipment_ID',
                        'Project', 'Priority', 'Conflict_Resolution']
            if 'Wins_Equipment' in resolved_sorted.columns:
                log_cols.append('Wins_Equipment')
            available_cols = [c for c in log_cols if c in resolved_sorted.columns]
            st.dataframe(resolved_sorted[available_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No resolved conflicts to display.")
    else:
        st.info("Conflict resolution data not available.")

    st.markdown("---")

    # ---- BOOKINGS BY CATEGORY ----
    st.subheader("📦 Bookings by Equipment Category")
    st.bar_chart(df['Equipment_Category'].value_counts())


# ============================================================
# PAGE: BOOKING MANAGER
# ============================================================
elif page == "Booking Manager":
    st.markdown('<div class="brand-header">EQUIPIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Yard Equipment Booking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="corp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">📋 Booking Manager</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_category = st.multiselect("Equipment Category", sorted(df['Equipment_Category'].unique()))
    with col2:
        filter_project = st.multiselect("Project", sorted(df['Project'].unique()))
    with col3:
        filter_priority = st.multiselect("Priority", ['Low', 'Medium', 'High', 'Urgent'])
    with col4:
        filter_status = st.multiselect("Status", sorted(df['Status'].unique()))

    filtered_df = df.copy()
    if filter_category:
        filtered_df = filtered_df[filtered_df['Equipment_Category'].isin(filter_category)]
    if filter_project:
        filtered_df = filtered_df[filtered_df['Project'].isin(filter_project)]
    if filter_priority:
        filtered_df = filtered_df[filtered_df['Priority'].isin(filter_priority)]
    if filter_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(filter_status)]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} bookings**")

    display_cols = ['Booking_ID', 'Booking_Date', 'Equipment_Category', 'Equipment_Type',
                    'Equipment_ID', 'Location', 'Project', 'Lift_Item', 'Duration',
                    'Priority', 'Status']
    if 'Has_Conflict' in filtered_df.columns:
        display_cols.extend(['Has_Conflict', 'Conflict_Resolution', 'Wins_Equipment'])

    available_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[available_cols], use_container_width=True, height=500)


# ============================================================
# PAGE: BOOKING
# ============================================================
elif page == "Booking":
    st.markdown('<div class="brand-header">EQUIPIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Yard Equipment Booking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="corp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">🔧 Equipment Booking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered booking with 5-factor priority scoring</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Booking Details")

        book_request_date = st.date_input("Request Date", datetime.now(), key='book_req')
        book_booking_date = st.date_input("Booking Date", datetime.now() + timedelta(days=3), key='book_date')

        book_project = st.selectbox("Project", sorted(df['Project'].unique()), key='book_proj')
        phases = list(PHASE_SCORES.keys())
        book_phase = st.selectbox("Project Phase", phases, key='book_phase')

        book_lift_item = st.text_input("Lift Item / Description", "Steel Block Section", key='book_lift')
        book_load_weight = st.number_input("Load Weight (Tonnes)", min_value=0.1, max_value=500.0, value=10.0, step=0.5, key='book_weight')

        book_category = st.selectbox("Equipment Category", sorted(df['Equipment_Category'].unique()), key='book_cat')

        suitable_df, suggested_types = suggest_equipment(df, book_category, book_load_weight)

        if suggested_types:
            st.markdown("💡 *AI Suggested based on load weight:*")
            book_type = st.selectbox("Equipment Type (AI Suggested)", suggested_types, key='book_type')

            suggested_units = sorted(suitable_df[suitable_df['Equipment_Type'] == book_type]['Equipment_ID'].unique().tolist())
            if suggested_units:
                unit_capacities = suitable_df[suitable_df['Equipment_Type'] == book_type][['Equipment_ID', 'Ton_Capacity']].drop_duplicates()
                unit_options = []
                for _, row in unit_capacities.iterrows():
                    cap = f" ({row['Ton_Capacity']}T)" if pd.notna(row['Ton_Capacity']) else ""
                    unit_options.append(f"{row['Equipment_ID']}{cap}")
                book_unit_display = st.selectbox("Equipment Unit (AI Suggested)", unit_options, key='book_unit')
                book_id = book_unit_display.split(" (")[0]
            else:
                book_id = st.selectbox("Equipment Unit", sorted(df[df['Equipment_Type'] == book_type]['Equipment_ID'].unique()), key='book_unit2')
        else:
            st.warning("⚠️ No suitable equipment for this weight. Showing all.")
            all_types = sorted(df[df['Equipment_Category'] == book_category]['Equipment_Type'].unique())
            book_type = st.selectbox("Equipment Type", all_types, key='book_type_all')
            book_id = st.selectbox("Equipment Unit", sorted(df[df['Equipment_Type'] == book_type]['Equipment_ID'].unique()), key='book_unit_all')

        book_duration = st.selectbox("Duration", list(DURATION_MAP.keys()), key='book_dur')
        book_priority = st.selectbox("Priority", ['Low', 'Medium', 'High', 'Urgent'], key='book_pri')
        book_locations = sorted(df['Location'].unique().tolist())
        book_location = st.selectbox("Location", book_locations, key='book_loc')

    with col2:
        st.subheader("🤖 AI Evaluation")

        if st.button("📋 Submit Booking Request", type="primary", use_container_width=True):
            new_booking = {
                'Request_Date': book_request_date,
                'Booking_Date': book_booking_date,
                'Equipment_Category': book_category,
                'Equipment_Type': book_type,
                'Equipment_ID': book_id,
                'Duration': book_duration,
                'Shift': 'Day Shift (0700-1900)',
                'Priority': book_priority,
                'Project': book_project,
                'Project_Phase': book_phase,
                'Location': book_location,
                'Lift_Item': book_lift_item,
                'Load_Weight': book_load_weight,
                'Rescheduled': False,
                'Status': 'Confirmed',
            }

            lead_time = (pd.to_datetime(book_booking_date) - pd.to_datetime(book_request_date)).days

            st.markdown("### 📋 Booking Summary")
            summary = {
                'Field': ['Project', 'Phase', 'Lift Item', 'Load Weight', 'Equipment', 'Unit',
                          'Date', 'Duration', 'Priority', 'Location', 'Lead Time'],
                'Value': [book_project, book_phase, book_lift_item, f"{book_load_weight} T",
                          book_type, book_id, str(book_booking_date), book_duration,
                          book_priority, book_location, f"{lead_time} days"]
            }
            st.table(pd.DataFrame(summary))

            st.markdown("---")

            conflicts, has_conflict = detect_conflicts(df, new_booking)

            if has_conflict:
                st.error(f"⚠️ **CONFLICT DETECTED** — {len(conflicts)} existing booking(s) overlap.")

                st.markdown("### 📌 Conflicting Bookings")
                conflict_display_cols = ['Booking_ID', 'Booking_Date', 'Duration', 'Priority', 'Project', 'Lift_Item']
                available_display = [c for c in conflict_display_cols if c in conflicts.columns]
                st.dataframe(conflicts[available_display], use_container_width=True)

                st.markdown("---")
                st.markdown("### 🤖 AI Priority Score Evaluation")
                st.caption("Score = Project Criticality + Schedule Impact + Urgency + Equipment Suitability + Location Efficiency")

                resolution_df, new_score, new_breakdown = resolve_conflict_ai(new_booking, conflicts, df)

                st.markdown("---")
                display_score_breakdown("📗 Your Booking", new_score, new_breakdown)

                for _, res in resolution_df.iterrows():
                    st.markdown("---")
                    display_score_breakdown(
                        f"📕 Existing: {res['Conflicting_Booking']}",
                        res['Existing_Score'],
                        res['Conflict_Breakdown']
                    )

                st.markdown("---")
                st.markdown("### 🏆 AI Decision")

                for _, res in resolution_df.iterrows():
                    if res['New_Booking_Wins']:
                        st.markdown(
                            f'<div class="score-win">'
                            f'<h3>✅ YOUR BOOKING WINS</h3>'
                            f'<p>Score: {res["Your_Score"]}/25 vs {res["Existing_Score"]}/25</p>'
                            f'<p>{res["Resolution"]}</p>'
                            f'</div>', unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="score-lose">'
                            f'<h3>❌ EXISTING BOOKING WINS</h3>'
                            f'<p>Score: {res["Existing_Score"]}/25 vs {res["Your_Score"]}/25</p>'
                            f'<p>{res["Resolution"]}</p>'
                            f'</div>', unsafe_allow_html=True
                        )

                st.markdown("---")
                st.markdown("### 🔐 Manager Override")
                st.markdown("Override the AI decision with Manager's authority.")

                mgr_col1, mgr_col2 = st.columns(2)
                with mgr_col1:
                    manager_name = st.text_input("Manager Name", key='mgr_name')
                    manager_reason = st.text_area("Override Reason", placeholder="e.g., Client deadline, safety critical...", key='mgr_reason')

                with mgr_col2:
                    st.markdown("")
                    st.markdown("")
                    if st.button("✅ Manager Override — Approve", type="primary", use_container_width=True):
                        if manager_name and manager_reason:
                            st.success("✅ **MANAGER OVERRIDE APPROVED**")
                            st.markdown(f"**Approved by:** {manager_name}")
                            st.markdown(f"**Reason:** {manager_reason}")
                            st.balloons()
                        else:
                            st.error("Please enter Manager Name and Reason.")

                    if st.button("❌ Accept AI Decision", use_container_width=True):
                        all_wins = resolution_df['New_Booking_Wins'].all()
                        if all_wins:
                            st.success("✅ **BOOKING CONFIRMED**")
                        else:
                            st.info("📋 **BOOKING QUEUED** — You'll be notified when available.")
                            st.markdown("**💡 Try:** Different date, different unit, or higher priority")

            else:
                st.success("✅ **BOOKING APPROVED** — No conflicts!")
                st.markdown(f"Equipment **{book_id}** ({book_type}) is available for **{book_booking_date}**.")

                eq_rows = df[df['Equipment_ID'] == book_id]
                eq_info = eq_rows.iloc[0] if len(eq_rows) > 0 else {}
                score, breakdown = calculate_ai_priority_score(new_booking, df, eq_info)

                st.markdown("---")
                display_score_breakdown("📗 Your Booking Score", score, breakdown)
                st.balloons()


# ============================================================
# PAGE: ANALYTICS — HISTORICAL INSIGHTS & TRENDS
# ============================================================
elif page == "Analytics":
    st.markdown('<div class="brand-header">EQUIPIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Yard Equipment Booking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="corp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">📈 Historical Insights & Trends</div>', unsafe_allow_html=True)
    st.markdown("")

    # ---- UTILIZATION TRENDS OVER TIME ----
    if 'Crane_Utilization_Pct' in df.columns:
        st.subheader("📉 Equipment Utilization Trend Over Time")
        monthly_util = df.groupby('Booking_Month')['Crane_Utilization_Pct'].mean()
        st.line_chart(monthly_util)

    st.markdown("---")

    # ---- CONFLICT RATE TRENDS ----
    if 'Has_Conflict' in df.columns:
        st.subheader("⚠️ Monthly Conflict Rate Trend")
        monthly_conflicts = df[df['Has_Conflict'] == 'Yes'].groupby('Booking_Month').size()
        monthly_total = df.groupby('Booking_Month').size()
        conflict_rate = (monthly_conflicts / monthly_total * 100).fillna(0)
        st.line_chart(conflict_rate)
        st.caption("Conflict rate = (Conflicts / Total Bookings) × 100 per month")

    st.markdown("---")

    # ---- TOP CONFLICTING EQUIPMENT ----
    if 'Has_Conflict' in df.columns:
        st.subheader("🔴 Top Conflicting Equipment")
        top_conflict_eq = df[df['Has_Conflict'] == 'Yes'].groupby(
            ['Equipment_ID', 'Equipment_Type']
        ).size().reset_index(name='Conflict_Count').sort_values('Conflict_Count', ascending=False).head(10)

        if not top_conflict_eq.empty:
            st.dataframe(top_conflict_eq, use_container_width=True, hide_index=True)
        else:
            st.success("No conflicts recorded.")

    st.markdown("---")

    # ---- PRIORITY DISTRIBUTION BY PROJECT ----
    st.subheader("📊 Priority Distribution by Project")
    priority_project = pd.crosstab(df['Project'], df['Priority'])
    st.dataframe(priority_project, use_container_width=True)

    st.markdown("---")

    # ---- WEATHER IMPACT ON OPERATIONS ----
    if 'Weather_Condition' in df.columns:
        st.subheader("🌦️ Weather Impact on Operations")
        weather_status = pd.crosstab(df['Weather_Condition'], df['Status'])
        st.dataframe(weather_status, use_container_width=True)

        if 'Wind_Speed_Knots' in df.columns:
            st.markdown("")
            st.markdown("**Average Wind Speed by Weather Condition**")
            wind_by_weather = df.groupby('Weather_Condition')['Wind_Speed_Knots'].mean().sort_values(ascending=False)
            st.bar_chart(wind_by_weather)

    st.markdown("---")

    # ---- BOOKING DURATION DISTRIBUTION ----
    st.subheader("⏱️ Booking Duration Distribution")
    st.bar_chart(df['Duration'].value_counts())

    st.markdown("---")

    # ---- AI MODEL PERFORMANCE ----
    st.subheader("🤖 AI Model Performance")
    if model is not None and features:
        ai_col1, ai_col2 = st.columns(2)
        with ai_col1:
            st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
            st.metric("Model Type", "Gradient Boosting Classifier")
            st.metric("Training Records", f"{len(df):,}")

        with ai_col2:
            st.markdown("**Feature Importance**")
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
    else:
        st.warning("AI model not available.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align:center; color:#5A6F8A; font-size:12px; padding:8px 0;">'
    f'⚙️ EQUIPIX v2.0 — Yard Equipment Booking System &nbsp;|&nbsp; AI Priority Engine &nbsp;|&nbsp; {datetime.now().strftime("%Y")}'
    f'</div>',
    unsafe_allow_html=True
)

