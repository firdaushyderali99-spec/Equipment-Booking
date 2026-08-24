
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
    page_title="Equipment Booking AI - Conflict Resolution",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
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


@st.cache_data
def load_data():
    """Load the equipment bookings dataset."""
    df = pd.read_excel('Equipment_Bookings_AI_Training.xlsx', engine='openpyxl')
    return df


def calculate_end_date(booking_date, duration):
    """Calculate end date based on booking date and duration."""
    days = DURATION_MAP.get(duration, 1)
    if days >= 1:
        return booking_date + timedelta(days=int(days) - 1)
    return booking_date


def suggest_equipment(df, category, load_weight):
    """Suggest equipment type and unit based on category and load weight."""
    available = df[df['Equipment_Category'] == category].copy()

    if 'Ton_Capacity' in available.columns and load_weight > 0:
        # Filter equipment that can handle the weight (with 20% safety margin)
        suitable = available[available['Ton_Capacity'] >= load_weight * 1.2]
        if suitable.empty:
            suitable = available[available['Ton_Capacity'] >= load_weight]
        if suitable.empty:
            suitable = available
    else:
        suitable = available

    # Get unique types and units
    suggested_types = sorted(suitable['Equipment_Type'].unique().tolist())
    return suitable, suggested_types


def detect_conflicts(df, new_booking):
    """Detect conflicts between a new booking and existing bookings."""
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

    same_equipment['Booking_Date_dt'] = pd.to_datetime(same_equipment['Booking_Date'])
    same_equipment['End_Date_dt'] = same_equipment.apply(
        lambda row: calculate_end_date(row['Booking_Date_dt'], row['Duration']), axis=1
    )

    conflicts = same_equipment[
        (same_equipment['Booking_Date_dt'] <= new_end) &
        (same_equipment['End_Date_dt'] >= new_start)
    ]

    return conflicts, len(conflicts) > 0


def resolve_conflict(new_booking, conflicts):
    """Resolve conflict based on priority and FCFS rules."""
    new_priority = PRIORITY_SCORES.get(new_booking['Priority'], 2)
    new_request_date = pd.to_datetime(new_booking['Request_Date'])

    results = []

    for _, conflict in conflicts.iterrows():
        conflict_priority = PRIORITY_SCORES.get(conflict['Priority'], 2)
        conflict_request_date = pd.to_datetime(conflict['Request_Date'])

        if new_priority > conflict_priority:
            resolution = "Priority Override - New booking WINS"
            new_wins = True
        elif new_priority < conflict_priority:
            resolution = "Yielded to Higher Priority - Existing booking WINS"
            new_wins = False
        else:
            if new_request_date <= conflict_request_date:
                resolution = "First-Come-First-Served - New booking WINS"
                new_wins = True
            else:
                resolution = "First-Come-First-Served - Existing booking WINS"
                new_wins = False

        results.append({
            'Conflicting_Booking': conflict['Booking_ID'],
            'Conflict_Equipment': conflict['Equipment_ID'],
            'Conflict_Priority': conflict['Priority'],
            'Conflict_Project': conflict.get('Project', 'N/A'),
            'Conflict_Lift_Item': conflict.get('Lift_Item', 'N/A'),
            'Conflict_Request_Date': conflict['Request_Date'],
            'Conflict_Booking_Date': conflict['Booking_Date'],
            'Conflict_Duration': conflict['Duration'],
            'Resolution': resolution,
            'New_Booking_Wins': new_wins
        })

    return pd.DataFrame(results)


@st.cache_resource
def train_model(df):
    """Train ML model for conflict resolution prediction."""
    df_model = df.copy()
    df_model['Priority_Score'] = df_model['Priority'].map(PRIORITY_SCORES)
    df_model['Duration_Days'] = df_model['Duration'].map(DURATION_MAP)
    df_model['Request_Date_dt'] = pd.to_datetime(df_model['Request_Date'])
    df_model['Booking_Date_dt'] = pd.to_datetime(df_model['Booking_Date'])
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
# MAIN APP
# ============================================================

st.markdown('<div class="main-header">🏗️ Equipment Booking AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Conflict Detection & Resolution System for Shipyard Operations</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏗️ Navigation")
    st.markdown("---")

    page = st.radio(
        "Select Module",
        ["📊 Dashboard", "📋 Booking Manager", "🤖 Booking", "📈 Analytics"],
        index=0
    )

    st.markdown("---")
    st.markdown("### System Info")
    st.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    st.info("🔄 Model: Gradient Boosting")

# Load data
try:
    df = load_data()
    st.sidebar.success(f"📦 Loaded: {len(df)} bookings")
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.info("Please ensure 'Equipment_Bookings_AI_Training.xlsx' is in the repo root.")
    st.stop()

# Load AI model
with st.sidebar:
    try:
        model, encoders, accuracy, features, report = train_model(df)
        st.success(f"🤖 AI Model: {accuracy*100:.1f}% accuracy")
    except Exception as e:
        model = None
        st.warning("⚠️ AI model unavailable")

# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.header("📊 Operations Dashboard")

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bookings by Equipment Category")
        st.bar_chart(df['Equipment_Category'].value_counts())

    with col2:
        st.subheader("Bookings by Priority")
        st.bar_chart(df['Priority'].value_counts())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Status Distribution")
        st.bar_chart(df['Status'].value_counts())

    with col2:
        st.subheader("Bookings by Project")
        st.bar_chart(df['Project'].value_counts())

    if 'Conflict_Resolution' in df.columns:
        st.markdown("---")
        st.subheader("⚠️ Conflict Resolution Summary")
        resolution_counts = df[df['Conflict_Resolution'] != 'No Conflict']['Conflict_Resolution'].value_counts()
        if not resolution_counts.empty:
            st.dataframe(resolution_counts.reset_index().rename(
                columns={'Conflict_Resolution': 'Resolution Method', 'count': 'Count'}
            ), use_container_width=True)


# ============================================================
# PAGE: BOOKING MANAGER
# ============================================================
elif page == "📋 Booking Manager":
    st.header("📋 Booking Manager")

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
# PAGE: BOOKING (UNIFIED)
# ============================================================
elif page == "🤖 Booking":
    st.header("🤖 Equipment Booking")
    st.markdown("Submit a booking request. AI will suggest equipment, detect conflicts, and help resolve them.")

    st.markdown("---")

    # --- BOOKING FORM ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Booking Details")

        book_request_date = st.date_input("Request Date", datetime.now(), key='book_req')
        book_booking_date = st.date_input("Booking Date", datetime.now() + timedelta(days=3), key='book_date')

        book_project = st.selectbox("Project", sorted(df['Project'].unique()), key='book_proj')
        book_lift_item = st.text_input("Lift Item / Description", "Steel Block Section", key='book_lift')
        book_load_weight = st.number_input("Load Weight (Tonnes)", min_value=0.1, max_value=500.0, value=10.0, step=0.5, key='book_weight')

        book_category = st.selectbox("Equipment Category", sorted(df['Equipment_Category'].unique()), key='book_cat')

        # AI suggests equipment based on category and weight
        suitable_df, suggested_types = suggest_equipment(df, book_category, book_load_weight)

        if suggested_types:
            st.markdown("💡 *AI Suggested equipment based on your load weight:*")
            book_type = st.selectbox("Equipment Type (AI Suggested)", suggested_types, key='book_type')

            # Suggest unit based on type
            suggested_units = sorted(suitable_df[suitable_df['Equipment_Type'] == book_type]['Equipment_ID'].unique().tolist())
            if suggested_units:
                # Show capacity info
                unit_capacities = suitable_df[suitable_df['Equipment_Type'] == book_type][['Equipment_ID', 'Ton_Capacity']].drop_duplicates()
                unit_options = []
                for _, row in unit_capacities.iterrows():
                    cap = f" ({row['Ton_Capacity']}T)" if pd.notna(row['Ton_Capacity']) else ""
                    unit_options.append(f"{row['Equipment_ID']}{cap}")

                book_unit_display = st.selectbox("Equipment Unit (AI Suggested)", unit_options, key='book_unit')
                book_id = book_unit_display.split(" (")[0]  # Extract ID without capacity
            else:
                book_id = st.selectbox("Equipment Unit", sorted(df[df['Equipment_Type'] == book_type]['Equipment_ID'].unique()), key='book_unit2')
        else:
            st.warning("⚠️ No suitable equipment found for this weight. Showing all options.")
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
            # Build booking object
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
                'Location': book_location,
                'Lift_Item': book_lift_item,
                'Load_Weight': book_load_weight,
                'Status': 'Confirmed',
            }

            lead_time = (pd.to_datetime(book_booking_date) - pd.to_datetime(book_request_date)).days

            # Booking Summary
            st.markdown("### 📋 Booking Summary")
            summary = {
                'Field': ['Project', 'Lift Item', 'Load Weight', 'Equipment', 'Unit ID', 'Booking Date', 'Duration', 'Priority', 'Location', 'Lead Time'],
                'Value': [book_project, book_lift_item, f"{book_load_weight} T", book_type, book_id, str(book_booking_date), book_duration, book_priority, book_location, f"{lead_time} days"]
            }
            st.table(pd.DataFrame(summary))

            st.markdown("---")

            # Check for conflicts
            conflicts, has_conflict = detect_conflicts(df, new_booking)

            if has_conflict:
                st.error(f"⚠️ **CONFLICT DETECTED** — {len(conflicts)} existing booking(s) overlap with your request.")

                # Show conflicting bookings
                st.markdown("### 📌 Existing Bookings in Conflict")
                conflict_display = conflicts[['Booking_ID', 'Booking_Date', 'Duration', 'Priority', 'Project', 'Lift_Item']].copy()
                conflict_display.columns = ['Booking ID', 'Date', 'Duration', 'Priority', 'Project', 'Lift Item']
                st.dataframe(conflict_display, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🤖 AI Evaluation")

                # AI evaluates priority
                resolution_df = resolve_conflict(new_booking, conflicts)

                for _, res in resolution_df.iterrows():
                    if res['New_Booking_Wins']:
                        st.success(f"✅ **AI Recommends: YOUR BOOKING WINS**")
                        st.markdown(f"**Reason:** {res['Resolution']}")
                        st.markdown(f"Your booking ({book_priority} priority) overrides "
                                    f"**{res['Conflicting_Booking']}** ({res['Conflict_Priority']} priority)")
                    else:
                        st.warning(f"⚠️ **AI Recommends: EXISTING BOOKING HAS PRIORITY**")
                        st.markdown(f"**Reason:** {res['Resolution']}")
                        st.markdown(f"Existing booking **{res['Conflicting_Booking']}** "
                                    f"({res['Conflict_Priority']} priority, {res['Conflict_Project']}) "
                                    f"takes precedence over your request ({book_priority} priority).")

                st.markdown("---")
                st.markdown("### 🔐 Manager Override")
                st.markdown("If this booking is critical, a Manager can override the AI decision.")

                # Store conflict state in session
                st.session_state['has_pending_conflict'] = True
                st.session_state['conflict_booking'] = new_booking
                st.session_state['conflict_details'] = resolution_df

                override_col1, override_col2 = st.columns(2)

                with override_col1:
                    if st.button("✅ Override - Approve My Booking", type="primary", use_container_width=True):
                        st.success("✅ **MANAGER OVERRIDE APPROVED**")
                        st.markdown("Your booking has been **approved** with Manager's authority.")
                        st.markdown(f"**{conflicts.iloc[0]['Booking_ID']}** will be rescheduled.")
                        st.balloons()

                with override_col2:
                    if st.button("❌ Accept AI Decision", use_container_width=True):
                        all_wins = resolution_df['New_Booking_Wins'].all()
                        if all_wins:
                            st.success("✅ **BOOKING CONFIRMED** — AI decision accepted. Your booking wins.")
                        else:
                            st.info("📋 **BOOKING QUEUED** — You'll be notified when equipment becomes available.")
                            st.markdown("**💡 Suggestions:**")
                            st.markdown("- Try a different date")
                            st.markdown("- Select a different equipment unit")
                            st.markdown("- Increase priority level if task is urgent")

            else:
                # No conflict - booking approved
                st.success("✅ **BOOKING APPROVED** — No conflicts detected!")
                st.markdown(f"Equipment **{book_id}** ({book_type}) is available for **{book_booking_date}**.")
                st.balloons()

                # AI confidence
                if model is not None:
                    try:
                        input_data = pd.DataFrame([{
                            'Priority_Score': PRIORITY_SCORES[book_priority],
                            'Duration_Days': DURATION_MAP[book_duration],
                            'Lead_Time': lead_time,
                            'Equipment_Category_Enc': encoders['equipment_cat'].transform([book_category])[0],
                            'Equipment_Type_Enc': encoders['equipment_type'].transform([book_type])[0],
                            'Shift_Enc': encoders['shift'].transform(['Day Shift (0700-1900)'])[0],
                            'Project_Enc': encoders['project'].transform([book_project])[0],
                            'Weather_Enc': 0,
                            'Wind_Speed_Knots': 10,
                            'Crane_Utilization_Pct': 60,
                        }])
                        probability = model.predict_proba(input_data)[0]
                        st.progress(probability[1])
                        st.caption(f"AI Confidence: {probability[1]*100:.1f}%")
                    except:
                        pass


# ============================================================
# PAGE: ANALYTICS
# ============================================================
elif page == "📈 Analytics":
    st.header("📈 Equipment Utilization Analytics")

    if 'Crane_Utilization_Pct' in df.columns:
        st.subheader("Equipment Utilization by Type")
        util_by_type = df.groupby('Equipment_Type')['Crane_Utilization_Pct'].mean().sort_values(ascending=False)
        st.bar_chart(util_by_type)

    st.markdown("---")

    if 'Has_Conflict' in df.columns:
        st.subheader("Conflicts by Equipment Type")
        conflict_by_type = df[df['Has_Conflict'] == 'Yes'].groupby('Equipment_Type').size().sort_values(ascending=False)
        if not conflict_by_type.empty:
            st.bar_chart(conflict_by_type)
        else:
            st.info("No conflicts recorded in the dataset.")

    st.markdown("---")

    st.subheader("Priority Distribution by Project")
    priority_project = pd.crosstab(df['Project'], df['Priority'])
    st.dataframe(priority_project, use_container_width=True)

    st.markdown("---")

    if 'Weather_Condition' in df.columns:
        st.subheader("Weather Impact on Operations")
        weather_status = pd.crosstab(df['Weather_Condition'], df['Status'])
        st.dataframe(weather_status, use_container_width=True)

    st.markdown("---")

    st.subheader("Booking Duration Distribution")
    duration_counts = df['Duration'].value_counts()
    st.bar_chart(duration_counts)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(f"🏗️ Equipment Booking AI System | Shipyard Operations | Built with Streamlit | {datetime.now().strftime('%Y')}")

