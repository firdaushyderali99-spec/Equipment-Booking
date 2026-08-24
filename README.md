
# 🏗️ Equipment Booking AI - Conflict Resolution System

Intelligent equipment booking management system for shipyard operations with ML-powered conflict detection.

## Features

- 📊 **Dashboard** - KPIs, charts, conflict statistics
- 📋 **Booking Manager** - Filter and browse all bookings
- ⚠️ **Conflict Detector** - Real-time conflict detection with resolution
- 🤖 **AI Prediction** - ML model predicts booking outcomes
- 📈 **Analytics** - Equipment utilization and weather impact

## Conflict Resolution Logic

1. **Default**: First-Come-First-Served (earlier Request Date wins)
2. **Override**: Higher Priority wins regardless of request date
   - Urgent (4) > High (3) > Medium (2) > Low (1)

## Deployment

This app is deployed on Streamlit Cloud.

## Tech Stack

- Streamlit
- Pandas
- scikit-learn (Gradient Boosting Classifier)
- OpenPyXL

