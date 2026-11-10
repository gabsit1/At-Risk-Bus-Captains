# 🚌 At-Risk Bus Captains Dashboard

An interactive Streamlit web application designed to analyze, monitor, and visualize bus captain performance metrics, risk factors, and safety insights.

---

## 🚀 Features
- **Cross-Validated Performance:** Evaluates model accuracy and metrics like ROC-AUC.
- **Feature Importance:** Visualizes key variables driving performance predictions using interactive Plotly charts.
- **Risk Scoring:** Identifies and highlights at-risk patterns to support proactive safety management.

---

## 🛠️ Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gabsit1/At-Risk-Bus-Captains.git](https://github.com/gabsit1/At-Risk-Bus-Captains.git)
   cd At-Risk-Bus-Captains
2. **Create and activate a virtual environment:**
   python -m venv .venv
   .venv\Scripts\activate
3.**Install the required Python packages:**
   pip install -r requirements.txt
4.**Generate the required data files:**
   python atrisk_pipeline.py
5.**Start the Streamlit dashboard:**
   streamlit run dashboard.py
