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

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gabsit1/At-Risk-Bus-Captains.git
   cd At-Risk-Bus-Captains
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   For Windows (PowerShell):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   For Windows (Command Prompt):

   ```cmd
   .venv\Scripts\activate.bat
   ```

4. **Install the required Python packages:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Generate the required data files:**

   ```bash
   python atrisk_pipeline.py
   ```

6. **Launch the Streamlit dashboard:**

   ```bash
   streamlit run dashboard.py
   ```
