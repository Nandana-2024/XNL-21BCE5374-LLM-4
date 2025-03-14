from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

def detect_model_drift(reference_data, current_data):
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(reference_data, current_data)
    dashboard.show()
