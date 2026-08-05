from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_weekly_report_endpoint():
    response = client.get("/reports/weekly")
    assert response.status_code == 200
    assert "total_predicted_sales" in response.json()


def test_monthly_report_endpoint():
    response = client.get("/reports/monthly")
    assert response.status_code == 200
    assert "monthly_revenue_forecast" in response.json()
