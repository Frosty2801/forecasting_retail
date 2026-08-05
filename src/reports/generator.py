from dataclasses import dataclass

from fpdf import FPDF

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WeeklyBusinessReport:
    """Structured weekly business report."""

    generated_at: str
    total_predicted_sales: float
    weekly_growth_pct: float
    best_selling_categories: list[str]
    worst_selling_categories: list[str]
    highest_sales_day: str
    lowest_sales_day: str
    recommendations: list[str]


@dataclass
class MonthlyBusinessReport:
    """Structured monthly business report."""

    generated_at: str
    monthly_revenue_forecast: float
    mom_growth_pct: float
    top_products: list[str]
    top_stores: list[int]
    demand_trends: str
    business_insights: list[str]


class ReportExporter:
    """Exports business reports to JSON, CSV, or PDF."""

    @staticmethod
    def to_json(report: WeeklyBusinessReport | MonthlyBusinessReport) -> str:
        import json

        return json.dumps(report.__dict__, indent=2)

    @staticmethod
    def to_pdf(report: WeeklyBusinessReport | MonthlyBusinessReport, output_path: str) -> None:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Retail Sales Business Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated At: {report.generated_at}", ln=True)
        pdf.ln(5)

        for key, value in report.__dict__.items():
            pdf.set_font("Arial", "B", 10)
            pdf.cell(50, 8, f"{key.replace('_', ' ').title()}:", ln=0)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 8, f"{value}", ln=1)

        pdf.output(output_path)
        logger.info(f"Report exported to PDF: {output_path}")
