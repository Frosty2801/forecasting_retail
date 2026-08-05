from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """Represents a data quality issue found during validation."""

    table_name: str
    issue_type: str  # missing_values, duplicates, invalid_dates, outliers, negative_sales, invalid_store_id, invalid_family
    severity: str    # WARNING, ERROR
    description: str
    affected_count: int


@dataclass
class ValidationReport:
    """Aggregate report of all data validation checks."""

    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.is_valid = False
