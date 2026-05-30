from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PUBLIC_ACCESS = 1
INTERNAL_ACCESS = 2
CONFIDENTIAL_ACCESS = 3
EMAIL_RESTRICTED_ACCESS = 99


ACCESS_LABELS = {
    PUBLIC_ACCESS: "Public",
    INTERNAL_ACCESS: "Internal",
    CONFIDENTIAL_ACCESS: "Confidential",
    EMAIL_RESTRICTED_ACCESS: "Email Restricted",
}


@dataclass(frozen=True)
class EmployeeAccount:
    user_id: str
    full_name: str
    email: str
    department: str
    access_level: int

    @property
    def access_label(self) -> str:
        return access_label(self.access_level)


DEMO_EMPLOYEES = [
    EmployeeAccount(
        user_id="11111111-1111-4111-8111-111111111111",
        full_name="Anna Keller",
        email="anna.keller@six-demo.local",
        department="Operations",
        access_level=PUBLIC_ACCESS,
    ),
    EmployeeAccount(
        user_id="22222222-2222-4222-8222-222222222222",
        full_name="Ben Meier",
        email="ben.meier@six-demo.local",
        department="Compliance",
        access_level=INTERNAL_ACCESS,
    ),
    EmployeeAccount(
        user_id="33333333-3333-4333-8333-333333333333",
        full_name="Clara Rossi",
        email="clara.rossi@six-demo.local",
        department="Regulatory Management",
        access_level=CONFIDENTIAL_ACCESS,
    ),
]


def access_label(access_level: int) -> str:
    return ACCESS_LABELS.get(access_level, f"Level {access_level}")


def access_options() -> list[tuple[str, int]]:
    return [
        (ACCESS_LABELS[PUBLIC_ACCESS], PUBLIC_ACCESS),
        (ACCESS_LABELS[INTERNAL_ACCESS], INTERNAL_ACCESS),
        (ACCESS_LABELS[CONFIDENTIAL_ACCESS], CONFIDENTIAL_ACCESS),
        (ACCESS_LABELS[EMAIL_RESTRICTED_ACCESS], EMAIL_RESTRICTED_ACCESS),
    ]


def infer_document_access(path: Path) -> tuple[int, str]:
    haystack = " ".join(path.parts).replace("_", " ").replace("-", " ").lower()
    if path.suffix.lower() in {".eml", ".msg"} or any(
        token in haystack for token in ("email", "e-mail", "mailbox")
    ):
        return EMAIL_RESTRICTED_ACCESS, ACCESS_LABELS[EMAIL_RESTRICTED_ACCESS]
    if "confidential" in haystack:
        return CONFIDENTIAL_ACCESS, ACCESS_LABELS[CONFIDENTIAL_ACCESS]
    if any(token in haystack for token in ("transcript", "internal", "master data")):
        return INTERNAL_ACCESS, ACCESS_LABELS[INTERNAL_ACCESS]
    return PUBLIC_ACCESS, ACCESS_LABELS[PUBLIC_ACCESS]


def employee_from_row(row: dict[str, object]) -> EmployeeAccount:
    return EmployeeAccount(
        user_id=str(row["user_id"]),
        full_name=str(row["full_name"]),
        email=str(row["email"]),
        department=str(row.get("department") or ""),
        access_level=int(row["access_level"]),
    )
