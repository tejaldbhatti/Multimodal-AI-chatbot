"""
Tax router — handles all /tax endpoints.
Exposes all tax calculators as REST API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.tools.tax_employee import calculate_employee_tax
from backend.tools.tax_freelancer import calculate_freelancer_tax
from backend.tools.tax_couple import calculate_couple_tax
from backend.tools.tax_single_parent import calculate_single_parent_tax
from backend.tools.tax_class_advisor import advise_steuerklasse
from backend.tools.steuerberater_prep import generate_steuerberater_checklist

router = APIRouter()


# ── Request models ───────────────────────────────────────
class EmployeeTaxRequest(BaseModel):
    country: str
    annual_income: float
    steuerklasse: str = "I"
    has_church_tax: bool = False


class FreelancerTaxRequest(BaseModel):
    country: str
    annual_revenue: float
    business_expenses: float = 0
    is_kleinunternehmer: bool = False
    freelancer_type: str = "freiberufler"


class CoupleTaxRequest(BaseModel):
    country: str
    income_partner1: float
    income_partner2: float


class SingleParentRequest(BaseModel):
    country: str
    annual_income: float
    number_of_children: int = 1


class SteuerklasseRequest(BaseModel):
    income_partner1: float
    income_partner2: float


class ChecklistRequest(BaseModel):
    country: str
    employment_type: str
    family_situation: str
    has_investments: bool = False
    has_rental_income: bool = False


# ── Endpoints ────────────────────────────────────────────
@router.post("/employee")
def employee_tax(request: EmployeeTaxRequest):
    """Calculate income tax for an employed person."""
    result = calculate_employee_tax.invoke({
        "country": request.country,
        "annual_income": request.annual_income,
        "steuerklasse": request.steuerklasse,
        "has_church_tax": request.has_church_tax
    })
    return {"data": result}


@router.post("/freelancer")
def freelancer_tax(request: FreelancerTaxRequest):
    """Calculate tax for a freelancer / self-employed person."""
    result = calculate_freelancer_tax.invoke({
        "country": request.country,
        "annual_revenue": request.annual_revenue,
        "business_expenses": request.business_expenses,
        "is_kleinunternehmer": request.is_kleinunternehmer,
        "freelancer_type": request.freelancer_type
    })
    return {"data": result}


@router.post("/couple")
def couple_tax(request: CoupleTaxRequest):
    """Calculate tax for a married couple."""
    result = calculate_couple_tax.invoke({
        "country": request.country,
        "income_partner1": request.income_partner1,
        "income_partner2": request.income_partner2
    })
    return {"data": result}


@router.post("/single-parent")
def single_parent_tax(request: SingleParentRequest):
    """Calculate tax for a single parent."""
    result = calculate_single_parent_tax.invoke({
        "country": request.country,
        "annual_income": request.annual_income,
        "number_of_children": request.number_of_children
    })
    return {"data": result}


@router.post("/steuerklasse")
def steuerklasse_advisor(request: SteuerklasseRequest):
    """Recommend best Steuerklasse for a German couple."""
    result = advise_steuerklasse.invoke({
        "income_partner1": request.income_partner1,
        "income_partner2": request.income_partner2
    })
    return {"data": result}


@router.post("/checklist")
def tax_checklist(request: ChecklistRequest):
    """Generate personalised tax advisor preparation checklist."""
    result = generate_steuerberater_checklist.invoke({
        "country": request.country,
        "employment_type": request.employment_type,
        "family_situation": request.family_situation,
        "has_investments": request.has_investments,
        "has_rental_income": request.has_rental_income
    })
    return {"data": result}


@router.get("/health")
def tax_health():
    return {"status": "tax router ok"}