from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):

    model_config = ConfigDict(extra="forbid")

    loan_amount: Optional[float] = None
    term: Optional[float] = None
    property_value: Optional[float] = None
    income: Optional[float] = None
    Credit_Score: Optional[float] = None
    LTV: Optional[float] = None
    dtir1: Optional[float] = None

    loan_limit: Optional[str] = None
    Gender: Optional[str] = None
    approv_in_adv: Optional[str] = None
    loan_type: Optional[str] = None
    loan_purpose: Optional[str] = None
    Credit_Worthiness: Optional[str] = None
    open_credit: Optional[str] = None
    business_or_commercial: Optional[str] = None
    Neg_ammortization: Optional[str] = None
    interest_only: Optional[str] = None
    lump_sum_payment: Optional[str] = None
    construction_type: Optional[str] = None
    occupancy_type: Optional[str] = None
    Secured_by: Optional[str] = None
    total_units: Optional[str] = None
    credit_type: Optional[str] = None

    co_applicant_credit_type: Optional[str] = Field(
        default=None,
        alias="co-applicant_credit_type"
    )

    age: Optional[str] = None
    submission_of_application: Optional[str] = None
    Region: Optional[str] = None
    Security_Type: Optional[str] = None
