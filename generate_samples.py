import os
import json

def generate_samples():
    # Create directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Define compliance rules
    rules = [
        {
            "id": "COMP-001",
            "name": "Interest Rate Cap Limit",
            "category": "Lending Regulations",
            "description": "The maximum interest rate cap on any adjustable-rate mortgage (ARM) or loan must not exceed 8.5% per annum at any point during the loan term."
        },
        {
            "id": "COMP-002",
            "name": "Late Payment Grace Period",
            "category": "Consumer Protection",
            "description": "The contract must specify a payment grace period of at least 15 calendar days from the payment due date before any late fee is assessed."
        },
        {
            "id": "COMP-003",
            "name": "Identity Verification (KYC)",
            "category": "AML/KYC Compliance",
            "description": "The agreement must explicitly state that the customer's identity was verified using a government-issued photo ID (e.g., driver's license or passport) prior to account activation."
        },
        {
            "id": "COMP-004",
            "name": "Late Fee Cap",
            "category": "Consumer Protection",
            "description": "The late payment fee must not exceed $50 or 5% of the overdue payment amount, whichever is lower."
        },
        {
            "id": "COMP-005",
            "name": "Right of Rescission (Cancellation)",
            "category": "Lending Regulations",
            "description": "The borrower must be granted a Right of Rescission allowing them to cancel the agreement without penalty by providing written notice within 3 business days of execution."
        }
    ]
    
    # Write rules.json
    rules_path = os.path.join("sample_data", "rules.json")
    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=4)
    print(f"Created sample rules at: {rules_path}")

    # 2. Define sample contract (with explicit compliance passes and failures)
    contract_text = """SECURED HOME MORTGAGE AGREEMENT

This Secured Home Mortgage Agreement ("Agreement") is entered into on this 1st day of June, 2026, by and between the borrower, John Doe ("Borrower"), and Apex Lending Solutions ("Lender").

SECTION 1: CUSTOMER IDENTIFICATION AND ONBOARDING
In accordance with standard onboarding procedures, the Lender has completed customer identity verification. The Borrower's identity verification was completed electronically via email confirmation and verification of a standard utility bill showing proof of address. No physical or government-issued photo identification was collected or required during the remote digital onboarding session. The account was activated immediately following email verification.

SECTION 2: INTEREST RATE AND ADJUSTMENTS
The initial interest rate for this loan is established at 5.5% per annum. This is an adjustable-rate mortgage. The interest rate may fluctuate semi-annually based on index changes. However, the parties agree that the maximum adjustable interest rate cap on this mortgage agreement shall not exceed 9.75% per annum at any point during the lifetime of this loan.

SECTION 3: PAYMENT TERMS AND GRACE PERIOD
Monthly payments are due on the first (1st) day of each calendar month. The Lender shall allow a grace period of 10 calendar days. If the full monthly payment is not received by the Lender by 11:59 PM on the 10th day of the month, the payment will be deemed late, and a late fee will be automatically assessed on the 11th.

SECTION 4: LATE FEES
For any payment received after the grace period specified in Section 3, a flat late payment penalty fee of $75.00 shall be charged to the Borrower's account.

SECTION 5: BORROWER'S RIGHT TO CANCEL (RESCISSION)
The Lender hereby discloses that the Borrower has the right to cancel this mortgage transaction, without any penalty or obligation, within three (3) business days from the date of the execution of this Agreement. To exercise this right, the Borrower must send written, signed notice of cancellation to the Lender's official email address or physical office prior to midnight of the third business day.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

Borrower: John Doe (Signature)
Lender Representative: Apex Lending Solutions (Signature)
"""

    # Write sample_contract.txt
    contract_path = os.path.join("sample_data", "sample_contract.txt")
    with open(contract_path, "w") as f:
        f.write(contract_text)
    print(f"Created sample contract at: {contract_path}")

if __name__ == "__main__":
    generate_samples()
