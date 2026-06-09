# AI-Driven Compliance Audit Report

**Target Document**: `sample_contract.txt`  
**Audit Generation**: Automated compliance check  

## Executive Summary

- **Total Rules Checked**: 5
- **Compliant (PASS)**: 1 ✅
- **Violations (FAIL)**: 4 ❌
- **Ambiguities (WARNING)**: 0 ⚠️
- **Not Applicable (N/A)**: 0 ➖

## Detailed Audit Trail

### 1. Interest Rate Cap Limit (COMP-001)
- **Category**: Lending Regulations
- **Requirement**: The maximum interest rate cap on any adjustable-rate mortgage (ARM) or loan must not exceed 8.5% per annum at any point during the loan term.
- **Evaluation Status**: **❌ FAIL**
- **Confidence Score**: 95%
- **Auditor Reasoning**: The mortgage agreement specifies an adjustable interest rate cap of 9.75% per annum, which exceeds the allowed regulatory limit of 8.5% per annum.
- **Verbatim Evidence Quote**:
  > "the maximum adjustable interest rate cap on this mortgage agreement shall not exceed 9.75% per annum"
- **Verification Check Notes**: *Auditor verified correctly. The contract states 9.75% max cap, violating the limit of 8.5%.*

---

### 2. Late Payment Grace Period (COMP-002)
- **Category**: Consumer Protection
- **Requirement**: The contract must specify a payment grace period of at least 15 calendar days from the payment due date before any late fee is assessed.
- **Evaluation Status**: **❌ FAIL**
- **Confidence Score**: 95%
- **Auditor Reasoning**: The contract establishes a grace period of only 10 calendar days before a late fee is assessed, violating the consumer protection rule of at least 15 days.
- **Verbatim Evidence Quote**:
  > "The Lender shall allow a grace period of 10 calendar days."
- **Verification Check Notes**: *Auditor is correct. Contract states 10 calendar days grace period, which violates the 15-day requirement.*

---

### 3. Identity Verification (KYC) (COMP-003)
- **Category**: AML/KYC Compliance
- **Requirement**: The agreement must explicitly state that the customer's identity was verified using a government-issued photo ID (e.g., driver's license or passport) prior to account activation.
- **Evaluation Status**: **❌ FAIL**
- **Confidence Score**: 95%
- **Auditor Reasoning**: The document states that onboarding was completed using email and utility bill confirmation, and explicitly notes that no physical or government-issued photo identification was collected.
- **Verbatim Evidence Quote**:
  > "No physical or government-issued photo identification was collected or required"
- **Verification Check Notes**: *Auditor identified correct clause. Verification is failed because utility bill is not a government photo ID.*

---

### 4. Late Fee Cap (COMP-004)
- **Category**: Consumer Protection
- **Requirement**: The late payment fee must not exceed $50 or 5% of the overdue payment amount, whichever is lower.
- **Evaluation Status**: **❌ FAIL**
- **Confidence Score**: 95%
- **Auditor Reasoning**: The late fee specified in Section 4 is a flat penalty of $75.00, which violates the regulatory cap of $50.
- **Verbatim Evidence Quote**:
  > "a flat late payment penalty fee of $75.00 shall be charged"
- **Verification Check Notes**: *Auditor is correct. Contract states 10 calendar days grace period, which violates the 15-day requirement.*

---

### 5. Right of Rescission (Cancellation) (COMP-005)
- **Category**: Lending Regulations
- **Requirement**: The borrower must be granted a Right of Rescission allowing them to cancel the agreement without penalty by providing written notice within 3 business days of execution.
- **Evaluation Status**: **✅ PASS**
- **Confidence Score**: 98%
- **Auditor Reasoning**: Section 5 explicitly states that the borrower has the right to cancel the transaction without penalty within three business days from the execution of the agreement.
- **Verbatim Evidence Quote**:
  > "Borrower has the right to cancel this mortgage transaction, without any penalty or obligation, within three (3) business days"
- **Verification Check Notes**: *Auditor verified correctly. The contract grants 3 business days, matching the rule.*

---

