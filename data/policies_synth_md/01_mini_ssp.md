# Mini-SSP / System Overview — Rivermark Operations Portal (ROP) (Fictional)

> **SYNTHETIC DEMO ARTIFACT — ACADEMIC USE ONLY**

**Document ID:** 01_MINI_SSP  
**Version:** 0.1  
**Effective Date:** 2026-02-11  
**Owner:** Information System Security Manager (ISSM) (Fictional Role)  
**Applies To:** Rivermark Operations Portal (ROP) (Fictional)  

---

## 1.0 Purpose
Describe the system boundary, major components, users, and security/privacy posture for Rivermark Operations Portal (ROP) (Fictional) operated by Rivermark Logistics (Fictional).

## 2.0 Scope
Applies to the Rivermark Operations Portal (ROP) system boundary and supporting services used by Rivermark Logistics (Fictional).

## 3.0 Roles and Responsibilities
- **System Owner (SO):** accountable for system risk decisions and resourcing.
- **ISSM:** maintains SSP artifacts and coordinates evidence collection.
- **System Administrator:** implements configurations and access provisioning.
- **Security Analyst:** monitors alerts and supports incident response.
- **Privacy Officer:** reviews PII handling and privacy requirements.

## 4.0 Policy Statements
- The system boundary shall include the web application, API tier, identity service, logging pipeline, and database.
- The system shall maintain audit logs for security-relevant events.
- The system shall process limited PII (name, email, user ID) and apply data minimization.
- *Intentional gap:* This document does not specify a formal continuous monitoring cadence.

## 5.0 Procedures
- Maintain a system diagram and component inventory.
- Maintain a list of integrated services and third-party dependencies.
- Review boundary changes during major releases.
- *Intentional gap:* No defined approval workflow for boundary changes is provided.

## 6.0 Exceptions
Exceptions require approval by the System Owner and ISSM and must be documented in the POA&M tracker.

## 7.0 Review Cadence
Review annually or upon major architectural change. *(Intentional gap: exact month/owner cadence not specified.)*

## 8.0 Definitions
- **System Boundary:** the set of components and services included for security assessment.
- **PII:** personally identifiable information.
