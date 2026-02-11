#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import yaml
from datetime import date

TOKEN_RE = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")

def render(template_text: str, ctx: dict) -> str:
    def repl(m):
        key = m.group(1)
        return str(ctx.get(key, f"{{{{{key}}}}}"))  # leave token if missing
    return TOKEN_RE.sub(repl, template_text)

def md_list(items):
    return "\n".join([f"- {x}" for x in items])

def main():
    repo = Path(__file__).resolve().parents[2]

    profile_path = repo / "data/policies_synth_md/system_profile.yaml"
    templates_dir = repo / "docs/templates"
    out_dir = repo / "data/policies_synth_md"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    ctx = {
        "SYNTHETIC_DISCLAIMER": profile.get("synthetic_disclaimer", "SYNTHETIC DEMO ARTIFACT — ACADEMIC USE ONLY"),
        "ORG_NAME": profile["organization"]["name"],
        "SYSTEM_NAME": profile["system"]["name"],
        "SYSTEM_DESCRIPTION": profile["system"].get("description", ""),
        "ENVIRONMENT": profile["system"].get("environment", "Hybrid"),
        "SYSTEM_TYPE": ", ".join(profile["system"].get("system_type", ["Web App", "API", "Database"])),
        "VERSION": "0.1",
        "EFFECTIVE_DATE": date.today().isoformat(),
        "OWNER_ROLE": "Fictional Role Owner",
        "REVIEW_CADENCE": "Review annually or upon major change.",
        "ROLES": "",
        "POLICY_STATEMENTS": "",
        "PROCEDURES": "",
        "EXCEPTIONS": "Exceptions require documented approval and POA&M tracking.",
        "GLOSSARY": "",
        "USER_ROLES": "\n".join([f"- **{u['type']}**: {u['description']}" for u in profile["system"]["users"]]),
        "DATA_TYPES": md_list(profile["system"]["data_types"]),
        "BOUNDARY": "- App tier\n- API tier\n- Identity service\n- Logging pipeline\n- Database",
        "CONSTRAINTS": md_list(profile.get("constraints", [])),
        "ESCALATION": "Define severity levels and notification targets (synthetic placeholder).",
        "LOGGING_REQUIREMENTS": "Log auth events, failures, privileged actions, and configuration changes.",
        "MONITORING_REVIEW": "Configure alerts and conduct periodic review (cadence TBD in synthetic pack).",
        "RETENTION_PROTECTION": "Retain logs for 90 days online and 1 year archived; protect from modification.",
        "TITLE": "",
        "DOC_ID": "",
    }

    docs = [
        ("mini_ssp_template.md", "01_mini_ssp.md", {
            "OWNER_ROLE": "Information System Security Manager (ISSM) (Fictional Role)",
        }),
        ("ac_policy_template.md", "02_access_control_policy.md", {
            "OWNER_ROLE": "Identity and Access Management (IAM) Lead (Fictional Role)",
            "ROLES": "- IAM Lead\n- System Administrators\n- Managers\n- Users",
            "POLICY_STATEMENTS": "- Access shall be granted based on least privilege.\n- Privileged accounts shall be separate from standard user accounts.\n- Access requests shall be approved prior to provisioning.\n- Accounts shall be disabled upon termination or role change.\n- *Intentional gap:* access recertification frequency is not defined.",
            "PROCEDURES": "1. User submits an access request with justification.\n2. Manager approves the request.\n3. Admin provisions access per role.\n4. Privileged access requires IAM Lead approval.\n5. *Intentional gap:* no periodic recertification procedure defined.",
        }),
        ("ir_plan_template.md", "03_incident_response_plan.md", {
            "OWNER_ROLE": "Security Operations Lead (Fictional Role)",
            "ROLES": "- IR Lead\n- Security Analyst\n- System Admin\n- Comms Lead",
            "POLICY_STATEMENTS": "- Incidents shall be triaged, contained, eradicated, and recovered.\n- Evidence shall be preserved for investigation.\n- Stakeholders shall be notified.\n- *Intentional gap:* escalation time objectives are not defined.",
            "PROCEDURES": "1. Detect and report.\n2. Triage and classify.\n3. Contain.\n4. Eradicate.\n5. Recover.\n6. Conduct post-incident review.\n7. *Intentional gap:* no exercise schedule defined.",
        }),
        ("logging_standard_template.md", "04_logging_monitoring_standard.md", {
            "OWNER_ROLE": "Security Engineering Lead (Fictional Role)",
            "ROLES": "- Security Engineering\n- Security Operations\n- System Admins",
            "LOGGING_REQUIREMENTS": "- Authentication events\n- Authorization failures\n- Privileged actions\n- Configuration changes",
            "MONITORING_REVIEW": "- Alerts on suspicious activity\n- *Intentional gap:* review cadence not specified for all event types\n- *Intentional gap:* no monitoring KPIs defined",
            "RETENTION_PROTECTION": "- Retain 90 days online, 1 year archived\n- Protect logs from modification",
        }),
    ]

    for tpl_name, out_name, overrides in docs:
        tpl_path = templates_dir / tpl_name
        text = tpl_path.read_text(encoding="utf-8")
        merged = dict(ctx)
        merged.update(overrides)
        rendered = render(text, merged)
        (out_dir / out_name).write_text(rendered, encoding="utf-8")
        print("Wrote:", out_dir / out_name)

if __name__ == "__main__":
    main()
