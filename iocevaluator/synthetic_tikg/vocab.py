"""Vocabulary for the SYNTHETIC TIKG: campaign families (threat behaviours), name pools and text phrases.

Everything here is invented.  Vendors, products, actors and indicators are fictional; CVE ids use the
non-existent years 2090-2099, domains/e-mails use reserved ``.example``/``.invalid`` names and IPs use the
RFC 5737 documentation ranges, so no value can collide with a real indicator.
"""
from __future__ import annotations

from typing import Dict, List

DEVICE_KINDS = ("fdm_printer", "sla_printer", "sls_printer", "dmls_printer", "cnc_machine", "robot_arm",
                "plc", "workstation", "controller", "historian_server")
PLATFORM_KINDS = ("fabos_firmware", "rtos_firmware", "windows", "linux", "plc_runtime", "cloud_service",
                  "mes_app", "cam_software")

# device kind -> platform kinds it can run on (used for T7 <D, runs_on, P>)
DEVICE_PLATFORM_COMPAT: Dict[str, tuple] = {
    "fdm_printer": ("fabos_firmware", "rtos_firmware", "linux"),
    "sla_printer": ("rtos_firmware", "linux"),
    "sls_printer": ("rtos_firmware", "linux", "windows"),
    "dmls_printer": ("rtos_firmware", "windows", "cam_software"),
    "cnc_machine": ("rtos_firmware", "plc_runtime", "windows"),
    "robot_arm": ("rtos_firmware", "linux", "plc_runtime"),
    "plc": ("plc_runtime", "rtos_firmware"),
    "workstation": ("windows", "linux", "cam_software", "cloud_service", "mes_app"),
    "controller": ("rtos_firmware", "plc_runtime", "linux"),
    "historian_server": ("windows", "linux", "mes_app"),
}
DEVICE_KIND_LABEL = {
    "fdm_printer": "FDM printer", "sla_printer": "SLA printer", "sls_printer": "SLS printer",
    "dmls_printer": "DMLS metal printer", "cnc_machine": "CNC machine", "robot_arm": "robotic arm",
    "plc": "PLC", "workstation": "engineering workstation", "controller": "motion controller",
    "historian_server": "process historian",
}
PLATFORM_KIND_LABEL = {
    "fabos_firmware": "FabOS firmware", "rtos_firmware": "SynRTOS firmware", "windows": "Windows 10 build",
    "linux": "Linux LTS", "plc_runtime": "PLC runtime", "cloud_service": "CloudPLM service",
    "mes_app": "MES suite", "cam_software": "CAM/slicer suite",
}

VENDORS = ["Axiom Fabrication", "Brightlayer", "Cobaltforge", "Dunmere Systems", "Eventide Mechatronics",
           "Farrow Additive", "Gantrix", "Helix Prototyping", "Ironvale Controls", "Juniper Fab",
           "Kestrel Automation", "Lumen Additive"]

ACTOR_ADJ = ["Crimson", "Hollow", "Static", "Amber", "Velvet", "Iron", "Glass", "Silent", "Copper", "Ashen",
             "Cobalt", "Pale", "Rusted", "Gilded", "Neon", "Slate"]
ACTOR_NOUN = ["Gantry", "Spindle", "Lathe", "Filament", "Hotend", "Bracket", "Resin", "Lattice", "Anvil",
              "Sprocket", "Platen", "Extruder", "Mandrel", "Gasket", "Rivet", "Turbine"]
ACTOR_SUFFIX = ["", " Group", " Collective", " Syndicate"]

ATTACK_TYPE_BASE = ["Ransomware", "Data breach", "Denial of service", "IP theft", "Sabotage", "Supply-chain tampering",
                    "Credential theft", "Remote code execution", "Defacement", "Cryptojacking", "Espionage",
                    "Wiper activity", "Insider misuse", "Fraud"]
ATTACK_TYPE_QUAL = ["build queue", "design repository", "printer fleet", "supplier portal", "CNC cell",
                    "quality records", "firmware update channel", "PLM database", "MES gateway", "robot cell",
                    "slicer toolchain", "engineering workstations", "historian", "cloud storage"]

TECHNIQUE_VERBS = ["Phishing delivery", "Script execution", "Signed binary proxy execution", "Obfuscated payload",
                   "Valid account abuse", "Exploit public-facing application", "Firmware modification",
                   "Remote service abuse", "Data staging", "File tampering", "Lateral tool transfer",
                   "Scheduled task persistence", "Credential dumping", "Network scanning", "Protocol abuse"]

COUNTRIES = ["GB", "US", "DE", "CN", "RU", "IN", "FR", "JP", "KR", "IT", "BR", "CA", "AU", "NL", "SE", "PL",
             "TR", "IL", "SG", "MX"]
INDUSTRIES = ["additive manufacturing", "aerospace", "automotive", "medical devices", "defence supply chain",
              "energy", "tooling", "electronics", "consumer goods", "construction"]

# Campaign families = latent threat behaviours.  Related campaigns (same family) share the device/platform mix,
# target sectors/countries, weakness vocabulary and attack-surface bias; unrelated families do not.
FAMILIES: List[dict] = [
    dict(key="cad_tampering", title="CAD/STL design-file tampering",
         devices={"workstation": 4, "dmls_printer": 2, "sls_printer": 1, "fdm_printer": 1},
         platforms={"windows": 3, "cam_software": 3, "cloud_service": 2, "linux": 1},
         weakness=["improper path validation in model import", "unchecked mesh header length",
                   "insecure deserialisation of design archives"],
         tradecraft=["spear-phishing with poisoned design archives", "modification of mesh files in transit"],
         src=["RU", "CN"], tgt=["US", "DE", "GB"], ind=["aerospace", "medical devices", "additive manufacturing"],
         p_net=0.55, impact_hi=0.7),
    dict(key="firmware_implant", title="printer firmware implantation",
         devices={"fdm_printer": 4, "sla_printer": 2, "controller": 2, "sls_printer": 1},
         platforms={"fabos_firmware": 4, "rtos_firmware": 3, "linux": 1},
         weakness=["unsigned firmware image acceptance", "buffer overflow in update parser",
                   "hard-coded service credential"],
         tradecraft=["malicious firmware update packages", "tampering with bootloader configuration"],
         src=["CN", "KR"], tgt=["JP", "DE", "US"], ind=["electronics", "additive manufacturing", "consumer goods"],
         p_net=0.30, impact_hi=0.6),
    dict(key="ot_ransomware", title="ransomware against shop-floor hosts",
         devices={"workstation": 3, "historian_server": 3, "plc": 2, "cnc_machine": 1},
         platforms={"windows": 4, "plc_runtime": 2, "mes_app": 2},
         weakness=["remote desktop service flaw", "SMB protocol memory corruption",
                   "privilege escalation in service installer"],
         tradecraft=["encryption of build queues and job archives", "double-extortion with leaked build files"],
         src=["RU", "BR"], tgt=["US", "GB", "IT"], ind=["automotive", "tooling", "energy"],
         p_net=0.80, impact_hi=0.85),
    dict(key="supplier_compromise", title="supplier artefact compromise",
         devices={"workstation": 3, "dmls_printer": 2, "cnc_machine": 2, "controller": 1},
         platforms={"cam_software": 4, "cloud_service": 2, "windows": 2, "mes_app": 1},
         weakness=["trusted installer update hijack", "dependency confusion in plugin loader",
                   "weak signature verification of vendor packages"],
         tradecraft=["trojanised vendor toolchain installers", "compromised supplier file shares"],
         src=["CN", "RU"], tgt=["FR", "DE", "NL"], ind=["defence supply chain", "aerospace", "tooling"],
         p_net=0.60, impact_hi=0.65),
    dict(key="cloud_repo_exfil", title="cloud repository exfiltration",
         devices={"workstation": 5, "historian_server": 1, "sls_printer": 1},
         platforms={"cloud_service": 5, "linux": 2, "windows": 1},
         weakness=["missing authorisation on storage API", "server-side request forgery in preview service",
                   "token leakage in build logs"],
         tradecraft=["abuse of misconfigured design repositories", "bulk download of PLM exports"],
         src=["IN", "TR"], tgt=["SE", "US", "CA"], ind=["medical devices", "consumer goods", "additive manufacturing"],
         p_net=0.92, impact_hi=0.5),
    dict(key="build_param_sabotage", title="slicer and build-parameter sabotage",
         devices={"sls_printer": 3, "dmls_printer": 3, "fdm_printer": 2, "workstation": 2},
         platforms={"cam_software": 4, "rtos_firmware": 2, "fabos_firmware": 1, "windows": 1},
         weakness=["parameter range not validated in job loader", "command injection in G-code post-processor",
                   "integrity check bypass for build recipes"],
         tradecraft=["subtle porosity-inducing parameter changes", "modification of orientation and layer settings"],
         src=["KR", "RU"], tgt=["AU", "GB", "JP"], ind=["aerospace", "automotive", "medical devices"],
         p_net=0.40, impact_hi=0.75),
    dict(key="remote_access_intrusion", title="remote-access intrusion into production networks",
         devices={"workstation": 3, "controller": 2, "robot_arm": 1, "historian_server": 2},
         platforms={"linux": 3, "windows": 3, "rtos_firmware": 1},
         weakness=["authentication bypass in VPN appliance", "path traversal in remote management portal",
                   "default credentials on jump host"],
         tradecraft=["VPN credential stuffing", "living-off-the-land lateral movement"],
         src=["IL", "CN"], tgt=["IT", "FR", "PL"], ind=["energy", "construction", "tooling"],
         p_net=0.95, impact_hi=0.8),
    dict(key="ics_protocol_abuse", title="industrial protocol abuse",
         devices={"plc": 4, "cnc_machine": 3, "controller": 3},
         platforms={"plc_runtime": 5, "rtos_firmware": 3},
         weakness=["unauthenticated write command acceptance", "stack overflow in protocol stack",
                   "denial of service via malformed frames"],
         tradecraft=["unauthorised register writes to motion controllers", "replay of captured control frames"],
         src=["RU", "TR"], tgt=["PL", "SE", "MX"], ind=["energy", "automotive", "tooling"],
         p_net=0.50, impact_hi=0.9),
    dict(key="mes_erp_intrusion", title="MES/ERP business-system intrusion",
         devices={"historian_server": 3, "workstation": 4, "plc": 1},
         platforms={"mes_app": 5, "windows": 3, "cloud_service": 1},
         weakness=["SQL injection in order-release module", "stored cross-site scripting in job dashboard",
                   "insecure direct object reference in work-order API"],
         tradecraft=["manipulation of work orders and bills of materials", "credential harvesting from MES portals"],
         src=["BR", "IN"], tgt=["CA", "MX", "US"], ind=["consumer goods", "electronics", "construction"],
         p_net=0.85, impact_hi=0.45),
    dict(key="robot_cell_attack", title="robotic cell and controller attack",
         devices={"robot_arm": 4, "controller": 3, "cnc_machine": 2, "plc": 1},
         platforms={"rtos_firmware": 4, "plc_runtime": 3, "linux": 2},
         weakness=["unauthenticated teach-pendant service", "integer overflow in trajectory parser",
                   "insecure update channel for motion firmware"],
         tradecraft=["trajectory manipulation of post-processing cells", "disabling of safety interlock logging"],
         src=["KR", "CN"], tgt=["JP", "KR", "SG"], ind=["automotive", "electronics", "defence supply chain"],
         p_net=0.35, impact_hi=0.8),
]
