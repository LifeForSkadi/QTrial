"""Optional Tianyan platform submission (guarded; needs a login key)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("qtrail")


def submit_qcis(qcis_path: str | Path, machine: str = "tianyan-287",
                shots: int = 12000, exp_name: str | None = None,
                token: str | None = None) -> dict | None:
    """Submit a QCIS file to the Tianyan platform.

    Requires a login key (arg > env TIANYAN_LOGIN_KEY). Returns the query ids
    on success, None with a friendly message when unavailable.
    """
    token = token or os.environ.get("TIANYAN_LOGIN_KEY", "") or         os.environ.get("CQLIB_LOGIN_KEY", "")
    if not token:
        log.warning("no Tianyan login key (arg / TIANYAN_LOGIN_KEY env); "
                    "skipping platform submission — local outputs are ready")
        return None
    try:
        from cqlib import TianYanPlatform, QuantumLanguage
        platform = TianYanPlatform(login_key=token, machine_name=machine)
        with open(qcis_path, encoding="utf-8") as f:
            qcis = f.read()
        # pre-validate against platform rules
        try:
            platform.qcis_check_regular(qcis)
        except Exception:
            pass  # check endpoint is advisory
        query_ids = platform.submit_job(
            circuit=qcis, language=QuantumLanguage.QCIS,
            name=exp_name or Path(qcis_path).stem, num_shots=shots)
        log.info("submitted to %s: query_ids=%s", machine, query_ids)
        return {"machine": machine, "query_ids": query_ids}
    except Exception as e:
        log.warning("platform submission failed (%s); local outputs remain", e)
        return None
