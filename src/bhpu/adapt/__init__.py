"""Domain adaptation for cross-domain (literature→JnJ) PU transfer.

Naive literature→JnJ PU scores ≈ random because a P-vs-U classifier learns a *domain shortcut*
rather than feasibility. This subpackage supplies the tools to diagnose and remove it:

* :class:`DomainDiscriminator` — measures domain separability (the diagnostic AUC) and produces
  covariate-shift importance weights ``w(x) = d(x) / (1 − d(x))``.
* :class:`CoralAligner` — closed-form second-order feature alignment (CORAL).

See ``docs/decisions/003-crossdomain-transfer-diagnosis.md``.
"""

from __future__ import annotations

from bhpu.adapt.coral import CoralAligner
from bhpu.adapt.discriminator import DomainDiscriminator

__all__ = ["CoralAligner", "DomainDiscriminator"]
