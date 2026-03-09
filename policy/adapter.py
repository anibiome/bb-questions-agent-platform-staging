from __future__ import annotations

from datetime import date
from typing import Sequence

from questions_agent_platform.pipeline.selection import Candidate as PipelineCandidate, CandidateSet as PipelineCandidateSet
from questions_agent_platform.policy.types import CandidateItem, CandidateSet


def adapt_candidate_set(
    *,
    pipeline_candidate_set: PipelineCandidateSet,
    user_id: str,
    day: date,
    deterministic_baseline_selected: Sequence[str],
) -> CandidateSet:
    """
    Convert the deterministic candidate set contract (pipeline) into the policy CandidateSet.

    Important: this adapter must not introduce any new item IDs. The policy action space
    is exactly the candidate IDs emitted by the deterministic generator.
    """
    candidates = tuple(_adapt_candidate(c) for c in pipeline_candidate_set.candidates)
    return CandidateSet(
        user_id=str(user_id),
        day=day,
        k_core=int(pipeline_candidate_set.k_core),
        candidates=candidates,
        mandatory_item_ids=tuple(str(i) for i in pipeline_candidate_set.mandatory_item_ids),
        deterministic_baseline_selected=tuple(str(i) for i in deterministic_baseline_selected),
    )


def _adapt_candidate(c: PipelineCandidate) -> CandidateItem:
    return CandidateItem(
        item_id=str(c.item_id),
        item_type=str(c.item_type),
        scale_ids=tuple(str(s) for s in c.scale_ids),
        deterministic_score=float(c.deterministic_score),
        constraint_tags=tuple(str(t) for t in c.constraint_tags),
        reason_codes=tuple(str(r) for r in c.reason_codes),
        features=dict(c.features),
    )

