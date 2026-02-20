from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .recursive_adic import depth_laplace_weight, recursive_depth
from .storage import SQLiteStore
from .tools.docs import extract_text_from_bytes
from .tools.filesystem import FileBrowser


REPO_KEYWORDS = (
    "recursive",
    "adic",
    "rdt",
    "topological",
    "optimizer",
    "entropy",
    "hyperreal",
    "hyper-real",
    "dual number",
    "dual-number",
)
TEXT_FILE_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".tex",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".ipynb",
}
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".cache",
}
SAFE_DEFAULT_HEURISTICS: dict[str, float] = {
    "retry_attempts": 2.0,
    "docs_priority": 0.62,
    "web_priority": 0.56,
    "planner_confidence": 0.5,
}
MIN_SCORE_FOR_POLICY_UPDATE = 0.72
MAX_HEURISTIC_STEP_DELTA = 0.08
FAILURE_STREAK_FOR_RECOVERY = 3


@dataclass
class DualNumber:
    real: float
    dual: float = 0.0

    def __add__(self, other: object) -> "DualNumber":
        rhs = _as_dual(other)
        return DualNumber(self.real + rhs.real, self.dual + rhs.dual)

    def __sub__(self, other: object) -> "DualNumber":
        rhs = _as_dual(other)
        return DualNumber(self.real - rhs.real, self.dual - rhs.dual)

    def __mul__(self, other: object) -> "DualNumber":
        rhs = _as_dual(other)
        return DualNumber(
            self.real * rhs.real,
            (self.dual * rhs.real) + (self.real * rhs.dual),
        )


@dataclass
class HyperReal:
    standard: float
    infinitesimal: float = 0.0

    def as_float(self) -> float:
        return float(self.standard + self.infinitesimal)


class RecursiveLearningLayer:
    """
    Recursive learning layer that fuses:
    1) task-trace self-reflection signals
    2) Recursive-Adic depth weighting
    3) dual-number gradient proxy
    4) hyperreal infinitesimal policy micro-updates
    5) repo/pdf algorithm source ingestion
    """

    def __init__(
        self,
        store: SQLiteStore,
        files: FileBrowser,
        repo_root: Path | None = None,
        pdf_paths: list[Path] | None = None,
        max_bootstrap_files: int = 160,
    ) -> None:
        self.store = store
        self.files = files
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else self.files.root
        if pdf_paths is None:
            defaults = [
                Path.home() / "Downloads" / "The_Recursive_Adic_Number_Field__Construction__Analysis__and_Recursive_Depth_Transforms (1).pdf",
                Path.home() / "Downloads" / "v.pdf",
            ]
            self.pdf_paths = defaults
        else:
            self.pdf_paths = [Path(p).expanduser().resolve() for p in pdf_paths]
        self.max_bootstrap_files = max(20, int(max_bootstrap_files))
        self.bootstrap_key = "recursive_learning.bootstrap.v1"
        self._bootstrapped = False
        self._last_bootstrap: dict[str, object] = {"status": "not_bootstrapped", "repo_docs": 0, "pdf_docs": 0}

    def run_pipeline(
        self,
        session_id: str,
        query: str,
        force_bootstrap: bool = False,
    ) -> dict[str, object]:
        bootstrap = self.ensure_bootstrap(session_id, force=force_bootstrap)
        look_for = query.strip() or "recursive adic dual hyperreal optimizer"
        hits = self.store.search_chunks(look_for, limit=8)

        lines = [f"Recursive learning pipeline query: {look_for}"]
        lines.append(
            f"Bootstrap status: {bootstrap.get('status')} | repo_docs={bootstrap.get('repo_docs', 0)} | "
            f"pdf_docs={bootstrap.get('pdf_docs', 0)}"
        )
        provenance: list[dict[str, object]] = []
        for hit in hits:
            source = str(hit.get("source") or hit.get("doc_name") or "")
            snippet = str(hit.get("text") or "")[:300]
            lines.append(
                f"- {hit.get('doc_name')}: score={float(hit.get('score') or 0.0):.3f} "
                f"depth={float(hit.get('radf_depth') or 0.0):.2f}"
            )
            provenance.append(
                {
                    "source_type": "doc",
                    "source": source,
                    "snippet": snippet,
                    "doc_id": str(hit.get("doc_id") or ""),
                    "url": "",
                    "path": source if source.startswith("/") else "",
                }
            )

        detail = (
            f"Recursive learning pipeline ready with {len(hits)} retrieved sources and "
            f"{bootstrap.get('repo_docs', 0)} repo docs fused."
        )
        return {
            "status": "ok",
            "detail": detail,
            "context_block": "\n".join(lines),
            "provenance": provenance,
            "bootstrap": bootstrap,
        }

    def ensure_bootstrap(self, session_id: str, force: bool = False) -> dict[str, object]:
        if not force and self._bootstrapped:
            return dict(self._last_bootstrap)
        if not force and self._is_bootstrapped(session_id):
            payload = {"status": "already_bootstrapped", "repo_docs": 0, "pdf_docs": 0}
            self._bootstrapped = True
            self._last_bootstrap = dict(payload)
            return payload

        payload = self.bootstrap_original_sources(session_id=session_id, max_files=self.max_bootstrap_files)
        self.store.upsert_preference(
            session_id,
            self.bootstrap_key,
            f"done:{int(time.time())}",
            source="recursive-learning-layer",
        )
        self.store.upsert_fact(
            session_id,
            "recursive_learning_bootstrap_repo_docs",
            str(payload.get("repo_docs", 0)),
            source="recursive-learning-layer",
        )
        self.store.upsert_fact(
            session_id,
            "recursive_learning_bootstrap_pdf_docs",
            str(payload.get("pdf_docs", 0)),
            source="recursive-learning-layer",
        )
        self._bootstrapped = True
        self._last_bootstrap = dict(payload)
        return payload

    def bootstrap_original_sources(self, session_id: str, max_files: int = 160) -> dict[str, object]:
        max_files = max(20, int(max_files))
        repo_docs = 0
        pdf_docs = 0
        scanned_files = 0
        provenance: list[dict[str, object]] = []

        if self.repo_root.exists() and self.repo_root.is_dir():
            candidate_paths = self._candidate_repo_files(self.repo_root, max_files=max_files * 3)
            for path in candidate_paths:
                if repo_docs >= max_files:
                    break
                scanned_files += 1
                text = _safe_read_text(path, max_chars=24000)
                if len(text) < 60:
                    continue
                if not _contains_keywords(text, REPO_KEYWORDS):
                    continue
                rel = str(path)
                try:
                    rel = str(path.relative_to(self.repo_root))
                except Exception:
                    pass
                doc_id = self.store.add_document(
                    name=f"repo-algo:{rel}"[:180],
                    source=str(path),
                    kind="repo-algorithm",
                    text=text,
                )
                repo_docs += 1
                if len(provenance) < 20:
                    provenance.append(
                        {
                            "source_type": "file",
                            "source": str(path),
                            "snippet": text[:280],
                            "doc_id": doc_id,
                            "path": str(path),
                            "url": "",
                        }
                    )
                self.store.add_artifact(
                    session_id=session_id,
                    artifact_type="repo-algorithm",
                    location=str(path),
                    source=str(path),
                    doc_id=doc_id,
                    description=f"recursive-learning source file: {rel}"[:220],
                )

        for path in self.pdf_paths:
            if not path.exists() or not path.is_file():
                continue
            try:
                data = path.read_bytes()
                text, kind = extract_text_from_bytes(path.name, data)
            except Exception:
                continue
            if not text.strip():
                continue
            doc_id = self.store.add_document(
                name=f"paper:{path.name}"[:180],
                source=str(path),
                kind=f"recursive-learning-{kind}",
                text=text[:180000],
            )
            pdf_docs += 1
            if len(provenance) < 24:
                provenance.append(
                    {
                        "source_type": "doc",
                        "source": str(path),
                        "snippet": text[:280],
                        "doc_id": doc_id,
                        "path": str(path),
                        "url": "",
                    }
                )
            self.store.add_artifact(
                session_id=session_id,
                artifact_type="paper",
                location=str(path),
                source=str(path),
                doc_id=doc_id,
                description="recursive-learning paper ingestion"[:220],
            )

        return {
            "status": "bootstrapped",
            "repo_docs": repo_docs,
            "pdf_docs": pdf_docs,
            "scanned_files": scanned_files,
            "provenance": provenance,
        }

    def adapt(
        self,
        session_id: str,
        task_id: str,
        user_text: str,
        strict_facts: bool,
        evidence_mode: bool,
        plan: list[dict[str, object]],
        tool_calls: list[dict[str, object]],
        provenance: list[dict[str, object]],
        evidence: list[dict[str, object]],
        heuristics: dict[str, float],
        baseline_score: float,
    ) -> dict[str, object]:
        bootstrap = self.ensure_bootstrap(session_id, force=False)

        total_steps = max(1, len(plan))
        done_steps = sum(1 for s in plan if str(s.get("status") or "") == "done")
        tool_errors = sum(1 for c in tool_calls if str(c.get("status") or "") == "error")
        provenance_count = len(provenance)
        evidence_pairs = 0
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if (item.get("sources") or []) and (item.get("snippets") or []):
                evidence_pairs += 1

        step_completion = done_steps / max(1, total_steps)
        tool_error_rate = tool_errors / max(1, len(tool_calls))
        evidence_density = evidence_pairs / max(1, len(evidence))

        complexity_index = max(1, min(96, len(plan) + len(tool_calls) + provenance_count + 1))
        depth = recursive_depth(complexity_index, alpha=self.store.radf_alpha)
        radf_weight = depth_laplace_weight(complexity_index, beta=self.store.radf_beta, alpha=self.store.radf_alpha)

        repo_hits = self.store.search_chunks("recursive adic dual hyperreal topological optimizer", limit=8)
        repo_signal_raw = sum(float(h.get("score") or 0.0) for h in repo_hits)
        repo_signal = _clamp(repo_signal_raw / max(1.0, 8.0), 0.0, 4.0)
        repo_signal_norm = _clamp(repo_signal / 4.0, 0.0, 1.0)

        x = DualNumber(_clamp(float(baseline_score), 0.0, 1.0), 1.0)
        quality = x * DualNumber(float(radf_weight), 0.0)
        regularizer = DualNumber((0.25 * evidence_density) + (0.20 * step_completion), 0.0)
        penalty = DualNumber((0.6 * tool_error_rate) + (0.15 if strict_facts and provenance_count == 0 else 0.0), 0.0)
        objective = quality + regularizer - penalty
        dual_grad = objective.dual

        epsilon_base = 1e-4 * (1.0 + min(3.0, float(depth)))
        hyper = HyperReal(
            standard=0.08 * dual_grad * (1.0 - tool_error_rate),
            infinitesimal=epsilon_base * (0.3 + repo_signal_norm + evidence_density),
        )
        hyper_delta = hyper.as_float()

        old_retry = _clamp(float(heuristics.get("retry_attempts", 2.0)), 1.0, 4.0)
        old_docs = _clamp(float(heuristics.get("docs_priority", 0.62)), 0.1, 0.95)
        old_web = _clamp(float(heuristics.get("web_priority", 0.56)), 0.1, 0.95)
        old_conf = _clamp(float(heuristics.get("planner_confidence", 0.5)), 0.0, 1.0)

        docs_gap = 1.0 if (strict_facts and provenance_count < 2) else 0.0
        new_docs = _clamp(
            old_docs
            + (0.06 * dual_grad * (1.0 + docs_gap))
            + (0.02 * repo_signal_norm)
            + (0.40 * hyper_delta),
            0.1,
            0.95,
        )
        new_web = _clamp(
            old_web
            + (0.03 if provenance_count == 0 else -0.02 * evidence_density)
            + (0.20 * hyper_delta),
            0.1,
            0.95,
        )
        new_retry = _clamp(
            old_retry
            + (0.60 * tool_error_rate)
            - (0.10 if tool_error_rate == 0 else 0.0)
            + abs(hyper_delta),
            1.0,
            4.0,
        )
        new_conf = _clamp(
            old_conf
            + (0.18 * (float(baseline_score) - 0.5))
            + (0.05 * dual_grad)
            - (0.15 * tool_error_rate)
            + (0.10 * hyper_delta),
            0.0,
            1.0,
        )

        blocked_reasons: list[str] = []
        base_score = _clamp(float(baseline_score), 0.0, 1.0)
        if base_score < MIN_SCORE_FOR_POLICY_UPDATE:
            blocked_reasons.append("low_success_score")
        if step_completion < 0.5:
            blocked_reasons.append("low_step_completion")
        if tool_error_rate > 0.34:
            blocked_reasons.append("high_tool_error_rate")
        if strict_facts and provenance_count <= 0:
            blocked_reasons.append("no_provenance_in_strict_mode")
        if evidence_mode and evidence_pairs <= 0:
            blocked_reasons.append("no_evidence_pairs")
        if strict_facts and evidence_mode and provenance_count < 2:
            blocked_reasons.append("insufficient_sources_for_strict_evidence")

        allow_policy_update = len(blocked_reasons) == 0
        failure_streak = self._session_fact_int(session_id, "recursive_learning_failure_streak", 0)
        success_streak = self._session_fact_int(session_id, "recursive_learning_success_streak", 0)
        if allow_policy_update:
            success_streak += 1
            failure_streak = 0
        else:
            failure_streak += 1
            success_streak = 0
        self.store.upsert_fact(
            session_id,
            "recursive_learning_failure_streak",
            str(failure_streak),
            source="recursive-learning-layer",
        )
        self.store.upsert_fact(
            session_id,
            "recursive_learning_success_streak",
            str(success_streak),
            source="recursive-learning-layer",
        )

        bounded_docs = _bounded_step(old_docs, new_docs, MAX_HEURISTIC_STEP_DELTA, 0.1, 0.95)
        bounded_web = _bounded_step(old_web, new_web, MAX_HEURISTIC_STEP_DELTA, 0.1, 0.95)
        bounded_retry = _bounded_step(old_retry, new_retry, MAX_HEURISTIC_STEP_DELTA, 1.0, 4.0)
        bounded_conf = _bounded_step(old_conf, new_conf, MAX_HEURISTIC_STEP_DELTA, 0.0, 1.0)

        candidate_updates: dict[str, dict[str, float]] = {}
        applied_updates: dict[str, dict[str, float]] = {}
        for key, old_value, next_value in (
            ("docs_priority", old_docs, bounded_docs),
            ("web_priority", old_web, bounded_web),
            ("retry_attempts", old_retry, bounded_retry),
            ("planner_confidence", old_conf, bounded_conf),
        ):
            if abs(next_value - old_value) < 0.005:
                continue
            update = {
                "old": round(old_value, 4),
                "new": round(next_value, 4),
                "delta": round(next_value - old_value, 4),
            }
            candidate_updates[key] = update
            if allow_policy_update:
                self.store.upsert_planning_heuristic(key, next_value, source="recursive-learning-layer")
                applied_updates[key] = update

        recovery_updates: dict[str, dict[str, float]] = {}
        if not allow_policy_update and failure_streak >= FAILURE_STREAK_FOR_RECOVERY:
            for key, old_value, low, high in (
                ("docs_priority", old_docs, 0.1, 0.95),
                ("web_priority", old_web, 0.1, 0.95),
                ("retry_attempts", old_retry, 1.0, 4.0),
                ("planner_confidence", old_conf, 0.0, 1.0),
            ):
                safe_value = float(SAFE_DEFAULT_HEURISTICS.get(key, old_value))
                recovered = _clamp(old_value + (0.35 * (safe_value - old_value)), low, high)
                if abs(recovered - old_value) < 0.005:
                    continue
                self.store.upsert_planning_heuristic(key, recovered, source="recursive-learning-recovery")
                recovery_updates[key] = {
                    "old": round(old_value, 4),
                    "new": round(recovered, 4),
                    "delta": round(recovered - old_value, 4),
                    "target_default": round(safe_value, 4),
                }

        novelty_score = _clamp(
            0.45 * repo_signal_norm
            + 0.20 * float(radf_weight)
            + 0.20 * min(1.0, abs(hyper_delta) * 10.0)
            + 0.15 * min(1.0, abs(dual_grad)),
            0.0,
            1.0,
        )
        if allow_policy_update:
            rule = (
                "Recursive learning update applied: fused Recursive-Adic depth weighting with dual-number gradient "
                f"({dual_grad:.4f}) and hyperreal micro-step ({hyper_delta:.6f}), novelty={novelty_score:.3f}."
            )
        else:
            reason = ",".join(blocked_reasons) if blocked_reasons else "safety_gate"
            rule = (
                "Recursive learning update blocked by policy gate to avoid reinforcing low-quality traces "
                f"(reasons={reason}); failure_streak={failure_streak}, novelty={novelty_score:.3f}."
            )
        trigger = (
            f"recursive-learning|strict={int(strict_facts)}|evidence={int(evidence_mode)}|"
            f"prov={provenance_count}|tool_errors={tool_errors}"
        )
        confidence = _clamp(0.45 + 0.4 * base_score + 0.15 * novelty_score, 0.05, 0.99)
        if not allow_policy_update:
            confidence = _clamp(confidence * 0.55, 0.05, 0.75)
        rule_id = self.store.add_improvement_rule(
            session_id=session_id,
            task_id=task_id,
            rule=rule[:500],
            trigger=trigger[:120],
            confidence=round(confidence, 3),
        )

        summary = (
            f"recursive_learning_v1 score={float(baseline_score):.3f} gate={'open' if allow_policy_update else 'closed'} "
            f"depth={depth:.3f} dual_grad={dual_grad:.4f} hyper_delta={hyper_delta:.6f}"
        )
        self.store.upsert_preference(
            session_id,
            "policy.recursive_learning",
            summary[:300],
            source="recursive-learning-layer",
        )
        self.store.upsert_fact(
            session_id,
            "recursive_learning_novelty_score",
            f"{novelty_score:.3f}",
            source="recursive-learning-layer",
        )
        self.store.upsert_fact(
            session_id,
            "recursive_learning_last_depth",
            f"{depth:.3f}",
            source="recursive-learning-layer",
        )

        event_id = self.store.add_recursive_learning_event(
            session_id=session_id,
            task_id=task_id,
            layer="recursive-learning-v1",
            score=float(baseline_score),
            depth=float(depth),
            dual_grad=float(dual_grad),
            hyper_delta=float(hyper_delta),
            heuristic_updates={
                "candidate": candidate_updates,
                "applied": applied_updates,
                "recovery": recovery_updates,
            },
            metrics={
                "step_completion": round(step_completion, 4),
                "tool_error_rate": round(tool_error_rate, 4),
                "provenance_count": provenance_count,
                "evidence_pairs": evidence_pairs,
                "repo_signal_norm": round(repo_signal_norm, 4),
                "novelty_score": round(novelty_score, 4),
                "allow_policy_update": allow_policy_update,
                "blocked_reasons": blocked_reasons,
                "failure_streak": failure_streak,
                "success_streak": success_streak,
                "query_preview": user_text[:120],
            },
        )

        return {
            "layer": "recursive-learning-v1",
            "bootstrap": bootstrap,
            "dual_number": {
                "objective_real": round(objective.real, 6),
                "objective_dual_grad": round(dual_grad, 6),
            },
            "hyper_real": {
                "standard": round(hyper.standard, 8),
                "infinitesimal": round(hyper.infinitesimal, 8),
                "delta": round(hyper_delta, 8),
            },
            "radf": {
                "complexity_index": complexity_index,
                "depth": round(depth, 6),
                "weight": round(radf_weight, 6),
            },
            "repo_signal_norm": round(repo_signal_norm, 6),
            "novelty_score": round(novelty_score, 6),
            "policy_gate": {
                "allow_update": allow_policy_update,
                "blocked_reasons": blocked_reasons,
                "failure_streak": failure_streak,
                "success_streak": success_streak,
            },
            "heuristic_updates": applied_updates,
            "candidate_updates": candidate_updates,
            "recovery_updates": recovery_updates,
            "improvement_rule_id": rule_id,
            "event_id": event_id,
            "summary": summary,
        }

    def _is_bootstrapped(self, session_id: str) -> bool:
        for item in self.store.list_preferences(session_id, limit=500):
            if item.key == self.bootstrap_key and str(item.value).startswith("done:"):
                return True
        return False

    def _session_fact_int(self, session_id: str, key: str, default: int = 0) -> int:
        for item in self.store.list_facts(session_id, limit=500):
            if item.key != key:
                continue
            try:
                return int(float(item.value))
            except Exception:
                return default
        return default

    def _candidate_repo_files(self, root: Path, max_files: int) -> list[Path]:
        out: list[Path] = []
        roots_to_walk = self._discover_repo_roots(root)

        for repo in roots_to_walk:
            for dirpath, dirnames, filenames in os.walk(repo):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")]
                base = Path(dirpath)
                for name in filenames:
                    if len(out) >= max_files:
                        return out
                    path = base / name
                    if path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                        continue
                    out.append(path)
        return out

    def _discover_repo_roots(self, root: Path) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []

        out: list[Path] = []
        if (root / ".git").exists():
            out.append(root)

        try:
            children = [p for p in sorted(root.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]
        except Exception:
            children = []

        for child in children:
            if child.name in SKIP_DIR_NAMES or child.name.startswith("."):
                continue
            if (child / ".git").exists():
                out.append(child)

        if not out:
            out.append(root)
        return out[:24]


def _as_dual(value: object) -> DualNumber:
    if isinstance(value, DualNumber):
        return value
    return DualNumber(float(value), 0.0)


def _safe_read_text(path: Path, max_chars: int = 24000) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)[:max_chars]
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")[:max_chars]


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    low = re.sub(r"\s+", " ", text.lower())
    return any(k in low for k in keywords)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _bounded_step(current: float, target: float, max_delta: float, low: float, high: float) -> float:
    delta = _clamp(target - current, -abs(max_delta), abs(max_delta))
    return _clamp(current + delta, low, high)
