"""Memory and session management for the Socratic Technical Study Agent.

Handles both:
1. Long-Term Memory: Persistent user profile (`data/user_profile.json`) tracking
   learner persona, learning preferences, and concept mastery with diagnosed misconceptions.
2. Short-Term Memory & Context Compaction: Timestamped session state (`data/sessions/*.json`)
   capturing dialogue history, tool invocations, and session summaries with resumption capability.
   Includes ContextCompactor to manage history bloat and non-blocking async background tasks.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Base directory for the repository
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_PROFILE_PATH = DEFAULT_DATA_DIR / "user_profile.json"
DEFAULT_SESSIONS_DIR = DEFAULT_DATA_DIR / "sessions"

DEFAULT_CONCEPTS = [
    "architecture_overview",
    "scaled_dot_product",
    "multi_head_attention",
    "positional_encoding",
    "computational_complexity",
]

# Dedicated thread pool for async file I/O operations
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="studyagent_io")


def _current_iso_timestamp() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def get_default_profile() -> Dict[str, Any]:
    """Generates a fresh default user profile structure."""
    return {
        "user_persona": {
            "background": "Software engineer studying modern deep learning architectures",
            "learning_style": "Prefers concrete tensor shapes and code examples over abstract equations",
            "personality_tone": "Curious, pragmatic, focused on real-world implementation trade-offs",
        },
        "concept_mastery": {
            concept: {"score": 0, "status": "unseen", "misconceptions": []}
            for concept in DEFAULT_CONCEPTS
        },
        "last_updated": _current_iso_timestamp(),
    }


# =====================================================================
# Context Compactor (Mitigating History Bloat)
# =====================================================================


class ContextCompactor:
    """Manages dialogue history bloat through intelligent context compaction.

    Preserves recent active turns verbatim while summarizing earlier exchanges into
    a dense pedagogical synopsis with key topics, equations, and diagnosed misconceptions.
    """

    def __init__(self, max_turns: int = 6, preserve_recent: int = 4):
        self.max_turns = max_turns
        self.preserve_recent = preserve_recent

    def is_compaction_needed(self, turns: List[Dict[str, Any]]) -> bool:
        """Determines if the current turn count exceeds the threshold."""
        return len(turns) > self.max_turns

    def compact(
        self,
        turns: List[Dict[str, Any]],
        existing_synopsis: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Compacts older turns into a running synopsis and returns (synopsis, recent_turns).

        Args:
            turns: Full list of dialogue turns.
            existing_synopsis: Prior summary from previous compactions.

        Returns:
            Tuple of (updated_synopsis, recent_turns_slice).
        """
        if not self.is_compaction_needed(turns):
            return existing_synopsis, list(turns)

        split_index = len(turns) - self.preserve_recent
        older_turns = turns[:split_index]
        recent_turns = turns[split_index:]

        # Extract topics, queries, and tool usages from older turns
        user_queries = [
            t["content"][:100] for t in older_turns if t.get("role") == "user"
        ]
        tools_used = []
        for t in older_turns:
            tools_used.extend(t.get("tools_invoked", []))
        unique_tools = sorted(list(set(tools_used)))

        compaction_lines = []
        if existing_synopsis:
            compaction_lines.append(f"Prior Context: {existing_synopsis.strip()}")

        compaction_lines.append(
            f"Compacted {len(older_turns)} earlier turns covering inquiries: "
            f"[{' | '.join(user_queries)}]."
        )
        if unique_tools:
            compaction_lines.append(
                f"Referenced tools: {', '.join(unique_tools)}."
            )

        updated_synopsis = " ".join(compaction_lines)
        return updated_synopsis, recent_turns


# =====================================================================
# Long-Term Memory (User Profile - Sync & Async)
# =====================================================================


def load_user_profile(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads the persistent user profile from disk synchronously."""
    profile_path = Path(path) if path else DEFAULT_PROFILE_PATH
    if not profile_path.exists():
        default_prof = get_default_profile()
        save_user_profile(default_prof, profile_path)
        return default_prof

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "user_persona" not in data:
            data["user_persona"] = get_default_profile()["user_persona"]
        if "concept_mastery" not in data:
            data["concept_mastery"] = get_default_profile()["concept_mastery"]
        return data
    except Exception as e:
        logger.warning(
            "Error reading user profile from %s: %s. Using default.", profile_path, e
        )
        return get_default_profile()


async def load_user_profile_async(
    path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Loads the persistent user profile from disk asynchronously in worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_IO_EXECUTOR, load_user_profile, path)


def save_user_profile(
    profile_data: Dict[str, Any], path: Optional[Path | str] = None
) -> Path:
    """Saves the user profile dictionary to disk synchronously."""
    profile_path = Path(path) if path else DEFAULT_PROFILE_PATH
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_data["last_updated"] = _current_iso_timestamp()

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    return profile_path


async def save_user_profile_async(
    profile_data: Dict[str, Any], path: Optional[Path | str] = None
) -> Path:
    """Saves the user profile dictionary to disk asynchronously in worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _IO_EXECUTOR, save_user_profile, profile_data, path
    )


def update_profile_trait(
    trait_category: str,
    detail: str,
    path: Optional[Path | str] = None,
) -> str:
    """Updates a persona trait in long-term memory synchronously."""
    valid_categories = ["background", "learning_style", "personality_tone"]
    norm_cat = trait_category.strip().lower()
    if norm_cat not in valid_categories:
        norm_cat = "background"

    profile = load_user_profile(path)
    current_val = profile["user_persona"].get(norm_cat, "")

    if current_val and current_val not in detail:
        profile["user_persona"][norm_cat] = f"{current_val}; {detail.strip()}"
    else:
        profile["user_persona"][norm_cat] = detail.strip()

    save_user_profile(profile, path)
    return f"Updated user persona trait '{norm_cat}' to: {profile['user_persona'][norm_cat]}"


async def update_profile_trait_async(
    trait_category: str,
    detail: str,
    path: Optional[Path | str] = None,
) -> str:
    """Updates a persona trait in long-term memory asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _IO_EXECUTOR, update_profile_trait, trait_category, detail, path
    )


def record_concept_mastery(
    concept_key: str,
    score: int,
    notes: str = "",
    path: Optional[Path | str] = None,
) -> str:
    """Updates learner score (0-100) and appends misconceptions synchronously."""
    profile = load_user_profile(path)
    clamped_score = max(0, min(100, int(score)))

    status = "unseen"
    if clamped_score >= 80:
        status = "mastered"
    elif clamped_score > 0:
        status = "learning"

    if "concept_mastery" not in profile:
        profile["concept_mastery"] = {}

    entry = profile["concept_mastery"].get(
        concept_key, {"score": 0, "status": "unseen", "misconceptions": []}
    )
    entry["score"] = clamped_score
    entry["status"] = status

    if notes and notes.strip():
        note_str = notes.strip()
        if note_str not in entry.get("misconceptions", []):
            entry.setdefault("misconceptions", []).append(note_str)

    profile["concept_mastery"][concept_key] = entry
    save_user_profile(profile, path)
    return (
        f"Recorded progress for '{concept_key}': score={clamped_score}, "
        f"status={status}, notes={notes or 'None'}"
    )


async def record_concept_mastery_async(
    concept_key: str,
    score: int,
    notes: str = "",
    path: Optional[Path | str] = None,
) -> str:
    """Updates learner score (0-100) and appends misconceptions asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _IO_EXECUTOR, record_concept_mastery, concept_key, score, notes, path
    )


def format_profile_for_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Formats long-term memory into a concise markdown section for system prompt injection."""
    prof = profile or load_user_profile()
    persona = prof.get("user_persona", {})
    mastery = prof.get("concept_mastery", {})

    lines = [
        "### Learner Profile & Context (Long-Term Memory)",
        f"- **Background**: {persona.get('background', 'Unknown')}",
        f"- **Learning Style**: {persona.get('learning_style', 'Adaptive')}",
        f"- **Tone & Goals**: {persona.get('personality_tone', 'Curious and thorough')}",
        "#### Concept Mastery:",
    ]

    for key, data in mastery.items():
        score = data.get("score", 0)
        status = data.get("status", "unseen")
        misc = data.get("misconceptions", [])
        misc_str = f" (Misconceptions: {', '.join(misc)})" if misc else ""
        lines.append(f"  - `{key}`: {score}% [{status}]{misc_str}")

    return "\n".join(lines)


# =====================================================================
# Short-Term Memory (Session Storage - Sync & Async)
# =====================================================================


class SessionManager:
    """Manages short-term conversation logs, session state persistence, and background tasks."""

    def __init__(
        self,
        sessions_dir: Optional[Path | str] = None,
        compactor: Optional[ContextCompactor] = None,
    ):
        self.sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.compactor = compactor or ContextCompactor()
        self._background_tasks: set[asyncio.Task] = set()

    def new_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new session object synchronously."""
        now_str = _current_iso_timestamp()
        if not session_id:
            tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{tag}"

        session_data: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": now_str,
            "updated_at": now_str,
            "turns": [],
            "session_summary": "Newly initialized study session.",
            "compacted_synopsis": "",
        }
        self.save_session(session_data)
        return session_data

    async def new_session_async(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new session object asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_IO_EXECUTOR, self.new_session, session_id)

    def save_session(self, session_data: Dict[str, Any]) -> Path:
        """Persists session state to JSON in the sessions directory synchronously."""
        session_id = session_data.get("session_id", "session_unnamed")
        session_file = self.sessions_dir / f"{session_id}.json"
        session_data["updated_at"] = _current_iso_timestamp()

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        return session_file

    async def save_session_async(self, session_data: Dict[str, Any]) -> Path:
        """Persists session state to JSON asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _IO_EXECUTOR, self.save_session, session_data
        )

    def save_session_background(self, session_data: Dict[str, Any]) -> None:
        """Dispatches session persistence to a non-blocking background async task."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.save_session_async(session_data))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # If no running loop, save synchronously
            self.save_session(session_data)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session by ID synchronously."""
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load session %s: %s", session_id, e)
            return None

    async def load_session_async(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Loads a session by ID asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_IO_EXECUTOR, self.load_session, session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all existing sessions ordered by updated_at descending."""
        sessions = []
        for file in self.sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append(
                        {
                            "session_id": data.get("session_id", file.stem),
                            "created_at": data.get("created_at", ""),
                            "updated_at": data.get("updated_at", ""),
                            "turns_count": len(data.get("turns", [])),
                            "session_summary": data.get("session_summary", ""),
                            "compacted_synopsis": data.get("compacted_synopsis", ""),
                            "file_path": str(file),
                        }
                    )
            except Exception as e:
                logger.warning("Error reading session file %s: %s", file, e)

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    async def list_sessions_async(self) -> List[Dict[str, Any]]:
        """Lists all existing sessions asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_IO_EXECUTOR, self.list_sessions)

    def add_turn(
        self,
        session_data: Dict[str, Any],
        role: str,
        content: str,
        tools_invoked: Optional[List[str]] = None,
        save_background: bool = False,
    ) -> None:
        """Appends a turn, performs compaction if needed, and saves session state."""
        turn: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": _current_iso_timestamp(),
        }
        if tools_invoked:
            turn["tools_invoked"] = tools_invoked
        session_data.setdefault("turns", []).append(turn)

        # Context compaction check
        existing_synopsis = session_data.get("compacted_synopsis", "")
        if self.compactor.is_compaction_needed(session_data["turns"]):
            synopsis, recent_turns = self.compactor.compact(
                turns=session_data["turns"], existing_synopsis=existing_synopsis
            )
            session_data["compacted_synopsis"] = synopsis
            session_data["turns"] = recent_turns

        # Update lightweight summary
        turns = session_data["turns"]
        user_queries = [t["content"] for t in turns if t.get("role") == "user"]
        if user_queries:
            recent_topics = user_queries[-2:]
            session_data["session_summary"] = (
                f"Active discussion covering: {' | '.join(recent_topics)}"
            )

        if save_background:
            self.save_session_background(session_data)
        else:
            self.save_session(session_data)

    async def add_turn_async(
        self,
        session_data: Dict[str, Any],
        role: str,
        content: str,
        tools_invoked: Optional[List[str]] = None,
    ) -> None:
        """Appends a turn and saves session state asynchronously."""
        self.add_turn(
            session_data=session_data,
            role=role,
            content=content,
            tools_invoked=tools_invoked,
            save_background=False,
        )
        await self.save_session_async(session_data)
