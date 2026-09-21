"""Rich interactive terminal interface for the Socratic Technical Study Agent.

Features:
- Startup session history inspection and resumption menu.
- Beautiful markdown rendering with code syntax highlighting.
- Real-time tool execution indicators and telemetry tracing.
- Graceful exit and session persistence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from studyagent.agent import SocraticStudyAgent
from studyagent.memory import SessionManager, load_user_profile
from studyagent.telemetry import TelemetryTracer

logging.basicConfig(level=logging.ERROR)
console = Console()


def print_banner() -> None:
    """Renders the introductory welcome banner."""
    welcome_text = """[bold cyan]🎓 SOCRATIC TECHNICAL STUDY AGENT[/bold cyan]
[dim]Active-Recall AI Research Mentor | Powered by Google ADK 2.0 & Gemini[/dim]

[italic yellow]"To understand is to invent." — Jean Piaget[/italic yellow]
Master 'Attention Is All You Need' and modern Transformer architectures through Socratic dialogue.
Type [bold green]'exit'[/bold green] or [bold green]'quit'[/bold green] to end your session. Use [bold green]'--trace'[/bold green] for tool transparency."""
    console.print(Panel(welcome_text, border_style="cyan", padding=(1, 2)))


def display_profile_summary() -> None:
    """Displays the current persistent user profile and concept mastery table."""
    profile = load_user_profile()
    persona = profile.get("user_persona", {})
    mastery = profile.get("concept_mastery", {})

    console.print("\n[bold magenta]👤 Learner Profile & Long-Term Memory[/bold magenta]")
    console.print(f"• [bold]Background:[/bold] {persona.get('background', 'Unknown')}")
    console.print(f"• [bold]Learning Style:[/bold] {persona.get('learning_style', 'Adaptive')}")
    console.print(f"• [bold]Personality / Tone:[/bold] {persona.get('personality_tone', 'Curious')}")

    table = Table(title="Concept Mastery (Active Recall Progress)", border_style="magenta")
    table.add_column("Concept", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Diagnosed Misconceptions / Notes", style="dim")

    for concept_key, data in mastery.items():
        score = data.get("score", 0)
        status = data.get("status", "unseen")
        misc = ", ".join(data.get("misconceptions", [])) or "None"
        color = "green" if score >= 80 else ("yellow" if score > 0 else "dim")
        table.add_row(concept_key, f"[{color}]{score}%[/{color}]", status, misc)

    console.print(table)
    console.print()


def prompt_session_choice(session_manager: SessionManager) -> str:
    """Presents existing sessions and prompts user to resume or create new."""
    sessions = session_manager.list_sessions()
    if not sessions:
        new_sess = session_manager.new_session()
        return new_sess["session_id"]

    table = Table(title="📚 Previous Study Sessions", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Session ID", style="cyan")
    table.add_column("Last Active", style="yellow")
    table.add_column("Turns", justify="right", style="green")
    table.add_column("Topic Summary", style="white")

    for idx, s in enumerate(sessions[:5], start=1):
        table.add_row(
            str(idx),
            s["session_id"],
            s.get("updated_at", "")[:19].replace("T", " "),
            str(s.get("turns_count", 0)),
            s.get("session_summary", "")[:50] + ("..." if len(s.get("session_summary", "")) > 50 else ""),
        )

    console.print(table)
    console.print("[dim]Select an option:[/dim]")
    console.print(" [bold green]1[/bold green] Resume latest session")
    console.print(" [bold green]2[/bold green] Start a brand new session")
    if len(sessions) > 1:
        console.print(f" [bold green]3-{min(len(sessions), 5)}[/bold green] Select specific past session")

    try:
        choice = Prompt.ask(
            "\n[bold yellow]Your choice[/bold yellow]",
            default="1",
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Selection cancelled. Goodbye![/yellow]")
        sys.exit(0)

    if choice == "2":
        new_sess = session_manager.new_session()
        console.print(f"[green]✓ Started new study session: [bold]{new_sess['session_id']}[/bold][/green]\n")
        return new_sess["session_id"]

    try:
        num = int(choice)
        if 1 <= num <= min(len(sessions), 5):
            chosen = sessions[num - 1]
            console.print(f"[green]✓ Resumed session: [bold]{chosen['session_id']}[/bold][/green]\n")
            return chosen["session_id"]
    except ValueError:
        pass

    # Default to latest
    chosen = sessions[0]
    console.print(f"[green]✓ Resumed latest session: [bold]{chosen['session_id']}[/bold][/green]\n")
    return chosen["session_id"]


async def run_chat_loop(
    session_id: str,
    model_name: Optional[str] = None,
    trace_enabled: bool = False,
) -> None:
    """Executes the interactive terminal REPL."""
    session_manager = SessionManager()
    TelemetryTracer.get_instance(trace_enabled=trace_enabled)
    agent = SocraticStudyAgent(
        model_name=model_name,
        session_manager=session_manager,
        trace_enabled=trace_enabled,
    )

    # If resuming a session with past turns, print a welcoming recap
    past_session = session_manager.load_session(session_id)
    if past_session and past_session.get("turns"):
        turns = past_session["turns"]
        console.print(
            Panel(
                f"[bold cyan]Resuming session {session_id}[/bold cyan]\n"
                f"[dim]Loaded {len(turns)} past dialogue turns. "
                f"Summary: {past_session.get('session_summary', 'N/A')}[/dim]",
                border_style="dim",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]Ready for study![/bold green] Ask me about the architecture of 'Attention Is All You Need', "
                "query-key-value intuition, softmax scaling, positional encodings, or modern implementations like FlashAttention.",
                border_style="green",
            )
        )

    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold green]Learner[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting study session. Saving state... Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            console.print("\n[bold cyan]Great work studying today! Session progress saved.[/bold cyan]")
            break

        if user_input.lower() in ["profile", "/profile"]:
            display_profile_summary()
            continue

        # Process agent response with live feedback
        console.print()
        response_text = ""
        tools_used = []

        with console.status("[bold cyan]Agent is thinking & reflecting...[/bold cyan]", spinner="dots"):
            try:
                async for event in agent.send_message_stream(
                    message=user_input, session_id=session_id
                ):
                    if event["type"] == "tool_call":
                        tool_name = event["name"]
                        tools_used.append(tool_name)
                        if not trace_enabled:
                            console.print(f"[dim cyan]  ⚙ Executing tool: [italic]{tool_name}[/italic][/dim cyan]")
                    elif event["type"] == "final_response":
                        response_text = event["text"]
            except Exception as e:
                console.print(f"[bold red]Error running agent: {e}[/bold red]")
                continue

        # Render assistant output in beautiful Markdown
        console.print("[bold blue]Agent (Socratic Mentor)[/bold blue]")
        console.print(Markdown(response_text))
        console.print()


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="Socratic Technical Study Agent for 'Attention Is All You Need'"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable real-time telemetry tracing of tool calls, latencies, and token estimates",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new session immediately without showing past sessions menu",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Resume a specific session by ID",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Gemini model name (default: gemini-3.5-flash)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Display current user profile and mastery progress, then exit",
    )

    args = parser.parse_args()

    if args.profile:
        display_profile_summary()
        sys.exit(0)

    print_banner()

    session_manager = SessionManager()

    try:
        if args.session:
            session_id = args.session
        elif args.new:
            new_sess = session_manager.new_session()
            session_id = new_sess["session_id"]
            console.print(f"[green]✓ Started new study session: [bold]{session_id}[/bold][/green]\n")
        else:
            session_id = prompt_session_choice(session_manager)

        asyncio.run(
            run_chat_loop(
                session_id=session_id,
                model_name=args.model,
                trace_enabled=args.trace,
            )
        )
    except (KeyboardInterrupt, EOFError, SystemExit):
        console.print("\n[dim]Session terminated. Goodbye![/dim]")


if __name__ == "__main__":
    main()
