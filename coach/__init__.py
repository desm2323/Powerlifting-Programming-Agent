"""coach — an intelligent powerlifting programming agent."""

# Load .env early so OPENAI_API_KEY (and friends) are visible before any
# coach.* module reads os.environ. Silently skip if python-dotenv isn't
# installed — the agent still works offline.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .loop import Agent

__all__ = ["Agent"]
