"""Stage 7 optional adapter: Muse Image background clean-up (P2, cut first). Interface only."""
from app.models.base import AdapterNotVerified


def clean_background(cutout_path: str) -> str | None:
    """Return a cleaned image path, or None if the guard rejects the edit."""
    raise AdapterNotVerified("Muse Image clean-up is P2 and not implemented")
