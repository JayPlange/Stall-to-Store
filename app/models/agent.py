"""Stage 11 adapter: Stage B WhatsApp ordering agent (P2). Interface only.

When built, tools call app.orders.service; the model never sees or sets prices directly.
"""
from app.models.base import AdapterNotVerified
from app.models.types import AgentReply, Msg, ToolSpec


def agent_turn(history: list[Msg], tools: list[ToolSpec]) -> AgentReply:
    raise AdapterNotVerified("Stage B agent is P2 and not implemented")
