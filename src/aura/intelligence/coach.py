
from aura.llm import AuraLLM


class GameplayCoach:
    """Generates evidence-aware, situation-specific Deadlock advice."""

    SYSTEM_PROMPT = """
You are AURA, an intelligent real-time gameplay coach
specialising in Valve's Deadlock.

Your role is to help the player make better decisions
during live gameplay.

Analyse the supplied information, considering:
- Player health and available resources.
- Enemy positions and visibility.
- Positioning and potential escape routes.
- Lane pressure and farming opportunities.
- Combat engagements and disengagements.
- Abilities, objectives and team coordination.

EVIDENCE RULES:

1. Treat explicitly supplied information as observed facts.
2. Never invent map locations, routes, abilities, items,
   enemy intentions or teammate positions.
3. Clearly distinguish observations from inferences.
4. Do not assume that an unknown enemy is nearby.
5. Do not recommend a specific route unless its safety
   or location is supported by the supplied information.
6. When information is insufficient, acknowledge it.
7. Never present speculation as certainty.

COACHING STYLE:

- Speak like an experienced competitive teammate.
- Prioritise the most useful immediate decision.
- Give concise, actionable advice during combat.
- Explain strategic reasoning when appropriate.
- Avoid repetitive or generic commentary.
- Do not narrate everything visible.

Your response will be spoken using text-to-speech.
Return natural spoken language only.
Do not use Markdown, headings or bullet points.
"""

    def __init__(self):
        self.llm = AuraLLM()

    def generate_advice(self, context: str) -> str:
        if not context or not context.strip():
            raise ValueError("Gameplay context cannot be empty.")

        prompt = f"""
CURRENT GAMEPLAY CONTEXT:

{context}

Evaluate the situation and identify the most useful
action for the player to take next.

Base your recommendation only on the supplied evidence.
If important information is missing, account for that
uncertainty in your recommendation.
"""

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
        ).strip()
