
from __future__ import annotations

import io
import json

from PIL import Image
from ollama import Client

from aura.config import settings


class DeadlockScreenAnalyser:
    """
    Uses a local vision-language model to extract structured
    gameplay information from Deadlock screenshots.
    """

    
    SYSTEM_PROMPT = """
    You are AURA's visual perception system for Valve's Deadlock.

    Your task is to extract accurate gameplay information
    from screenshots.

    CRITICAL: DISTINGUISH WORLD CONTENT FROM HUD CONTENT.

    The screenshot contains two different visual layers:

    1. GAME WORLD:
    - The 3D environment.
    - The player's character.
    - Physically visible friendly and enemy characters.
    - World objects, structures and objectives.

    2. HUD / INTERFACE:
    - Character portraits.
    - Team indicators.
    - Health bars.
    - Ability icons.
    - Item icons.
    - Currency and statistics.
    - Minimap and scoreboard.

    Never identify a HUD portrait as a physically
    present character in the game world.

    ENTITY DETECTION:

    1. Only classify a character as a visible world entity
    if its actual character model is visible in the
    3D environment.
    2. Do not count character portraits at the top of
    the screen as physically present enemies.
    3. Identify the player's own character separately.
    4. Distinguish enemy and friendly characters only
    when supported by visible evidence.
    5. If a character's identity is uncertain, use unknown.
    6. Do not invent additional characters.

    PLAYER HUD:

    1. Read health and maximum health from the HUD.
    2. Read currency only when its value is legible.
    3. Identify visible abilities and their displayed
    cooldowns when possible.
    4. Never confuse shop prices with player currency.
    5. Return null when a value cannot be read reliably.

    MINIMAP:

    1. Identify whether a minimap is actually visible.
    2. Do not interpret unrelated HUD elements as a minimap.
    3. Only report positions supported by visible evidence.

    CONFIDENCE:

    1. Never automatically assign confidence 1.0.
    2. Use lower confidence when text or visual details
    are ambiguous.
    3. Distinguish clearly visible information from inference.
    4. Record important uncertainties in unknowns.
    5. Do not claim certainty about information you cannot see.

    GENERAL RULES:

    - Never invent abilities, items, characters or objectives.
    - Do not infer events outside the screenshot.
    - Do not provide gameplay advice.
    - This component extracts observations only.
    - Use null for unreadable values.
    - Use empty arrays when nothing is visible.

    Return valid JSON only, following the requested schema.
    Do not use Markdown or code fences.
    """


    def __init__(self):
        self.client = Client(host=settings.ollama_host)
        self.model = "qwen3-vl:8b-instruct"

        print(f"Initialising Deadlock visual analyser: {self.model}")

    def analyse(self, image: Image.Image) -> dict:
        """
        Analyse a PIL screenshot and return structured observations.
        The image is transmitted as in-memory PNG bytes.
        """

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()

        
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Analyse this Deadlock screenshot. "
                        "Extract only the information visible "
                        "in the image."
                    ),
                    "images": [image_bytes],
                },
            ],
            format="json",
            options={
                "num_predict": 1024,
                "temperature": 0.1,
            },
            stream=False,
        )


        
        content = response.message.content or ""

        if not content.strip():
            raise RuntimeError(
                "The vision model returned no final content. "
                f"Done reason: {getattr(response, 'done_reason', None)}"
            )

        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"The vision model returned invalid JSON: {content}"
            ) from error

        

        return result
