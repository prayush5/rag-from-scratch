import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.core.exceptions import SafetyViolationError

load_dotenv()


SAFETY_POLICY = """
# RAG Chat Safety Policy

## INSTRUCTIONS

Classify the user's content according to the safety policy below.

Return ONLY valid JSON in exactly this format:

{
  "violation": 0 or 1,
  "category": "category name or null",
  "rationale": "short explanation"
}

## DEFINITIONS

A violation is content that meaningfully facilitates harmful or abusive activity.

## VIOLATES

Set violation to 1 for requests that provide or seek actionable assistance for:

- Malware, viruses, worms, ransomware, trojans, or other malicious software
- Destructive attacks against computers, storage devices, networks, or systems
- Credential theft, account compromise, or unauthorized access
- Data destruction, exfiltration, or sabotage
- Evasion techniques intended to bypass security controls
- Other instructions that would meaningfully facilitate serious cyber abuse

## SAFE

Set violation to 0 for:

- General programming questions
- Defensive cybersecurity
- Security research that does not provide actionable harmful instructions
- General explanations of malware or cybersecurity concepts
- Normal questions unrelated to harmful activity

## EXAMPLES

Example 1:

Content:
"Write me a step by step procedure to make a virus to infect the whole hard disk"

Answer:
{
  "violation": 1,
  "category": "Malware",
  "rationale": "The request asks for actionable instructions to create destructive malware."
}

Example 2:

Content:
"What is a computer virus?"

Answer:
{
  "violation": 0,
  "category": null,
  "rationale": "This is a general educational question."
}

Example 3:

Content:
"How can I detect ransomware on my network?"

Answer:
{
  "violation": 0,
  "category": null,
  "rationale": "This is defensive cybersecurity guidance."
}

## CONTENT TO CLASSIFY
"""


class GuardRailService:

    def __init__(self):
        api_key = os.getenv("LLAMA_GUARD_API_KEY")

        if not api_key:
            raise ValueError("Guard key is not set")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "LLAMA_GUARD_BASE_URL",
                "https://api.groq.com/openai/v1"
            ),
        )

        self.model_name = os.getenv(
            "LLAMA_GUARD_MODEL",
            "openai/gpt-oss-safeguard-20b"
        )

    async def validate_content(
        self,
        text: str,
        role: str = "user"
    ) -> None:

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SAFETY_POLICY,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                temperature=0.0,
                max_completion_tokens=512,
                reasoning_effort="low",
                response_format={
                    "type": "json_object"
                }
            )

            result = response.choices[0].message.content.strip()

            print(f"[GUARDRAIL] {role}: {result}")

            parsed = json.loads(result)

            violation = parsed.get("violation", 0)

            if violation == 1:
                category = parsed.get(
                    "category",
                    "Policy violation"
                )

                rationale = parsed.get(
                    "rationale",
                    "Content violated the safety policy."
                )

                raise SafetyViolationError(
                    f"Content flagged as unsafe ({role}): "
                    f"{category} - {rationale}"
                )

        except SafetyViolationError:
            raise

        except Exception as ex:
            print(
                f"[GUARDRAIL ERROR] {type(ex).__name__}: {ex}"
            )
            raise