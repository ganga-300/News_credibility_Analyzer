from graph.state import AgentState
from utils.helpers import get_groq_client
from utils.prompts import CLAIM_EXTRACTION_PROMPT


def extract_claims_node(state: AgentState):

    client = get_groq_client()

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        text=state["input_text"]
    )

    print("\n" + "=" * 60)
    print("CLAIM EXTRACTION PROMPT")
    print("=" * 60)
    print(prompt)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content

        print("\n" + "=" * 60)
        print("CLAIM EXTRACTION RESPONSE")
        print("=" * 60)
        print(content)

        claims = [
            line.strip("- ").strip()
            for line in content.split("\n")
            if line.strip().startswith("-")
        ]

        if not claims:
            claims = [state["input_text"][:500]]

        print("\nPARSED CLAIMS:")
        for i, c in enumerate(claims, 1):
            print(f"{i}. {c}")

        state["extracted_claims"] = claims

    except Exception as e:
        print("\nERROR:", str(e))
        state["extracted_claims"] = [state["input_text"][:500]]

    return state