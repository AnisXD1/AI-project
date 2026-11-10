import json
import os
import re
from typing import List, Sequence, Tuple

from openai import OpenAI


MODEL_NAME = "gpt-4o-mini"


def score(candidate: str, target: str) -> float:
    if not target:
        return 1.0 if candidate == target else 0.0
    matches = sum(
        guessed == goal for guessed, goal in zip(candidate, target)
    )
    return matches / len(target)


def parse_candidates(raw_text: str) -> List[str]:
    text = raw_text.strip()

    # Try a JSON array first: ["abc", "def"]
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    except json.JSONDecodeError:
        pass

    # Try extracting quoted strings from plain text
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    items = []
    for a, b in matches:
        items.append(a or b)
    if items:
        return items

    # Fallback: split lines and keep non-empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def require_api_key() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Set it with: export OPENAI_API_KEY=your_api_key_here"
        )
    return OpenAI(api_key=api_key)


def ask_model_for_candidates(
    client: OpenAI,
    target: str,
    parents: Sequence[str],
    scores: Sequence[float],
    size: int,
    temperature: float = 0.9,
) -> List[str]:
    parent_info = "\n".join(
        f"- {candidate!r} -> {score_value * 100:.1f}% match"
        for candidate, score_value in zip(parents, scores)
    )

    prompt = f"""
You are generating candidate strings to evolve toward text with a model.
Target: {target!r}
Best previous candidates:
{parent_info}

Create exactly {size} new candidate strings that are likely to become closer to the target.
Rules:
1. Each candidate should be a string of similar length to the target.
2. Keep the strongest ideas from the best candidates.
3. Add small but helpful variations.
4. Output only valid JSON as a list of strings.
5. No markdown fences, no commentary.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at generating creative text variants for optimization tasks. Return only valid JSON arrays of strings.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "[]"
    candidates = parse_candidates(content)
    cleaned = []
    for item in candidates:
        item = item.strip()
        if item:
            cleaned.append(item)
    if not cleaned:
        raise RuntimeError(f"The model returned no usable candidates. Raw output: {content!r}")
    return cleaned[:size]


def evolve_with_model(
    target: str,
    population_size: int = 8,
    max_generations: int = 20,
    model_temperature: float = 0.9,
) -> Tuple[str, int, List[Tuple[str, float]]]:
    client = require_api_key()

    # Seed population with model-generated guesses.
    population = ask_model_for_candidates(
        client=client,
        target=target,
        parents=[],
        scores=[],
        size=population_size,
        temperature=model_temperature,
    )
    population = [candidate[:len(target)] if len(candidate) >= len(target) else candidate.ljust(len(target), " ") for candidate in population]

    history: List[Tuple[str, float]] = []
    for generation in range(1, max_generations + 1):
        scored = [(candidate, score(candidate, target)) for candidate in population]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_candidate, best_fitness = scored[0]
        history.append((best_candidate, best_fitness))

        print(f"Generation {generation:02d}: {best_fitness * 100:.2f}% -> {best_candidate!r}")

        if best_candidate == target:
            return best_candidate, generation, history

        top_candidates = [candidate for candidate, _ in scored[: min(3, len(scored))]]
        top_scores = [fitness for _, fitness in scored[: min(3, len(scored))]]
        new_candidates = ask_model_for_candidates(
            client=client,
            target=target,
            parents=top_candidates,
            scores=top_scores,
            size=population_size,
            temperature=model_temperature,
        )

        # Keep the very best candidate in the population.
        population = [best_candidate] + [candidate[:len(target)] for candidate in new_candidates[:population_size - 1]]

    best_candidate = max(population, key=lambda candidate: score(candidate, target))
    return best_candidate, max_generations, history


def main() -> None:
    target = input("Enter the target string: ").strip()
    if not target:
        raise SystemExit("The target cannot be empty.")

    print("\nUsing a real language model to generate candidate guesses...\n")
    best_guess, generations, history = evolve_with_model(
        target=target,
        population_size=6,
        max_generations=25,
        model_temperature=0.9,
    )

    final_score = score(best_guess, target)
    print(f"\nBest final guess: {best_guess!r}")
    print(f"Final score: {final_score * 100:.2f}%")
    print(f"Reached in {generations} generations." if best_guess == target else f"Stopped after {generations} generations.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        print("\nInstall the dependency with: pip install openai")
        print("Then set your API key: export OPENAI_API_KEY=your_key_here")
