"""An educational genetic algorithm that evolves guesses toward a target string.

Run with:
    python string_breeder.py

This is a small optimization experiment, not an AI that understands language and
not a model of real biological crossbreeding. Read the functions from top to
bottom to follow the algorithm: score -> select -> crossover -> mutate -> repeat.
"""

import argparse
import random
import string
from typing import List, Sequence, Tuple


# The algorithm can produce ordinary letters, digits, spaces, and punctuation.
# Characters present in the target are added too, so any target can be represented.
BASE_ALPHABET = string.ascii_letters + string.digits + " .,!?'-"


def score(candidate: str, target: str) -> float:
    """Return how many characters are correct in the correct positions.

    This is called a fitness function: it gives the algorithm a measurable way
    to compare guesses. The result is between 0.0 (no matches) and 1.0 (exact).
    """
    if not target:
        return 1.0 if candidate == target else 0.0

    matching_positions = sum(
        guessed_character == target_character
        for guessed_character, target_character in zip(candidate, target)
    )
    return matching_positions / len(target)


def make_random_candidate(target: str, alphabet: str) -> str:
    """Create one random guess with the same length as the target."""
    return "".join(random.choice(alphabet) for _ in target)


def choose_parents(population: Sequence[str], target: str) -> Tuple[str, str]:
    """Select the two guesses with the highest fitness scores.

    This simple example uses deterministic selection: the best two always win.
    More advanced genetic algorithms often use probabilistic selection instead.
    """
    ranked_candidates = sorted(
        population,
        key=lambda candidate: score(candidate, target),
        reverse=True,
    )
    return ranked_candidates[0], ranked_candidates[1]


def crossover(parent_a: str, parent_b: str) -> str:
    """Make one child by choosing each character from either parent.

    This is per-character crossover. For example, parents ABCD and WXYZ might
    produce ABYZ. The child can combine useful characters from both parents.
    """
    return "".join(
        random.choice((character_a, character_b))
        for character_a, character_b in zip(parent_a, parent_b)
    )


def mutate(candidate: str, alphabet: str, mutation_rate: float) -> str:
    """Randomly replace each character with the given per-character chance.

    Mutation introduces new possibilities. Without it, crossover can only
    rearrange characters that are already present in the selected parents.
    """
    changed_characters = []
    for character in candidate:
        if random.random() < mutation_rate:
            changed_characters.append(random.choice(alphabet))
        else:
            changed_characters.append(character)
    return "".join(changed_characters)


def make_next_generation(
    population: Sequence[str], target: str, alphabet: str, mutation_rate: float
) -> List[str]:
    """Breed a new population from the current population.

    Elitism means copying the best candidate unchanged into the next generation.
    It protects progress: a good guess cannot be lost just because of mutation.
    The remaining children are made by crossing over the two best parents.
    """
    parent_a, parent_b = choose_parents(population, target)
    next_population = [parent_a]  # Keep the best candidate unchanged (elitism).

    while len(next_population) < len(population):
        child = crossover(parent_a, parent_b)
        child = mutate(child, alphabet, mutation_rate)
        next_population.append(child)

    return next_population


def evolve(
    target: str,
    population_size: int = 100,
    mutation_rate: float = 0.03,
    max_generations: int = 100_000,
    report_every: int = 100,
) -> Tuple[str, int]:
    """Run the genetic algorithm and return the best guess and generation count."""
    # Include every target character in the alphabet, even uncommon symbols.
    alphabet = "".join(dict.fromkeys(BASE_ALPHABET + target))
    population = [
        make_random_candidate(target, alphabet) for _ in range(population_size)
    ]

    for generation in range(max_generations + 1):
        best_candidate = max(population, key=lambda item: score(item, target))
        best_fitness = score(best_candidate, target)

        # Print occasional progress rather than flooding the terminal.
        if generation == 0 or generation % report_every == 0 or best_fitness == 1.0:
            percentage = best_fitness * 100
            print(
                f"Generation {generation:>6}: "
                f"{percentage:6.2f}% match | {best_candidate!r}"
            )

        if best_candidate == target:
            return best_candidate, generation

        if generation < max_generations:
            population = make_next_generation(
                population, target, alphabet, mutation_rate
            )

    return max(population, key=lambda item: score(item, target)), max_generations


def positive_integer(value: str) -> int:
    """argparse helper that rejects zero and negative count settings."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return number


def mutation_probability(value: str) -> float:
    """argparse helper that checks a probability is between 0 and 1."""
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve random strings toward a target using a genetic algorithm."
    )
    parser.add_argument("--population", type=positive_integer, default=100,
                        help="number of guesses in each generation (default: 100)")
    parser.add_argument("--mutation", type=mutation_probability, default=0.03,
                        help="mutation chance per character, from 0 to 1 (default: 0.03)")
    parser.add_argument("--max-generations", type=positive_integer, default=100_000,
                        help="stop after this many generations (default: 100000)")
    parser.add_argument("--report-every", type=positive_integer, default=100,
                        help="print progress every N generations (default: 100)")
    args = parser.parse_args()

    target = input("Enter the target string: ")
    if not target:
        parser.error("the target string cannot be empty")

    print("\nStarting evolution. The percentage is positional character fitness.\n")
    best_candidate, generations = evolve(
        target=target,
        population_size=args.population,
        mutation_rate=args.mutation,
        max_generations=args.max_generations,
        report_every=args.report_every,
    )

    if best_candidate == target:
        print(f"\nSuccess: reached {target!r} in {generations} generations.")
    else:
        print(
            f"\nStopped after {generations} generations. Best guess: "
            f"{best_candidate!r} ({score(best_candidate, target) * 100:.2f}% match)."
        )
        print("Try a larger population or a different mutation chance.")


if __name__ == "__main__":
    main()
