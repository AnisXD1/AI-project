# String Breeder (Python)

An educational, standard-library-only example of a **genetic algorithm**. It evolves random candidate strings toward a target by repeatedly scoring guesses, selecting the best two, combining them, and introducing random mutations.

> This is a small optimization demo, not an AI that understands language and not a model of real biological crossbreeding. “Parents,” “breeding,” and “mutation” are algorithm analogies.

## Run it

Requires Python 3.8 or newer. In a terminal, from the repository folder:

```bash
python string_breeder.py
```

Type a target string when prompted. It prints the best guess and its fitness every 100 generations by default. To change the settings:

```bash
python string_breeder.py --population 150 --mutation 0.05 --max-generations 50000 --report-every 50
```

- `--population`: how many guesses are in each generation.
- `--mutation`: chance each character changes, as a decimal from `0` to `1` (`0.03` means 3%).
- `--max-generations`: stop after this many generations if the target has not been reached.
- `--report-every`: how often to print progress.

## Read the code as a learning path

Open `string_breeder.py` and follow these functions in order:

1. **`score(candidate, target)` — fitness:** counts characters that match the target at the same position. Dividing by the target length makes the score a percentage from 0% to 100%.
2. **`make_random_candidate(...)` — initialization:** creates the first population of guesses.
3. **`choose_parents(...)` — selection:** ranks guesses and chooses the two best. This example always selects the top two to keep the selection rule easy to see.
4. **`crossover(...)` — recombination:** makes a child by taking each character from one parent or the other.
5. **`mutate(...)` — exploration:** randomly replaces characters. Mutation can discover characters that neither selected parent currently has.
6. **`make_next_generation(...)` — replacement:** creates the next population and keeps the best guess unchanged (called **elitism**).
7. **`evolve(...)` — the loop:** repeats scoring and breeding until it finds the target or reaches the generation limit.

## What to notice

- The algorithm is optimizing a clearly defined score; it does not understand what the string means.
- The fitness function compares exact positions. A different task would need a different fitness function.
- Mutation is random, so success is not guaranteed within a limited number of generations. A higher mutation rate explores more but can also disrupt useful guesses.
- The selected parents are the best two, not two randomly chosen candidates. This makes the demonstration simple, but can reduce diversity; more advanced algorithms often use tournament or probability-based selection.
- The alphabet includes the target's characters, so the exact answer is possible to create.

The existing `index.html` is the browser-based version of the same idea; `string_breeder.py` is the commented Python learning version.
