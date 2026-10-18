# Conventions for every model in this repo

These are the rules the daily job (and any human contributor) follows. The standard is set by the two original models: implement from the equations, and check the result against something other than its own plot.

## Deliverables per model

One commit on `main` containing exactly:

1. `src/<model_name>.py` — a single, self-contained script. NumPy for the model; Matplotlib for the figure; SciPy only for validation (e.g. `solve_ivp` as a reference solver, `special` for closed forms), never as the model itself. Running it with no arguments must reproduce the figure and print the validation numbers.
2. `figures/<model_name>.png` — one figure, 150 dpi, that shows the behaviour *and* the validation (e.g. simulated vs analytical overlaid).
3. A new `### <Model name>` subsection in `README.md` under **Method**, a row in the **Validation** table, a line in **Reproduce**, and any honest new entry under **Limitations and failure cases**. Keep the README's existing voice: plain, quantitative, no hype.
4. The matching box ticked in `ROADMAP.md`, with the script path appended in parentheses, exactly like the first two rows.
5. `requirements.txt` updated only if a new dependency is genuinely needed.

## Code standards

- Equations written out with parameters named as in the cited paper; cite the paper in a module docstring.
- Explicit integration with a stated `dt`; state the integrator in the docstring. Deterministic: seed any RNG and print the seed.
- Module layout: parameters → model functions → `simulate()` → `validate()` → `plot()` → `if __name__ == "__main__":`.
- Runtime under 5 minutes on a laptop CPU. No external data downloads.
- No notebooks, no classes for their own sake, no framework (Brian2, NEURON, NEST) — that is the point of "from scratch".

## Validation standard

Every model must pass at least one quantitative check that does not depend on the model's own output looking right. Acceptable kinds, roughly in order of strength:

1. A closed-form result (firing rate, fixed point, bifurcation point, steady-state amplitude, attenuation length).
2. A reference numerical solver (`solve_ivp` with tight tolerances) or a convergence-in-dt study.
3. A published number or regime boundary from the original paper, cited with figure or table number.
4. A limiting case that reduces to an already-validated model in this repo.

Report the check as a number (relative error, max deviation, agreement within N%) in the README table, and have the script assert it with a stated tolerance so a regression fails loudly.

## Commit message

```
Add <Model name>: <one-line result with the validation number>
```

Example: `Add FitzHugh–Nagumo: Hopf onset current from sweep matches Jacobian prediction to 0.4%`.

## When something does not validate

Do not loosen the tolerance until it passes. Either fix the implementation, or commit the model with the discrepancy stated plainly under Limitations and the README row marked "does not match — see Limitations". Honesty about failure is part of the standard.
