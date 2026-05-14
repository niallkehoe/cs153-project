# tools/

Tool schema definitions and the call router for the experiment loop.

## Files

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models and JSON schemas for all scientist tool calls |
| `router.py` | Dispatches parsed tool calls to the right agent or terminates the loop |

## Available Tools (scientist-facing)

### `propose_experiment`
The scientist calls this to request data from the simulator.

```json
{
  "name": "propose_experiment",
  "description": "Propose an experiment to carry out and receive observational results.",
  "parameters": {
    "description": "Free-text description of the experiment",
    "parameters": {
      "key": "value pairs of experimental variables, e.g. {mass_kg: 1.0, height_m: 5.0}"
    }
  }
}
```

### `conclude`
The scientist calls this to end the loop and record a final hypothesis.

```json
{
  "name": "conclude",
  "description": "Record your final conclusion about the phenomenon under investigation.",
  "parameters": {
    "hypothesis": "Your conclusion in plain language",
    "confidence": "high | medium | low"
  }
}
```

## Tool Router

`ToolRouter` parses the scientist's raw generation text, extracts the tool call (if any), and either:
- Forwards a `propose_experiment` call to `SimulatorAgent` and returns the result to the scientist
- Captures a `conclude` call and signals the orchestrator to end the loop
- Handles malformed or missing tool calls with a gentle re-prompt

The router also enforces the `max_turns` limit — if the scientist has not called `conclude` within the allowed turns, the loop is terminated and the most recent generation is passed to the judge as the de-facto hypothesis.
