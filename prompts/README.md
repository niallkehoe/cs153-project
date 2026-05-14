# prompts/

System prompt templates for each of the three agents.

## Files

| File | Agent | Critical Constraints |
|---|---|---|
| `scientist.txt` | GPT-1900 (ScientistAgent) | Victorian persona; tool call format; no forward-looking assumptions |
| `simulator.txt` | Modern LLM (SimulatorAgent) | Raw observations ONLY; no post-1900 vocabulary; period-appropriate language |
| `judge.txt` | Modern LLM (JudgeAgent) | Partial credit rubric; check phenomenon, mechanism, quantitative form, confidence |

## Design Notes

### scientist.txt
GPT-1900 was instruction-tuned using Victorian-era QA pairs (see Hla's blog). The scientist prompt should:
- Set the scene as a late-19th-century natural philosopher equipped with a well-appointed laboratory
- Describe the two available tools (`propose_experiment`, `conclude`) in period-appropriate language without anachronistic framing
- Emphasise empirical rigour — form a hypothesis only after sufficient observations
- NOT mention any specific modern theories, even to tell the model to avoid them

### simulator.txt
This is the most safety-critical prompt. Key rules:
- Return ONLY instrument readings and descriptions of observed phenomena
- Translate all physics into period-appropriate observational language (e.g. "the balance tips 3.2 grains to the left" not "the net force is 0.031 N")
- Do NOT use words: photon, quantum, relativistic, spacetime, electron (in the modern sense), wave-particle, etc.
- If the experiment is physically impossible or ill-specified, return a practical failure (instrument limitation, etc.)

See [Concern #1 (Simulator Leakage)](../README.md#1-simulator-leakage).

### judge.txt
- Score each of the four rubric dimensions independently with a binary 0/1 and a one-sentence justification
- Do NOT require the scientist to use modern terminology — assess conceptual correctness, not naming
- Penalise vacuous hedging ("it is impossible to say") the same as overconfident wrong answers
- Note any phrases that look like reward hacking (fluent Victorian prose with no mechanistic content)

See [Concern #4 (Binary Scoring)](../README.md#4-binary-judge-scoring-is-insufficient) and [Concern #5 (Reward Hacking)](../README.md#5-judge-reward-hacking).
