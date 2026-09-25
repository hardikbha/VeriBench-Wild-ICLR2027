# Frozen prompts

These files reproduce, verbatim, the prompt templates listed in the paper's appendix. They are the
historical prompts used to produce the reported results and are kept unchanged. Placeholders appear
in braces.

| File | Use |
|---|---|
| `t0_generator_mistral.txt` | T0 generator template (Mistral). Inputs: evidence note and target mechanisms; the original post is not a generator input. The Qwen generator uses the same note-and-mechanism contract with its own chat formatting. |
| `t0_evaluator_judge.txt` | T0 evaluator prompt used by all eight judges. It also receives the original claim and judges each output as a rewrite of it. |
| `t0_mechanism_definitions.txt` | Mechanism definitions substituted into the evaluator prompt. |
| `t1_prompted_typology.txt`, `t1_body_blocks.txt` | Prompted T1 template and its claim/note body blocks (omitted channels are absent, not empty). |
| `binary_transfer_icl.txt` | Binary-verdict transfer prompt; the demonstration block is repeated 16 times for ICL and omitted for zero-shot and QLoRA. Labels are scored as continuations, not parsed. |
| `external_verdict.txt`, `external_verdict_definitions.txt` | External evidence-only diagnostic prompt and verdict definitions. |
| `t1_mask_vocabulary.txt`, `external_mask_vocabulary_additions.txt` | Fixed phrase lists for the trigger-masked condition. |

## Parser rules for the T0 evaluator

- **Strict (frozen):** case-insensitive search for `VERDICT: YES`, `VERDICT: NO`, or
  `VERDICT: NEUTRAL`; unmatched responses count as NO.
- **Lenient:** additionally reads a verdict on the first non-empty line (optionally bolded or
  punctuated) when no `VERDICT:` field is found.
- All 476 judgments stay in the denominator; unparsed responses count as not-YES.
- Both parsers are applied to all eight judges. The three original judges stored only parsed
  verdicts, so they were re-run with the frozen protocol to store raw replies; their strict
  verdicts reproduce 100% (Mistral-7B), 99.8% (Phi-3-medium), and 93.9% (Qwen2.5-0.5B) of the
  originally stored ones.

Note that the evaluator prompt asks for exactly one verdict token but also specifies a
`VERDICT:`-prefixed format. The paper reports the resulting parser sensitivity as
protocol-specific; the prompt is not corrected here.
