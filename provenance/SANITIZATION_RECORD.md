# Anonymous-release sanitization record

The public release removes machine-specific absolute filesystem defaults from archival execution scripts.

No prompts, schemas, parser logic, normalization rules, generation parameters, model revisions, random seeds, data-selection rules, metric definitions, or statistical procedures were changed by this sanitization.

## Original execution runner hashes

- `run_structured_extraction_v2.py`
  - SHA-256: `98ed4423ab40a278827525c0c3f5cb5fcc1ef12bd404bc7589551aecd0a564dc`

- `run_structured_extraction_gemma4_v2.py`
  - SHA-256: `b74de19b32d2a801838b9261017b283a8f6ad7c2f6ab70bfbed13cacf11d26ab`

## Anonymous-safe public-copy hashes

- `code/run_structured_extraction_v2.py`
  - SHA-256: `730838cc4a25172a97ea9007d3fef8d8e5f70ade3df29f529262531046cb07a3`

- `code/run_structured_extraction_gemma4_v2.py`
  - SHA-256: `7ab3dcc87a6ef6b3f675808d06915803bb5f6960834f9a8f9ee6091061ce7512`

The V3 public runner verifies the hashes of the anonymous-safe public copies. The original exact execution hashes are preserved above and in `ORIGINAL_RELEASE_FILE_HASHES_20260903.sha256`.

The release build verified programmatically that the pre-existing scripts modified for anonymization differ from the September release only by the declared path substitutions.
