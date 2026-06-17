# Orchestration

## Run the agent without the server

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the requirement agent directly and print the final output as a JSON
dictionary:

```bash
python -m app.cli --auto-answer "The system shall report voltage."
```

The default output only includes final retrieval subqueries. `final_subquery` is
the final field to use. Draft-stage fields such as `draft_subquery`, and the
legacy `subquery` alias, are omitted from the default output.

You can also read the requirement from a file:

```bash
python -m app.cli --auto-answer --file Data/test_query.txt
```

Print the complete raw final state as JSON:

```bash
python -m app.cli --auto-answer --json "The system shall report voltage."
```

Print a human-readable summary:

```bash
python -m app.cli --auto-answer --summary "The system shall report voltage."
```

Without `--auto-answer`, the CLI stays in the same process and prompts for
clarification answers when the graph pauses.

If the package is installed in editable mode, the script entry point is also
available:

```bash
requirement-agent --auto-answer "The system shall report voltage."
```
