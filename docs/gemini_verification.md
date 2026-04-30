# Gemini MCP verification — step-by-step

This doc walks through verifying that the `creseq-mcp` server is reachable
from Google's Gemini CLI as an MCP client. Three demo prompts cover one
simple tool call, one tool with structured arguments, and one chained
multi-tool sequence. After completing the steps, four screenshots go in
`docs/screenshots/`.

---

## 1. Install Gemini CLI

```bash
# Option A — via npm (recommended on macOS)
npm install -g @google/generative-ai-cli

# Option B — via Homebrew if a tap is available
brew install google/tap/gemini-cli
```

Verify the binary is on PATH:

```bash
gemini --version
```

If you do not yet have an API key, get a free one from
<https://aistudio.google.com/app/apikey> and export it:

```bash
export GEMINI_API_KEY="paste-key-here"
```

(Persist this in your shell rc file if you want it to stick.)

---

## 2. Register `creseq-mcp` with Gemini

Open (or create) `~/.gemini/settings.json` and add:

```json
{
  "mcpServers": {
    "creseq": {
      "command": "/Users/sarrahrose/Downloads/x/BioE-134-Final-Proj/.venv/bin/python",
      "args": ["-m", "creseq_mcp.server"],
      "cwd": "/Users/sarrahrose/Downloads/x/BioE-134-Final-Proj"
    }
  }
}
```

Two notes:

- The absolute python path points at the project's venv, which already has
  `mcp`, `pandas`, `pyjaspar`, `biopython`, `statsmodels`, etc. installed.
  Without this, Gemini will spawn the system python and fail to import.
- The `cwd` is what makes default paths like `~/.creseq/uploads/` resolve
  correctly inside the tools.

If `~/.gemini/settings.json` already exists, merge the `mcpServers` block
into the existing JSON instead of overwriting.

---

## 3. Confirm the server is registered

```bash
gemini  # opens the interactive client
```

At the Gemini prompt, type:

```
/mcp list
```

You should see a `creseq` entry and a tool count of 22. If the count is
0 or the entry says "error", check:

- `~/.gemini/settings.json` is valid JSON (run `python -m json.tool < ~/.gemini/settings.json`).
- The python path in `command` is the venv path, not `/usr/bin/python3`.
- Running `.venv/bin/python -m creseq_mcp.server` from a separate terminal
  starts cleanly and prints no errors before being killed with Ctrl+C.

**📸 Screenshot 1 — `01_mcp_list.png`**: capture the `/mcp list` output
showing `creseq` registered with 22 tools.

---

## 4. Demo prompt 1 — single-tool, default paths

This prompt exercises `tool_prepare_counts` with all default arguments
(reads `~/.creseq/uploads/{plasmid,rna}_counts.tsv`, writes
`~/.creseq/uploads/counts_long.tsv`). It is the cheapest end-to-end check
that tool-use is working.

Paste this into Gemini:

> I just finished QC on my CRE-seq library. Run `prepare_counts` on the
> default upload directory and tell me the summary — how many rows, how
> many elements, and how many replicates ended up in the long-format table?

**Expected behavior:** Gemini issues a tool call to `tool_prepare_counts`
with no arguments, the server returns
`{n_rows: 24720, n_elements: 600, n_replicates: 2, output_path: ".../counts_long.tsv"}`,
and Gemini summarises that back in natural language.

**📸 Screenshot 2 — `02_prepare_counts.png`**: capture the full exchange
including the tool-call block (showing `tool_prepare_counts` with its
arguments) and Gemini's natural-language reply quoting the three numbers.

---

## 5. Demo prompt 2 — tool with structured arguments

This exercises `call_active_elements`, which takes a list of element_ids
as one of its arguments — a useful test that Gemini can construct
non-trivial argument shapes.

First, prep the per-element activity table from the long-format counts
(one shell command, ~2 sec):

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('/Users/sarrahrose/.creseq/uploads/counts_long.tsv', sep='\t')
import numpy as np
df['log2_activity'] = np.log2((df['rna_counts']+1)/(df['dna_counts']+1))
agg = df.groupby('element_id').agg(
    mean_activity=('log2_activity','mean'),
    std_activity=('log2_activity','std'),
    n_barcodes=('log2_activity','size'),
).reset_index()
agg.to_csv('/tmp/activity_per_element.tsv', sep='\t', index=False)
print(f'wrote {len(agg)} elements to /tmp/activity_per_element.tsv')
"
```

Then in Gemini:

> The activity table at `/tmp/activity_per_element.tsv` has 600 elements.
> The first 150 negative controls are named `NEGCTRL000` through
> `NEGCTRL149`. Use `call_active_elements` to classify them at FDR 0.05
> using the empirical method, and tell me how many active elements were
> called and what the null distribution looked like.

**Expected behavior:** Gemini constructs the
`negative_controls=["NEGCTRL000", ..., "NEGCTRL149"]` argument, calls
the tool, and reports something like *"213 active out of 450 test
elements; null centred at -0.14 with scale 0.37 (median/MAD on 150
controls)."*

**📸 Screenshot 3 — `03_call_active_elements.png`**: capture the tool-use
block (showing the list argument flattened cleanly) and Gemini's reply
naming `n_active`, `null_distribution.center`, and `null_distribution.scale`.

---

## 6. Demo prompt 3 — chained multi-tool

This exercises `plot_creseq` chained off the previous step's output.
Demonstrates that the agent can carry context between tool calls.

> Now make me a volcano plot of the classified table at
> `/tmp/activity_per_element_classified.tsv` and save it to
> `/tmp/volcano_demo.png`. Tell me what the description says.

**Expected behavior:** Gemini issues a `tool_plot_creseq` call with
`plot_type="volcano"` and the right path. The server returns
`{plot_path: ".../volcano_demo.png", description: "Volcano plot of N elements; X active at FDR<0.05 ..."}`.

After the response comes back, verify the PNG exists:

```bash
ls -la /tmp/volcano_demo.png
open /tmp/volcano_demo.png   # macOS preview
```

**📸 Screenshot 4 — `04_plot_creseq.png`**: capture the tool-use block,
Gemini's reply including the description string, and the rendered PNG
visible alongside (or in a follow-up screenshot of the open file).

---

## 7. Where the screenshots go

```
docs/
  screenshots/
    01_mcp_list.png            # /mcp list showing creseq + 22 tools
    02_prepare_counts.png      # tool_prepare_counts call + summary reply
    03_call_active_elements.png  # call_active_elements with list arg + reply
    04_plot_creseq.png         # plot_creseq + rendered volcano
```

These four images are the rubric-required evidence that the MCP wrappers
"work with the Gemini MCP code." Reference them in `README_sarrah.md`
under a "Verified with Gemini CLI" section.

---

## Fallback: Claude Desktop

If Gemini CLI installation is blocked (network, permissions, API quota),
Claude Desktop uses the identical MCP protocol and is acceptable
evidence. Add the same `mcpServers` block to
`~/Library/Application Support/Claude/claude_desktop_config.json`,
restart Claude Desktop, and run the same three prompts. Screenshot the
tool-use blocks the same way and label the README section "Verified
with Claude Desktop (MCP reference client)".

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/mcp list` shows 0 tools | Wrong python path in settings.json — must be `.venv/bin/python` absolute path |
| Tool call returns "module not found" | The venv is missing dependencies; run `uv sync` or `pip install -e .` from the project root |
| `prepare_counts` errors on missing file | Generate the test data first: `.venv/bin/python scripts/generate_test_data.py` |
| Gemini hangs forever | The motif_enrichment tool can take 30-60 s on first call (downloads JASPAR); avoid it for the demo |
| Plot file not created | Confirm matplotlib's Agg backend is active by checking the tool's stderr — should be silent |
