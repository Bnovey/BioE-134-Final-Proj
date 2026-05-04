# BioE-134 Final Project, Bowman's contributions

## 1. Project Overview

For our final project, our team built a Cre-Seq analyzer. Cre-seq is a massivily parallel reporter essay that measures the transcriptional activity of thousands of candidate cis-regulatory elements simultaneously by linking each element to a unique DNA barcode and quantifying barcode abundance in RNA versus DNA. Within our project, we built tools for processing raw sequencing  data and gathering read counts, quality control analysis of the library, activity calling, plotting, RAG based litererature search and annotation, and an MCP-backed Claude agent that allows users to run and interpret the full pipeline through natural language. We were inspired to build this from Agarwal, Inoue et al. (2025), 'Massively parallel characterization of transcriptional regulatory elements,' Nature [1].

**My contributions**: I worked on the data processing, QC, and MCP server for this project. The data processing steps (association and counting) are exposed through the MCP server alongside the QC tools, allowing users to run the full pipeline end-to-end through natural language.

## 2. Data Collection

While the earlier stages of our project were built on synthetic data, real data was needed to properly assess our pipeline. I sourced our data from Agarwal, et al. [1]. The data was stored in the ENCODE portal under experiment ENCSR463IRX. At first, the organization of this data was very confusing. There are multiple pages for different file types, and it was not immediately clear which files corresponded to which steps of the pipeline. After carefully reading the methods section of the paper, I identified the six files I needed: the association R1 reads, the barcode index, the plasmid DNA counts,two RNA replicate FASTQs, and the design FASTA. 

The box drive is linked here: https://berkeley.box.com/s/ouz2tfv72dnfbqnu3utagnw8mtdhkfc7

 | File | Purpose |                                         
  |---|---|
  | rna_rep2.fastq.gz | RNA replicate 2 — barcode counts (numerator) |                                                          
  | rna_rep1_correct.fastq.gz | RNA replicate 1 — barcode counts (numerator) |                                                  
  | assoc_bc.fastq.gz | Barcode index — i5 reads linking barcodes to oligos |                                                  
  | dna_rep1.fastq.gz | Plasmid DNA barcode counts (denominator) |                                                            
  | reference.fa | Design FASTA — all designed oligo sequences |


## 3. Library QC Module

- File: `creseq_mcp/qc/library.py`
- I built 9 tools as part of the qc module. 

| Tool | Inputs | Output | Description | Citation |
|---|---|---|---|---|
| `tool_barcode_complexity` | `mapping_table_path` (str), `min_reads_per_barcode` (int) | dict — per-oligo barcode counts pass/fail | Counts how many distinct barcodes support each designed oligo and the median read depth per barcode. PASS when median barcodes/oligo ≥ 10. | [1][2] |
| `tool_oligo_recovery` | `mapping_table_path` (str), `design_manifest_path` (str), `thresholds` (list[int]) | dict — recovery rates by category, pass/fail | Measures what fraction of designed oligos were recovered, broken out by category. PASS when test element recovery ≥ 80% and positive control recovery ≥ 95%. | [5] |
| `tool_barcode_collision_analysis` | `mapping_table_path` (str, optional), `min_read_support` (int) | dict — collision rate, pass/fail | Identifies barcodes that map to more than one designed oligo, which introduces noise in activity measurements. PASS when collision rate < 1%. | [6] |
| `tool_barcode_uniformity` | `plasmid_count_path` (str, optional), `min_barcodes_per_oligo` (int) | dict — Gini coefficient, pass/fail | Measures evenness of barcode abundance across the plasmid pool using the Gini coefficient. A high Gini indicates a few barcodes dominating. PASS when median Gini < 0.30. | [7] |
| `tool_gc_content_bias` | `mapping_table_path` (str, optional), `design_manifest_path` (str, optional), `gc_bins` (int) | dict — recovery by GC bin, pass/fail | Checks whether oligos with extreme GC content were lost during synthesis by stratifying recovery across GC bins. PASS when no bins show recovery < 50% of the median. | [8] |
| `tool_oligo_length_qc` | `mapping_table_path` (str, optional), `design_manifest_path` (str, optional) | dict — fraction full-length, pass/fail | Checks for synthesis truncations by comparing observed alignment length to designed oligo length. PASS when median fraction full-length ≥ 0.80. | [5] |
| `tool_plasmid_depth_summary` | `plasmid_count_path` (str, optional) | dict — read depth statistics, pass/fail | Reports barcode-level read count statistics in the plasmid DNA library. PASS when median DNA count ≥ 10 and fewer than 10% of barcodes have zero counts. | [2] |
| `tool_variant_family_coverage` | `mapping_table_path` (str, optional), `design_manifest_path` (str, optional) | dict — family recovery rates, pass/fail | Checks that each variant family (reference oligo + all mutants) is fully recovered. Missing a reference makes delta score computation impossible. PASS when ≥ 80% of families fully recovered. | [1] |
| `tool_library_summary_report` | `mapping_table_path` (str, optional), `plasmid_count_path` (str, optional), `design_manifest_path` (str, optional) | dict — overall pass/fail, failed checks, per-tool summaries | Runs all applicable QC tools in a single call and returns an overall pass/fail with per-tool summaries. Main entry point for one-shot library QC. | — |

Example Prompts: 
- "Run a full QC report on my library and tell me if it passes."
- "How many barcodes are supporting each oligo?"                                                           
- "What fraction of my designed oligos were actually recovered? Break it down
  by category."                                                                 
- "Are any barcodes mapping to more than one oligo?"                                                              
- "Is my plasmid library evenly represented, or are a few barcodes dominating?
   Give me the Gini coefficient."                                               
- "Did high or low GC oligos drop out during synthesis? Check for GC content
  bias in my recovery."                                                         
- "Are my oligos the right length, or is there evidence of synthesis   
  truncations?"                                                                 
- "What's the sequencing depth of my plasmid DNA library? Are any barcodes
  missing coverage?"                                                            
- "For each variant family, did all the mutants make it in — including the
  reference oligo?"  

## 4. Association Step

- File: `creseq_mcp/association/association.py`
- Replaces Nextflow/MPRAflow for the one-time library build step: R1 FASTQ + design FASTA → `mapping_table.tsv`

**Protocol note** — In the lentiMPRA protocol, the random barcode is sequenced as the i5 index read during paired-end Illumina sequencing. It appears in every FASTQ header after the final colon (`1:N:0:BARCODE+i7index`). R1 reads the oligo insert (used for alignment). Neither R1 nor R2 contains the barcode in the sequence itself.

**Inputs**

- `fastq_r1`:  R1 FASTQ — oligo insert reads
- `fastq_r2`: R2 FASTQ — paired oligo reads for better alignment (Optional)
- `design_fasta`: FASTA of all designed oligo sequences 
- `outdir`: Directory to write output TSVs 
- `fastq_bc`: Separate barcode index FASTQ (ENCODE format: i5 read as its own file; read names must match R1) 
- `labels_path`: TSV with oligo_id + designed_category columns 
- `min_cov`: Minimum reads per barcode–oligo pair (default 3) 
- `min_frac`: Minimum fraction mapping to same oligo (default 0.5) 
- `mapq_threshold`:  Minimum minimap2 mapping quality (default 20) 
- `starcode_dist`: STARCODE edit-distance for clustering (default 1)

**Pipeline**
1. Parse R1 headers → raw barcode per read
2. STARCODE → cluster barcodes within edit-distance 1 (error correction) [4]
3. mappy/minimap2 → align R1 sequences to design FASTA → oligo_id per read [3]
4. Join → (clustered barcode, oligo_id) per read
5. Filter → min_cov reads AND min_frac mapping to same oligo
6. Write → `mapping_table.tsv`, `plasmid_counts.tsv`, `design_manifest.tsv`

**Outputs**

`mapping_table.tsv`: barcode, oligo_id, n_reads, cigar, md
`plasmid_counts.tsv`: barcode, oligo_id, dna_count 
`design_manifest.tsv`: oligo_id, sequence, designed_category, variant_family

**Note on file paths** — When called through the MCP server, `outdir` is not set by the user. Output files are written to a temporary association directory and then copied into `UPLOAD_DIR` (`~/.creseq/uploads/` by default, overridable via the `CRESEQ_UPLOAD_DIR` environment variable). All downstream tools resolve their input paths from `UPLOAD_DIR` automatically. This is to ensure matching file locations across pipeline steps — the association output, DNA/RNA counts, and QC tools all read from and write to the same directory without the user needing to pass paths between calls.


## 5. DNA and RNA Counting

- File: `creseq_mcp/activity/counting.py`
- Counts barcode occurrences in DNA/RNA FASTQs using the mapping table produced by the association step.

**Inputs - DNA Counting**

- `fastq_path`: DNA barcode FASTQ
- `mapping_table_path`: Barcode→oligo mapping TSV from association 
- `upload_dir`: Directory to write output 
- `barcode_len`: Length of barcode to extract from read (default 20)
- `barcode_end`: Which end of read contains barcode: `"3prime"` or `"5prime"` (default `"3prime"`)
- `max_mismatch`: Allowed mismatches when matching barcodes (default 0) 

**Inputs — RNA Counting**

-`fastq_paths`: One FASTQ per RNA replicate 
-`mapping_table_path`: Barcode→oligo mapping TSV from association 
-`upload_dir`: Directory to write output
- `rep_names`: Replicate labels (default `rep1`, `rep2`, …) 
-`barcode_len`: Length of barcode to extract (default 20)
- `barcode_end`: `"3prime"` or `"5prime"` (default `"3prime"`)
- `max_mismatch`:  Allowed mismatches (default 0) 

**Outputs**

- `plasmid_counts.tsv`: barcode, oligo_id, dna_count 
- `rna_counts.tsv`: barcode, oligo_id, rna_count_rep1, rna_count_rep2, …

**Another note on file paths** If running the full pipeline from the association stage, `mapping_table_path` is not required as an input to downstream tools — it is resolved automatically from `UPLOAD_DIR`. If skipping the association stage and providing a pre-built mapping table, the path must be passed explicitly. Again, `UPLOAD_DIR` is internally managed by the server and is not exposed as a user-facing parameter; it can be redirected by setting the `CRESEQ_UPLOAD_DIR` environment variable before starting the server.

## 6. MCP Server

- File: `creseq_mcp/server.py`

I chose to use FastMCP package for our project. It allows for simple and consistent tool/resource registration. Each tool is registered with a single `@mcp.tool()` decorator — FastMCP reads the function's type hints and docstring to auto-generate the JSON schema that Claude uses to understand what the tool does and what arguments it accepts. Rather than writing raw JSON wrappers by hand, we wrote Python functions with explicit signatures, type hints, and docstrings, and FastMCP handled the schema generation automatically. The function signatures and docstrings serve as the source of truth for the wrapper content.

The server is initialized as:

```python
mcp = FastMCP("creseq-mcp", instructions="...")
```

The `instructions` field is sent to the model at the start of every session and sets global context. It functions like a system prompt scoped to the MCP server — Claude receives it before any tool calls are made, so it shapes how the agent interprets user requests and which assumptions it brings into the conversation. In our case, the instructions tell the agent the full scope of available tools, that file path arguments are optional, and that QC thresholds are calibrated for CRE-seq rather than other MPRA variants. 

**MCP JSON wrappers** — each tool's wrapper is auto-generated by FastMCP from the function's type hints and docstring. For example, `tool_plasmid_depth_summary` is defined as:

```python
@mcp.tool()
def tool_plasmid_depth_summary(plasmid_count_path: str | None = None) -> dict:
    """
    Barcode-level read-count statistics in the plasmid DNA library.
    PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts.
    """
```

FastMCP exposes this to the LLM as a JSON schema equivalent to:

```json
{
  "name": "tool_plasmid_depth_summary",
  "description": "Barcode-level read-count statistics in the plasmid DNA library. PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "plasmid_count_path": {
        "type": "string",
        "description": "Path to plasmid counts TSV. Optional — resolved from UPLOAD_DIR if omitted."
      }
    }
  }
}
```

All 9 QC tools, the association tool, DNA/RNA counting tools, and literature tools follow this same pattern. Inputs marked `str | None = None` are optional; when omitted, the server resolves the path from `UPLOAD_DIR` automatically via the `_path()` helper.

**Resources** — the Agarwal et al. 2025 paper is registered as a queryable resource at `paper://agarwal2025-lentimpra` using `@mcp.resource(...)`. Claude can retrieve the full text of the paper and use it to annotate results or answer questions about the assay design.

## 7. Pytests

72 tests across 3 files. All tests use synthetic in-memory fixtures and cover edge cases including empty inputs, zero counts, below-threshold values, and missing columns — no real sequencing data required to run them.

| File | Tests | Coverage |
|---|---|---|
| `tests/qc/test_library.py` | 38 | All 9 QC tools — pass/fail logic, edge cases, empty inputs, threshold boundaries |
| `tests/test_association.py` | 24 | FASTQ header barcode parsing, gzip detection, filter logic, barcode–oligo assignment |
| `tests/test_counting.py` | 10 | DNA and RNA barcode counting, mismatch tolerance, multi-replicate output |

## 8. Bugs and Issues Along the Way

**Table Names/File Locations**

I ran into a number of issues along the way. Because we are dealing with many files and tables, having names and locations match is key. This is one of the issues in chaining together current tools as well. Everytime I ran the full pipeline, I would find errors related to these two artifacts. Slowly, I uncovered these errors and ensured that file locations and table column names agree.

Since I wanted to add the ability to skip association (it takes over an hour to complete), I had to make sure intermediate files were saved in place where users can access. This added more complexity to file saving and as the server needed to know whether outputs came from a fresh association run or from pre-existing files already in UPLOAD_DIR.

**Wrong Files**

One of the more frustrating issues was using the wrong RNA FASTQ file. The ENCODE portal organizes files across multiple pages, and I initially downloaded an R2 read file thinking it was R1. Because R2 reads the oligo from the opposite end, the barcodes weren't being found and counting returned near-zero matches. It wasn't obvious at first because the pipeline ran without errors, the counts were just returned zeros. After looking through the methods section and consulting claude, I was able to find the right RNA FASTQ file. 

**Other Issues** 

Along the way, I ran into small issues with file formats and simple bugs in code. These included a dict vs. 2-tuple unpacking bug in the QC tools, a same-file copy error when outputs were written back to their own source, a broken else-branch in the skip-association toggle, and UPLOAD_DIR path instability when the server was run from different working directories. None were individually difficult, but each only surfaced when the full pipeline was run end-to-end. 

## 9. LLM Usage

Most of my code was written with Claude Code. This was a great test on my ability to accurately prompt what functions I want written. Many times, I ran into issues with Claude not understanding what I want to write and it through my part of the project off track. To get over this, I had to carefully think about all aspects of the functions I wanted to write and even ideated with claude to help me flush out my ideas. 

The more I write with an LLM, the more I learn what level of abstraction I would be "vibe coding" at. How granualar should my prompts be? Should I ask it go write the entire QC library in one shot, or should I go tool by tool. What am I confortable with? I played around with this idea along the project and I am slowly finding that the LLM and I write code best at a level above individual functions but not too abstract where I'm not interacting with those smaller functions. 

## 10. Collaboration

I collaborated with Sarrah Rose, Arjun Gurjar, and Zach Rao on this project. Sarrah worked on plotting and downstream analysis, Arjun worked on the frontend, and Zach worked on RAG based tools. 


## 11. Citations

[1] Agarwal, V., Inoue, F. et al. Massively parallel characterization of transcriptional regulatory elements. *Nature* 637, 569–577 (2025). https://doi.org/10.1038/s41586-024-08430-9

[2] Gordon, M.G., Inoue, F., Martin, B. et al. lentiMPRA and MPRAflow for high-throughput functional characterization of gene regulatory elements. *Nat Protoc* 15, 2387–2412 (2020). https://doi.org/10.1038/s41596-020-0333-5

[3] Li, H. Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics* 34(18), 3094–3100 (2018). https://doi.org/10.1093/bioinformatics/bty191

[4] Zorita, E., Cuscó, P. & Filion, G.J. Starcode: sequence clustering based on all-pairs search. *Bioinformatics* 31(12), 1913–1919 (2015). https://doi.org/10.1093/bioinformatics/btv053

[5] Kuiper, B.P., Prins, R.C. & Billerbeck, S. Oligo pools as an affordable source of synthetic DNA for cost-effective library construction in protein- and metabolic pathway engineering. *ChemBioChem* 23(7), e202100507 (2022). https://doi.org/10.1002/cbic.202100507

[6] Johnson, M.S., Venkataram, S. & Kryazhimskiy, S. Best practices in designing, sequencing, and identifying random DNA barcodes. *J Mol Evol* 91, 263–280 (2023). https://doi.org/10.1007/s00239-022-10083-z

[7] Planet, E., Stephan-Otto Attolini, C., Reina, O., Flores, O. & Rossell, D. htSeqTools: high-throughput sequencing quality control, processing and visualization in R. *Bioinformatics* 28(4), 589–590 (2012). https://doi.org/10.1093/bioinformatics/btr700

[8] Benjamini, Y. & Speed, T.P. Summarizing and correcting the GC content bias in high-throughput sequencing. *Nucleic Acids Res.* 40(10), e72 (2012). https://doi.org/10.1093/nar/gks001