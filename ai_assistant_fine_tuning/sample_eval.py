"""Sample questions for human evaluation validation."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import markdown
import yaml

from ai_assistant_fine_tuning.batch_util import read_jsonl

logger = logging.getLogger(__name__)


@click.command("sample-eval")
@click.option(
    "--sft-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to SFT training data JSONL file.",
)
@click.option(
    "--dpo-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to DPO training data JSONL file (optional).",
)
@click.option(
    "--answers",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to answers.jsonl (from gen-answers) to get original learner answers.",
)
@click.option(
    "--sample-size",
    type=int,
    default=10,
    show_default=True,
    help="Number of samples to take from each dataset.",
)
@click.option(
    "--sft-sample-size",
    type=int,
    required=False,
    help="Override sample size for SFT data (overrides --sample-size).",
)
@click.option(
    "--dpo-sample-size",
    type=int,
    required=False,
    help="Override sample size for DPO data (overrides --sample-size).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("human_eval"),
    show_default=True,
    help="Output directory for YAML and HTML files.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for sampling reproducibility.",
)
def sample_eval(
    sft_data: Path,
    dpo_data: Path | None,
    answers: Path,
    sample_size: int,
    sft_sample_size: int | None,
    dpo_sample_size: int | None,
    output_dir: Path,
    seed: int,
) -> None:
    """Sample questions from training data for human evaluation.

    Takes SFT and/or DPO training data, matches with original answers from
    answers.jsonl, and generates YAML + HTML files for human scoring.
    
    Samples are stratified to include varied quality levels (when possible).
    """
    random.seed(seed)
    
    # Determine actual sample sizes
    sft_n = sft_sample_size if sft_sample_size is not None else sample_size
    dpo_n = dpo_sample_size if dpo_sample_size is not None else sample_size
    
    # Load answers.jsonl and create lookup by (question content)
    click.echo(f"Loading original answers from {answers}")
    answer_records = read_jsonl(answers)
    # Create lookup: question text -> full record
    answer_lookup = {rec["question"]: rec for rec in answer_records}
    click.echo(f"Loaded {len(answer_lookup)} unique questions from answers")
    
    # Load and sample SFT data
    click.echo(f"\nLoading SFT data from {sft_data}")
    sft_records = read_jsonl(sft_data)
    click.echo(f"Loaded {len(sft_records)} SFT records")
    sft_samples = _sample_from_sft(sft_records, answer_lookup, sft_n)
    click.echo(f"Sampled {len(sft_samples)} from SFT")
    
    # Load and sample DPO data (if provided)
    dpo_samples = []
    if dpo_data:
        click.echo(f"\nLoading DPO data from {dpo_data}")
        dpo_records = read_jsonl(dpo_data)
        click.echo(f"Loaded {len(dpo_records)} DPO records")
        dpo_samples = _sample_from_dpo(dpo_records, answer_lookup, dpo_n)
        click.echo(f"Sampled {len(dpo_samples)} from DPO")
    
    # Combine all samples and shuffle to avoid bias
    all_samples = sft_samples + dpo_samples
    random.shuffle(all_samples)
    
    if not all_samples:
        click.echo("No samples generated. Exiting.")
        raise SystemExit(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    html_dir.mkdir(exist_ok=True)
    
    # Generate YAML and HTML for each sample
    click.echo(f"\nGenerating {len(all_samples)} sample files in {output_dir}")
    
    index_data = []
    for i, sample in enumerate(all_samples, 1):
        sample_id = f"{i:03d}"
        
        # Generate YAML (in root)
        yaml_path = output_dir / f"sample_{sample_id}.yaml"
        _write_yaml(sample, i, yaml_path)
        
        # Generate HTML (in html/ subdirectory)
        html_path = html_dir / f"sample_{sample_id}.html"
        _write_html(sample, i, html_path)
        
        index_data.append({
            "id": i,
            "source": sample["source"],
            "category": sample["category"],
            "yaml": f"sample_{sample_id}.yaml",
            "html": f"html/sample_{sample_id}.html",
        })
    
    # Generate index.html (in root)
    index_path = output_dir / "index.html"
    _write_index(index_data, index_path)
    
    click.echo(f"\n✓ Done! Generated {len(all_samples)} samples")
    click.echo(f"  - SFT: {len(sft_samples)}")
    click.echo(f"  - DPO: {len(dpo_samples)}")
    click.echo(f"  - Index: {index_path}")


def _sample_from_sft(
    sft_records: list[dict],
    answer_lookup: dict[str, dict],
    n: int,
) -> list[dict[str, Any]]:
    """Sample n records from SFT data, matching with original answers."""
    samples = []
    
    # Try to match each SFT record with original answer
    matched = []
    for rec in sft_records:
        messages = rec.get("messages", [])
        if len(messages) < 3:
            continue
        
        system_msg = messages[0].get("content", "")
        question = messages[1].get("content", "")
        
        # Look up original answer
        answer_rec = answer_lookup.get(question)
        if answer_rec:
            matched.append({
                "system_message": system_msg,
                "question": question,
                "answer": answer_rec["answer"],
                "source": answer_rec.get("source", "unknown"),
                "category": "SFT",
            })
    
    # Sample n records
    if len(matched) <= n:
        return matched
    
    return random.sample(matched, n)


def _sample_from_dpo(
    dpo_records: list[dict],
    answer_lookup: dict[str, dict],
    n: int,
) -> list[dict[str, Any]]:
    """Sample n records from DPO data, extracting rejected (original) answer."""
    samples = []
    
    for rec in dpo_records:
        prompt = rec.get("prompt", "")
        rejected = rec.get("rejected", "")
        
        # Parse prompt: system_message \n\n question \n\n
        parts = prompt.split("\n\n", 1)
        if len(parts) < 2:
            continue
        
        system_msg = parts[0].strip()
        question = parts[1].strip()
        
        # Look up to get source (optional, use rejected answer directly)
        answer_rec = answer_lookup.get(question)
        source = answer_rec.get("source", "unknown") if answer_rec else "unknown"
        
        samples.append({
            "system_message": system_msg,
            "question": question,
            "answer": rejected,
            "source": source,
            "category": "DPO",
        })
    
    # Sample n records
    if len(samples) <= n:
        return samples
    
    return random.sample(samples, n)


def _write_yaml(sample: dict[str, Any], sample_id: int, output_path: Path) -> None:
    """Write a single sample to YAML file."""
    yaml_data = {
        "sample_id": sample_id,
        "source": sample["source"],
        "category": sample["category"],
        "system_message": sample["system_message"],
        "question": sample["question"],
        "answer": sample["answer"],
        "human_score": None,
        "human_notes": "",
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def _write_html(sample: dict[str, Any], sample_id: int, output_path: Path) -> None:
    """Write a single sample to HTML file with markdown rendering."""
    md = markdown.Markdown(extensions=["fenced_code", "tables", "nl2br"])
    
    system_html = md.convert(sample["system_message"])
    question_html = md.convert(sample["question"])
    answer_html = md.convert(sample["answer"])
    
    html_content = f"""<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample {sample_id:03d}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
            color: #007acc;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 15px 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }}
        .metadata strong {{
            color: #555;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-left: 3px solid #007acc;
            border-radius: 3px;
        }}
        .section h2 {{
            margin-top: 0;
            color: #007acc;
            font-size: 1.4em;
        }}
        pre {{
            background: #f0f0f0;
            padding: 15px;
            overflow-x: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 0.9em;
        }}
        pre code {{
            background: transparent;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: #f0f0f0;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #007acc;
            text-decoration: none;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <a href="../index.html" class="back-link">← Back to Index</a>
    
    <h1>Sample {sample_id:03d}</h1>
    
    <div class="metadata">
        <strong>Source:</strong> {sample["source"]}<br>
        <strong>Category:</strong> {sample["category"]}
    </div>
    
    <div class="section">
        <h2>System Context</h2>
        {system_html}
    </div>
    
    <div class="section">
        <h2>Question</h2>
        {question_html}
    </div>
    
    <div class="section">
        <h2>Answer</h2>
        {answer_html}
    </div>
    
    <div class="metadata" style="margin-top: 30px; background: #fff3cd; border-left-color: #ffc107;">
        <strong>Human Evaluation:</strong><br>
        Score: _____ / 10<br>
        Notes: _______________________
    </div>
</body>
</html>
"""
    
    output_path.write_text(html_content, encoding="utf-8")


def _write_index(index_data: list[dict], output_path: Path) -> None:
    """Write index.html with links to all samples."""
    rows = []
    for item in index_data:
        # Truncate source if too long
        source = item["source"]
        if len(source) > 60:
            source = "..." + source[-57:]
        
        rows.append(f"""
        <tr>
            <td>{item["id"]}</td>
            <td>{source}</td>
            <td>{item["category"]}</td>
            <td>
                <a href="{item["html"]}">View HTML</a> |
                <a href="{item["yaml"]}">View YAML</a>
            </td>
        </tr>
        """)
    
    rows_html = "\n".join(rows)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation Samples - Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1 {{
            color: #007acc;
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }}
        .summary {{
            background: #f5f5f5;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #007acc;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        a {{
            color: #007acc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-sft {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-dpo {{
            background: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <h1>Human Evaluation Samples</h1>
    
    <div class="summary">
        <strong>Total Samples:</strong> {len(index_data)}<br>
        <strong>SFT Samples:</strong> {sum(1 for x in index_data if x["category"] == "SFT")}<br>
        <strong>DPO Samples:</strong> {sum(1 for x in index_data if x["category"] == "DPO")}
    </div>
    
    <table>
        <thead>
            <tr>
                <th style="width: 60px;">ID</th>
                <th>Source</th>
                <th style="width: 100px;">Category</th>
                <th style="width: 180px;">Links</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    
    <div class="summary" style="margin-top: 30px; background: #fff3cd; border-left-color: #ffc107;">
        <strong>Instructions:</strong><br>
        1. Open each HTML file to review the question and answer<br>
        2. Rate the answer quality on a scale of 1-10<br>
        3. Add your score to the corresponding YAML file in the <code>human_score</code> field<br>
        4. Optionally add notes in the <code>human_notes</code> field
    </div>
</body>
</html>
"""
    
    output_path.write_text(html_content, encoding="utf-8")
