import os
import argparse
import json
import pypdf
from typing import List, Dict, Any

# Rich terminal output library
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.text import Text

# Import compliance agent logic
from compliance_agent import run_compliance_audit

console = Console()

def parse_document(file_path: str) -> str:
    """Parses TXT or PDF documents and extracts raw text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        console.print(f"[cyan]Parsing PDF document: {file_path}...[/cyan]")
        try:
            reader = pypdf.PdfReader(file_path)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n--- Page Break ---\n\n".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {str(e)}")
    else:
        # Default to raw text
        console.print(f"[cyan]Reading text document: {file_path}...[/cyan]")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def save_reports(reports: List[Dict[str, Any]], base_path: str, doc_name: str):
    """Saves the audit results as both JSON and Markdown formats."""
    # 1. Save JSON
    json_path = f"{base_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)
        
    # 2. Save Markdown
    md_path = f"{base_path}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AI-Driven Compliance Audit Report\n\n")
        f.write(f"**Target Document**: `{doc_name}`  \n")
        f.write(f"**Audit Generation**: Automated compliance check  \n\n")
        
        # Summary statistics
        total = len(reports)
        passes = sum(1 for r in reports if r["status"] == "PASS")
        fails = sum(1 for r in reports if r["status"] == "FAIL")
        warnings = sum(1 for r in reports if r["status"] == "WARNING")
        nas = sum(1 for r in reports if r["status"] == "N/A")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Rules Checked**: {total}\n")
        f.write(f"- **Compliant (PASS)**: {passes} ✅\n")
        f.write(f"- **Violations (FAIL)**: {fails} ❌\n")
        f.write(f"- **Ambiguities (WARNING)**: {warnings} ⚠️\n")
        f.write(f"- **Not Applicable (N/A)**: {nas} ➖\n\n")
        
        f.write("## Detailed Audit Trail\n\n")
        for i, r in enumerate(reports):
            status_emoji = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL" if r["status"] == "FAIL" else "⚠️ WARNING" if r["status"] == "WARNING" else "➖ N/A"
            f.write(f"### {i+1}. {r['rule_name']} ({r['rule_id']})\n")
            f.write(f"- **Category**: {r['category']}\n")
            f.write(f"- **Requirement**: {r['description']}\n")
            f.write(f"- **Evaluation Status**: **{status_emoji}**\n")
            f.write(f"- **Confidence Score**: {r['confidence_score']}%\n")
            f.write(f"- **Auditor Reasoning**: {r['reason']}\n")
            if r["evidence"]:
                f.write(f"- **Verbatim Evidence Quote**:\n  > \"{r['evidence'].strip()}\"\n")
            else:
                f.write(f"- **Verbatim Evidence Quote**: *None provided*\n")
                
            if r["validation_notes"]:
                f.write(f"- **Verification Check Notes**: *{r['validation_notes']}*\n")
            f.write("\n---\n\n")
            
    console.print(f"[green]Saved JSON report to: {json_path}[/green]")
    console.print(f"[green]Saved Markdown report to: {md_path}[/green]")

def display_dashboard(reports: List[Dict[str, Any]], doc_name: str):
    """Renders a beautiful color-coded dashboard using rich table formatting."""
    total = len(reports)
    passes = sum(1 for r in reports if r["status"] == "PASS")
    fails = sum(1 for r in reports if r["status"] == "FAIL")
    warnings = sum(1 for r in reports if r["status"] == "WARNING")
    nas = sum(1 for r in reports if r["status"] == "N/A")
    
    # Calculate avg confidence
    avg_conf = sum(r["confidence_score"] for r in reports) / total if total > 0 else 0
    
    # Header Panel
    console.print("\n")
    console.print(Panel(
        Text(f"AI-DRIVEN COMPLIANCE AUDIT & COMPLIANCE VALIDATOR\nDocument: {doc_name}", style="bold white", justify="center"),
        subtitle="Powered by LangGraph & AMD ROCm",
        subtitle_align="right",
        style="cyan"
    ))
    
    # Summary Metrics
    summary_text = (
        f"[bold green]PASS: {passes}[/bold green] | "
        f"[bold red]FAIL: {fails}[/bold red] | "
        f"[bold yellow]WARNING: {warnings}[/bold yellow] | "
        f"[bold blue]N/A: {nas}[/bold blue] | "
        f"[bold white]Avg Confidence: {avg_conf:.1f}%[/bold white]"
    )
    console.print(Panel(summary_text, title="Audit Summary", expand=False))
    
    # Table Grid
    table = Table(title="Compliance Audit Trail Matrix", expand=True)
    table.add_column("Rule ID", style="dim", width=10)
    table.add_column("Rule Name & Category", style="bold white")
    table.add_column("Status", justify="center", width=12)
    table.add_column("Conf.", justify="right", width=8)
    table.add_column("Extracted Evidence & Reasoning")
    
    for r in reports:
        # Status styling
        status = r["status"]
        if status == "PASS":
            status_styled = "[bold green]PASS [OK][/bold green]"
        elif status == "FAIL":
            status_styled = "[bold red]FAIL [X][/bold red]"
        elif status == "WARNING":
            status_styled = "[bold yellow]WARN [!][/bold yellow]"
        else:
            status_styled = "[bold blue]N/A [-][/bold blue]"
            
        # Confidence score styling
        conf = r["confidence_score"]
        if conf >= 90:
            conf_styled = f"[bold green]{conf}%[/bold green]"
        elif conf >= 70:
            conf_styled = f"[yellow]{conf}%[/yellow]"
        else:
            conf_styled = f"[red]{conf}%[/red]"
            
        # Evidence / Reason block
        evidence_text = f"[bold italic white]Quote:[/bold italic white] \"{r['evidence']}\"" if r['evidence'] else "[italic dim]No Quote[/italic dim]"
        detail_block = f"{r['reason']}\n{evidence_text}"
        
        # Add to table
        table.add_row(
            r["rule_id"],
            f"{r['rule_name']}\n[dim]{r['category']}[/dim]",
            status_styled,
            conf_styled,
            detail_block
        )
        
    console.print(table)
    console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="AI-Driven Audit & Compliance Validator")
    parser.add_argument("--doc", type=str, default="sample_data/sample_contract.txt", help="Path to text or PDF document")
    parser.add_argument("--rules", type=str, default="sample_data/rules.json", help="Path to JSON rules file")
    parser.add_argument("--model_type", type=str, default="mock", choices=["mock", "hf", "openai", "gemini"], help="LLM Provider")
    parser.add_argument("--model_name", type=str, default="", help="Hugging Face Model ID or Cloud Model Name")
    parser.add_argument("--api_key", type=str, default="", help="API Key for cloud LLM services")
    parser.add_argument("--output", type=str, default="audit_report", help="Prefix of generated output report files")
    
    args = parser.parse_args()
    
    # Read APIs from environment if not passed
    api_key = args.api_key
    if not api_key:
        if args.model_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif args.model_type == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            
    # Load document
    try:
        document_text = parse_document(args.doc)
    except Exception as e:
        console.print(f"[bold red]Error loading document: {str(e)}[/bold red]")
        return
        
    # Load rules
    if not os.path.exists(args.rules):
        console.print(f"[bold red]Rules file not found: {args.rules}[/bold red]")
        return
    with open(args.rules, "r", encoding="utf-8") as f:
        rules = json.load(f)
        
    console.print(f"[yellow]Loaded {len(rules)} compliance rules.[/yellow]")
    
    # Run audit pipeline with progress bar
    reports = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task(description=f"Running LangGraph Compliance Agent using model_type: {args.model_type}...", total=None)
        reports = run_compliance_audit(
            document_text=document_text,
            rules=rules,
            model_type=args.model_type,
            model_name=args.model_name,
            api_key=api_key
        )
        
    # Display table dashboard
    doc_name = os.path.basename(args.doc)
    display_dashboard(reports, doc_name)
    
    # Save files
    save_reports(reports, args.output, doc_name)

if __name__ == "__main__":
    main()
