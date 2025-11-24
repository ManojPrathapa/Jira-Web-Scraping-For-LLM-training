import asyncio
import json
import os
from src.scraper import JiraIngestion
from src.processor import AIProcessor
from src.vector_db import VectorDB
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

def generate_llm_dataset(issues):
    """
    Creates a JSONL file formatted for OpenAI GPT-3.5 Fine-Tuning.
    This demonstrates 'Data Engineering for AI'.
    """
    output_path = "data_lake/fine_tuning_dataset.jsonl"
    os.makedirs("data_lake", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for issue in issues:
            # We teach the LLM to answer questions about the ticket
            training_example = {
                "messages": [
                    {"role": "system", "content": "You are a Jira Assistant. Answer questions based on the ticket details."},
                    {"role": "user", "content": f"What is the status of ticket {issue.key}?"},
                    {"role": "assistant", "content": f"Ticket {issue.key} is currently {issue.status}. It is assigned to {issue.assignee}."}
                ]
            }
            f.write(json.dumps(training_example) + "\n")
    return output_path

async def run_pipeline():
    console.rule("[bold blue]🚀 Starting Data Pipeline")
    
    ingestor = JiraIngestion()
    processor = AIProcessor()
    vectordb = VectorDB()
    
    # 1. EXTRACT
    with console.status("[bold green]Fetching ALL issues...") as status:
        raw_issues = await ingestor.fetch_all_issues()
        console.log(f"✅ Extracted {len(raw_issues)} total issues.")

    # 2. TRANSFORM (LLM Prep)
    jsonl_file = generate_llm_dataset(raw_issues)
    console.log(f"✅ Generated LLM Training Data: [underline]{jsonl_file}[/underline]")

    # 3. EMBED & LOAD
    enriched_issues = []
    for issue in track(raw_issues, description="Processing Vectors..."):
        try:
            enriched = processor.enrich_issue(issue)
            enriched_issues.append(enriched)
        except Exception as e:
            console.print(f"[red]Error on {issue.key}: {e}")

    if enriched_issues:
        vectordb.upsert_batch(enriched_issues)
        console.log(f"✅ Loaded {len(enriched_issues)} vectors to Qdrant.")
    
    console.rule("[bold green]Pipeline Success")

if __name__ == "__main__":
    asyncio.run(run_pipeline())