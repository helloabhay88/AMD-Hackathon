import os
import json
import re
import torch
from typing import TypedDict, List, Dict, Any

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LangGraph Imports
from langgraph.graph import StateGraph, END

# Global dictionary to cache multiple local Hugging Face pipelines
_local_hf_pipelines = {}

# --- STATE DEFINITION ---
class AuditState(TypedDict):
    document_text: str
    rules: List[Dict[str, Any]]
    current_rule_index: int
    retrieved_context: List[str]
    current_result: Dict[str, Any]
    final_reports: List[Dict[str, Any]]
    model_name: str
    validator_name: str
    prompt_tokens: int
    generated_tokens: int

# --- UTILITY: JSON CLEANER ---
def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Extracts and parses JSON from a model's response, handling markdown blocks."""
    # Search for json code block first
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Check for generic code block
        match_generic = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match_generic:
            json_str = match_generic.group(1).strip()
        else:
            json_str = text.strip()
            
    # Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: simple regex search if JSON parsing fails completely
        status_match = re.search(r'"status":\s*"([^"]+)"', json_str)
        reason_match = re.search(r'"reason":\s*"([^"]+)"', json_str)
        evidence_match = re.search(r'"evidence":\s*"([^"]+)"', json_str)
        
        status = status_match.group(1) if status_match else "WARNING"
        reason = reason_match.group(1) if reason_match else "Failed to parse model output as JSON."
        evidence = evidence_match.group(1) if evidence_match else None
        
        return {
            "status": status,
            "reason": reason,
            "evidence": evidence
        }

# --- LLM RUNNER WRAPPER ---
def run_llm_inference(model_name: str, prompt: str) -> tuple[str, int, int]:
    """Invokes the local HuggingFace LLM with ROCm.
    Returns: (generated_text, prompt_tokens, generated_tokens)
    """
    # Local Hugging Face execution (utilizes AMD GPU via ROCm PyTorch)
    try:
        global _local_hf_pipelines
        if model_name not in _local_hf_pipelines:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            print(f"Loading local HuggingFace model: {model_name}...")
            
            # Check for AMD GPU (ROCm)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"ROCm CUDA status: {torch.cuda.is_available()} - Loading model on: {device}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load in 8-bit or 4-bit depending on resources if needed, otherwise default
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            _local_hf_pipelines[model_name] = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.001,
                return_full_text=False
            )
        
        hf_pipeline = _local_hf_pipelines[model_name]
        
        # Format prompt using chat template if available for instruction-tuned models
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = hf_pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = prompt

        # Run text generation
        output = hf_pipeline(formatted_prompt)
        generated_text = output[0]['generated_text']
        
        # Calculate tokens
        try:
            p_tok = len(hf_pipeline.tokenizer.encode(formatted_prompt))
            g_tok = len(hf_pipeline.tokenizer.encode(generated_text))
        except Exception:
            p_tok = 0
            g_tok = 0
            
        return generated_text.strip(), p_tok, g_tok
    except Exception as e:
        warning_msg = f'{{"status": "WARNING", "reason": "Local HF pipeline error: {str(e)}", "evidence": null}}'
        return warning_msg, 0, 0

# --- RAG PIPELINE: CHROMA VECTOR STORE ---
def setup_vector_db(document_text: str) -> Chroma:
    """Chunks the text document and indexes it in ChromaDB."""
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(document_text)
    
    # Embedding model (lightweight, runs locally on ROCm GPU/CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Create VectorDB
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    return db

# --- LANGGRAPH NODE DEFINITIONS ---

def retrieve_node(state: AuditState) -> Dict[str, Any]:
    """Retrieves document chunks matching the current compliance rule."""
    current_rule = state["rules"][state["current_rule_index"]]
    query = f"{current_rule['name']} {current_rule['description']}"
    
    # Setup database using document text in the state
    db = setup_vector_db(state["document_text"])
    docs = db.similarity_search(query, k=3)
    
    retrieved_chunks = [d.page_content for d in docs]
    
    return {
        "retrieved_context": retrieved_chunks
    }

def auditor_node(state: AuditState) -> Dict[str, Any]:
    """Uses LLM to evaluate compliance and extract evidence."""
    current_rule = state["rules"][state["current_rule_index"]]
    context_str = "\n---\n".join(state["retrieved_context"])
    
    prompt = f"""You are an expert financial and regulatory compliance auditor.
Your task is to audit a document against a specific compliance rule.

RULE:
Category: {current_rule['category']}
Rule ID: {current_rule['id']}
Rule Name: {current_rule['name']}
Description: {current_rule['description']}

RETRIEVED DOCUMENT CONTEXT:
{context_str}

Evaluate the document context against the rule. Decide the Status:
- PASS: The document explicitly complies with the rule.
- FAIL: The document explicitly violates the rule.
- WARNING: The document mentions something relevant but there is ambiguity, missing details, or a potential issue.
- N/A: The rule is not mentioned or not applicable to this document.

CRITICAL RULES FOR AUDITING:
1. CHECK FOR NEGATIVE CLAUSES/EXCEPTIONS: Pay extremely close attention to negative statements in the document context (e.g., "no", "not", "except", "without", "never", "no ... was collected/required"). If a rule requires an action (e.g., requiring government-issued photo ID) and the document states that this action was NOT taken or NOT required, the status MUST be FAIL.
2. DO NOT ASSUME COMPLIANCE: Do not assume that general statements of "onboarding complete" or "verification complete" satisfy specific compliance rules if the required method (e.g., photo ID) was explicitly bypassed or not used.
3. EXTRACT EVIDENCE EXACTLY VERBATIM: You MUST extract the EXACT verbatim sentence(s)/quote(s) from the document context that support your decision as "evidence". Do not modify even a single character, punctuation mark, or spacing. If no evidence exists, set it to null.

Provide your response in raw JSON format with the following fields:
{{
    "status": "PASS" | "FAIL" | "WARNING" | "N/A",
    "reason": "Detailed explanation of why the document complies, violates, or is ambiguous relative to the rule.",
    "evidence": "Verbatim quote from the document context supporting this status, or null if not applicable."
}}
JSON:"""

    model_name = state.get("model_name", "Qwen/Qwen2.5-14B-Instruct")
    llm_output, p_tok, g_tok = run_llm_inference(model_name, prompt)
    auditor_result = clean_and_parse_json(llm_output)
    
    # Temporarily store token metrics in current_result so they transfer to validator
    auditor_result["_prompt_tokens"] = p_tok
    auditor_result["_generated_tokens"] = g_tok
    
    return {
        "current_result": auditor_result
    }

def validator_node(state: AuditState) -> Dict[str, Any]:
    """Performs self-reflection verification to eliminate hallucinations and compute score."""
    current_rule = state["rules"][state["current_rule_index"]]
    context_str = "\n---\n".join(state["retrieved_context"])
    auditor_res = state["current_result"]
    
    # Retrieve the Auditor's tokens
    auditor_p_tok = auditor_res.pop("_prompt_tokens", 0)
    auditor_g_tok = auditor_res.pop("_generated_tokens", 0)
    
    prompt = f"""You are a compliance audit validator. Your task is to verify the findings of a primary auditor agent.

COMPLIANCE RULE:
Category: {current_rule['category']}
Rule Name: {current_rule['name']}
Description: {current_rule['description']}

RETRIEVED DOCUMENT CONTEXT:
{context_str}

AUDITOR'S FINDINGS:
Proposed Status: {auditor_res.get('status', 'WARNING')}
Proposed Reason: {auditor_res.get('reason', '')}
Proposed Evidence Quote: {auditor_res.get('evidence', 'None')}

Verify the following:
1. Evidence Verification: Is the proposed evidence quote actually present verbatim in the retrieved document context? (True/False)
2. Logic Verification: Does the proposed evidence quote and reasoning logically support the proposed status? Pay special attention to negative constraints (e.g., "no photo ID collected"). If the auditor proposed PASS but the document context explicitly states that a required item/method was NOT collected/used, then the status is incorrect and must be FAIL.

Calculate a confidence score (0-100) based on:
- 100: Evidence is verbatim, and logic is watertight.
- 70-90: Evidence is verbatim, but there is slight room for interpretation or the reasoning is somewhat soft.
- 30-60: The evidence is only partially matching, or the reasoning doesn't fully support the status.
- 0: The evidence quote is not found in the text at all (hallucinated), or the reasoning is completely contradictory.

Provide your validation response in raw JSON format.

IMPORTANT: The "validated_status" field MUST represent the final compliance status of the contract (PASS if the contract complies, FAIL if the contract violates the rule, WARNING if ambiguous, N/A if not mentioned). For example, if the auditor correctly identified a compliance violation (Proposed Status: FAIL) and you agree that the violation is correct, you MUST set "validated_status" to "FAIL". If the auditor incorrectly passed the rule but you find that a requirement was violated or not met, you MUST override and set "validated_status" to "FAIL".

Response JSON structure:
{{
    "validated_status": "PASS" | "FAIL" | "WARNING" | "N/A",
    "is_evidence_verbatim": true | false,
    "confidence_score": 0-100,
    "validation_reason": "Explanation of your validation check, highlighting any discrepancy or confirming validity."
}}
JSON:"""

    validator_name = state.get("validator_name", "Qwen/Qwen2.5-7B-Instruct")
    llm_output, val_p_tok, val_g_tok = run_llm_inference(validator_name, prompt)
    validator_res = clean_and_parse_json(llm_output)
    
    # Calculate rule total tokens
    rule_p_tok = auditor_p_tok + val_p_tok
    rule_g_tok = auditor_g_tok + val_g_tok
    
    # Merge reports
    report = {
        "rule_id": current_rule["id"],
        "rule_name": current_rule["name"],
        "category": current_rule["category"],
        "description": current_rule["description"],
        "status": validator_res.get("validated_status", auditor_res.get("status", "WARNING")),
        "reason": auditor_res.get("reason", ""),
        "evidence": auditor_res.get("evidence", None),
        "confidence_score": validator_res.get("confidence_score", 50),
        "is_evidence_verbatim": validator_res.get("is_evidence_verbatim", False),
        "validation_notes": validator_res.get("validation_reason", ""),
        "prompt_tokens": rule_p_tok,
        "generated_tokens": rule_g_tok
    }
    
    # Accumulate running totals in state
    total_p = state.get("prompt_tokens", 0) + rule_p_tok
    total_g = state.get("generated_tokens", 0) + rule_g_tok
    
    # Return appended list and updated token state accumulators
    new_reports = list(state.get("final_reports", []))
    new_reports.append(report)
    
    return {
        "final_reports": new_reports,
        "prompt_tokens": total_p,
        "generated_tokens": total_g
    }

# --- LANGGRAPH CONTROL ROUTING ---

def router_edge(state: AuditState) -> str:
    """Conditional router that goes to next rule or terminates."""
    next_idx = state["current_rule_index"] + 1
    if next_idx < len(state["rules"]):
        return "continue"
    return "end"

def increment_index(state: AuditState) -> Dict[str, Any]:
    """Helper node to increment rule index before loop restarts."""
    return {
        "current_rule_index": state["current_rule_index"] + 1
    }

# --- GRAPH BUILDER ---

def build_compliance_graph() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AuditState)
    
    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("audit", auditor_node)
    workflow.add_node("validate", validator_node)
    workflow.add_node("increment", increment_index)
    
    # Set Entry Point
    workflow.set_entry_point("retrieve")
    
    # Add Connections (Edges)
    workflow.add_edge("retrieve", "audit")
    workflow.add_edge("audit", "validate")
    
    # Route from validator node based on rule count
    workflow.add_conditional_edges(
        "validate",
        router_edge,
        {
            "continue": "increment",
            "end": END
        }
    )
    
    workflow.add_edge("increment", "retrieve")
    
    return workflow.compile()

# --- MAIN RUNNER FUNCTION ---
def run_compliance_audit(
    document_text: str,
    rules: List[Dict[str, Any]],
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    validator_name: str = "Qwen/Qwen2.5-7B-Instruct"
) -> List[Dict[str, Any]]:
    """Initializes and executes the LangGraph workflow."""
    app = build_compliance_graph()
    
    initial_state = {
        "document_text": document_text,
        "rules": rules,
        "current_rule_index": 0,
        "retrieved_context": [],
        "current_result": {},
        "final_reports": [],
        "model_name": model_name,
        "validator_name": validator_name,
        "prompt_tokens": 0,
        "generated_tokens": 0
    }
    
    # Execute Graph
    final_state = app.invoke(initial_state)
    
    # Print token usage summary
    total_prompt = final_state.get("prompt_tokens", 0)
    total_gen = final_state.get("generated_tokens", 0)
    print("\n" + "="*40)
    print("        TOKEN USAGE SUMMARY")
    print("="*40)
    print(f"Total Prompt Tokens:    {total_prompt:,}")
    print(f"Total Generated Tokens: {total_gen:,}")
    print(f"Total Tokens Consumed:  {total_prompt + total_gen:,}")
    print("="*40 + "\n")
    
    return final_state["final_reports"]
