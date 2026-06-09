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

# --- STATE DEFINITION ---
class AuditState(TypedDict):
    document_text: str
    rules: List[Dict[str, Any]]
    current_rule_index: int
    retrieved_context: List[str]
    current_result: Dict[str, Any]
    final_reports: List[Dict[str, Any]]
    model_type: str
    model_name: str
    api_key: str

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
def run_llm_inference(state: AuditState, prompt: str) -> str:
    """Invokes the selected LLM (Local HF with ROCm, Gemini, OpenAI, or Mock)."""
    model_type = state.get("model_type", "mock").lower()
    model_name = state.get("model_name", "")
    api_key = state.get("api_key", "")

    if model_type == "mock":
        return get_mock_response(prompt)
        
    elif model_type == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name if model_name else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'{{"status": "WARNING", "reason": "OpenAI error: {str(e)}", "evidence": null}}'
            
    elif model_type == "gemini":
        try:
            # Simple HTTP request to avoid external package setup issues in Jupyter
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name if model_name else 'gemini-1.5-flash'}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
            }
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            res_data = res.json()
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f'{{"status": "WARNING", "reason": "Gemini API error: {str(e)}", "evidence": null}}'
            
    elif model_type == "hf":
        # Local Hugging Face execution (utilizes AMD GPU via ROCm PyTorch)
        try:
            global _local_hf_pipeline
            if '_local_hf_pipeline' not in globals():
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
                _local_hf_pipeline = pipeline(
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.001
                )
            
            # Run text generation
            output = _local_hf_pipeline(prompt)
            generated_text = output[0]['generated_text']
            # Remove prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        except Exception as e:
            return f'{{"status": "WARNING", "reason": "Local HF pipeline error: {str(e)}", "evidence": null}}'
            
    return get_mock_response(prompt)

def get_mock_response(prompt: str) -> str:
    """Deterministic mock responses for offline execution/testing."""
    # Analyze prompt content to simulate LLM thinking
    if "validator" in prompt.lower() or "validation" in prompt.lower():
        # Validator mock logic
        if "SECTION 1" in prompt and "photo ID" in prompt:
            return json.dumps({
                "validated_status": "FAIL",
                "is_evidence_verbatim": True,
                "confidence_score": 95,
                "validation_reason": "Auditor identified correct clause. Verification is failed because utility bill is not a government photo ID."
            })
        elif "SECTION 2" in prompt and "interest rate" in prompt:
            return json.dumps({
                "validated_status": "FAIL",
                "is_evidence_verbatim": True,
                "confidence_score": 95,
                "validation_reason": "Auditor verified correctly. The contract states 9.75% max cap, violating the limit of 8.5%."
            })
        elif "SECTION 3" in prompt and "grace period" in prompt:
            return json.dumps({
                "validated_status": "FAIL",
                "is_evidence_verbatim": True,
                "confidence_score": 95,
                "validation_reason": "Auditor is correct. Contract states 10 calendar days grace period, which violates the 15-day requirement."
            })
        elif "SECTION 4" in prompt and "late payment penalty fee of $75.00" in prompt:
            return json.dumps({
                "validated_status": "FAIL",
                "is_evidence_verbatim": True,
                "confidence_score": 90,
                "validation_reason": "Auditor correctly identified the fee of $75.00, which exceeds the limit of $50."
            })
        elif "SECTION 5" in prompt and "cancel this mortgage transaction" in prompt:
            return json.dumps({
                "validated_status": "PASS",
                "is_evidence_verbatim": True,
                "confidence_score": 98,
                "validation_reason": "Auditor verified correctly. The contract grants 3 business days, matching the rule."
            })
        # Default validator pass
        return json.dumps({
            "validated_status": "PASS",
            "is_evidence_verbatim": True,
            "confidence_score": 90,
            "validation_reason": "Auditor findings verified. Document is compliant."
        })
    else:
        # Auditor mock logic
        if "COMP-001" in prompt or "Interest Rate Cap Limit" in prompt:
            return json.dumps({
                "status": "FAIL",
                "reason": "The mortgage agreement specifies an adjustable interest rate cap of 9.75% per annum, which exceeds the allowed regulatory limit of 8.5% per annum.",
                "evidence": "the maximum adjustable interest rate cap on this mortgage agreement shall not exceed 9.75% per annum"
            })
        elif "COMP-002" in prompt or "Late Payment Grace Period" in prompt:
            return json.dumps({
                "status": "FAIL",
                "reason": "The contract establishes a grace period of only 10 calendar days before a late fee is assessed, violating the consumer protection rule of at least 15 days.",
                "evidence": "The Lender shall allow a grace period of 10 calendar days."
            })
        elif "COMP-003" in prompt or "Identity Verification (KYC)" in prompt:
            return json.dumps({
                "status": "FAIL",
                "reason": "The document states that onboarding was completed using email and utility bill confirmation, and explicitly notes that no physical or government-issued photo identification was collected.",
                "evidence": "No physical or government-issued photo identification was collected or required"
            })
        elif "COMP-004" in prompt or "Late Fee Cap" in prompt:
            return json.dumps({
                "status": "FAIL",
                "reason": "The late fee specified in Section 4 is a flat penalty of $75.00, which violates the regulatory cap of $50.",
                "evidence": "a flat late payment penalty fee of $75.00 shall be charged"
            })
        elif "COMP-005" in prompt or "Right of Rescission" in prompt:
            return json.dumps({
                "status": "PASS",
                "reason": "Section 5 explicitly states that the borrower has the right to cancel the transaction without penalty within three business days from the execution of the agreement.",
                "evidence": "Borrower has the right to cancel this mortgage transaction, without any penalty or obligation, within three (3) business days"
            })
        return json.dumps({
            "status": "N/A",
            "reason": "Rule is not mentioned or applicable in the provided contract context.",
            "evidence": None
        })

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

You MUST extract the EXACT verbatim sentence(s)/quote(s) from the document context that support your decision as "evidence". Do not modify even a single character. If no evidence exists, set it to null.

Provide your response in raw JSON format with the following fields:
{{
    "status": "PASS" | "FAIL" | "WARNING" | "N/A",
    "reason": "Detailed explanation of why the document complies, violates, or is ambiguous relative to the rule.",
    "evidence": "Verbatim quote from the document context supporting this status, or null if not applicable."
}}
JSON:"""

    llm_output = run_llm_inference(state, prompt)
    auditor_result = clean_and_parse_json(llm_output)
    
    return {
        "current_result": auditor_result
    }

def validator_node(state: AuditState) -> Dict[str, Any]:
    """Performs self-reflection verification to eliminate hallucinations and compute score."""
    current_rule = state["rules"][state["current_rule_index"]]
    context_str = "\n---\n".join(state["retrieved_context"])
    auditor_res = state["current_result"]
    
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
2. Logic Verification: Does the proposed evidence quote and reasoning logically support the proposed status? (True/False)

Calculate a confidence score (0-100) based on:
- 100: Evidence is verbatim, and logic is watertight.
- 70-90: Evidence is verbatim, but there is slight room for interpretation or the reasoning is somewhat soft.
- 30-60: The evidence is only partially matching, or the reasoning doesn't fully support the status.
- 0: The evidence quote is not found in the text at all (hallucinated), or the reasoning is completely contradictory.

Provide your validation response in raw JSON format with the following fields:
{{
    "validated_status": "PASS" | "FAIL" | "WARNING" | "N/A",
    "is_evidence_verbatim": true | false,
    "confidence_score": 0-100,
    "validation_reason": "Explanation of your validation check, highlighting any discrepancy or confirming validity."
}}
JSON:"""

    llm_output = run_llm_inference(state, prompt)
    validator_res = clean_and_parse_json(llm_output)
    
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
        "validation_notes": validator_res.get("validation_reason", "")
    }
    
    # Return appended list
    new_reports = list(state.get("final_reports", []))
    new_reports.append(report)
    
    return {
        "final_reports": new_reports
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
    model_type: str = "mock",
    model_name: str = "",
    api_key: str = ""
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
        "model_type": model_type,
        "model_name": model_name,
        "api_key": api_key
    }
    
    # Execute Graph
    final_state = app.invoke(initial_state)
    return final_state["final_reports"]
