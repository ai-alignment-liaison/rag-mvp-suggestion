from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
import httpx

# Import ingestion functions
try:
    from ingestion_pipeline import load_mit_sheet, row_to_docs, pdf_to_docs, PAPERS_DIR, CHROMA_PERSIST_DIR, EMBED
except ImportError as e:
    print(f"Warning: Could not import ingestion_pipeline: {e}")
    print("Some data loading features may not work.")
    PAPERS_DIR = Path("./papers")
    CHROMA_PERSIST_DIR = Path("./vector_store")

load_dotenv()

# ───────────────────────── Pydantic Schemas ──────────────────────
class UserProfileSummary(BaseModel):
    """Schema for the summarizer's output containing user values and AI understanding."""
    user_values: str = Field(
        description="Summary of the user's organizational values, priorities, and ethical considerations"
    )
    ai_understanding: str = Field(
        description="Summary of the user's understanding of AI, their experience level, and technical background"
    )

class SearchQueries(BaseModel):
    """Schema for search queries generated from user profile."""
    queries: List[str] = Field(
        description="List of search queries relevant to the user profile, focusing on regional laws, industry risks, regulations, and specific concerns",
        min_items=2,
        max_items=5
    )

# ───────────────────────── Configuration ─────────────────────────
def load_prompts():
    """Load prompts from YAML configuration file with error handling."""
    try:
        with open("configurations/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: configurations/prompts.yaml not found. Creating fallback prompts.")
        return {
            "freeform_interview": "Please tell me more about your organization and AI needs.",
            "summarizer": "Please summarize the conversation as JSON with keys 'user_values' and 'ai_understanding'.",
            "search_system": "Generate 2 short search queries relevant to this profile. Return JSON list.",
            "writer_system": "Write a 700-word actionable Responsible AI strategy.",
            "reviewer_system": "Review and improve the following Responsible AI strategy."
        }
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}. Using fallback prompts.")
        return {}

prompts = load_prompts()

# ───────────────────────── Resources ─────────────────────────
try:
    EMBED = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en", encode_kwargs={"normalize_embeddings": True})
except Exception as e:
    print(f"Warning: Could not initialize embeddings: {e}")
    EMBED = None

COL_DIR = Path("./vector_store")
COL_DIR.mkdir(parents=True, exist_ok=True)

try:
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.3,
        http_client=httpx.Client(proxies=None)
    )
except Exception as e:
    print(f"Error: Could not initialize OpenAI client: {e}")
    print("Please check your OPENAI_API_KEY in .env file.")
    sys.exit(1)

# ───────────────────────── State Dataclass ───────────────────
@dataclass
class GraphState:
    conversation_history: List[Union[AIMessage, HumanMessage]] = field(default_factory=list)
    ai_messages: List[AIMessage] = field(default_factory=list)
    user_messages: List[HumanMessage] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    evidences: Optional[List[Dict]] = None
    strategy: Optional[str] = None
    freeform_question_count: int = 0
    databases_available: Dict[str, bool] = field(default_factory=lambda: {"risks": False, "papers": False})

# ───────────────────────── Data Loading Functions ─────────────
def load_mit_risks_database() -> bool:
    """Try to load MIT risks database. Returns True if successful."""
    try:
        print("📊 Loading MIT AI Risks database...")
        
        # Check if environment variable is set
        sheet_url = os.getenv("MIT_RISKS_SHEET_URL")
        if not sheet_url:
            print("⚠️  MIT_RISKS_SHEET_URL not found in environment. Skipping MIT risks.")
            return False
        
        # Try to load the sheet
        from ingestion_pipeline import load_mit_sheet, row_to_docs
        
        df = load_mit_sheet(sheet_url)
        if df.empty:
            print("⚠️  MIT risks sheet is empty. Skipping.")
            return False
        
        # Convert to documents
        mit_docs = []
        for _, row in df.iterrows():
            mit_docs.extend(row_to_docs(row))
        
        if not mit_docs:
            print("⚠️  No documents extracted from MIT risks sheet. Skipping.")
            return False
        
        # Create/update ChromaDB collection
        if EMBED:
            Chroma.from_documents(
                documents=mit_docs,
                embedding=EMBED,
                collection_name="mit_ai_risks",
                persist_directory=str(COL_DIR)
            )
            print(f"✅ Loaded {len(mit_docs)} documents to MIT AI Risks database")
            return True
        else:
            print("⚠️  Embeddings not available. Skipping MIT risks.")
            return False
            
    except Exception as e:
        print(f"⚠️  Error loading MIT risks database: {e}")
        print("Continuing without MIT risks data...")
        return False

def load_papers_database() -> bool:
    """Try to load papers database. Returns True if successful."""
    try:
        print("📚 Loading Responsible AI Papers database...")
        
        if not PAPERS_DIR.exists() or not PAPERS_DIR.is_dir():
            print(f"⚠️  Papers directory not found: {PAPERS_DIR}. Skipping papers.")
            return False
        
        # Find PDF files
        pdf_files = list(PAPERS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {PAPERS_DIR}. Skipping papers.")
            return False
        
        # Convert PDFs to documents
        from ingestion_pipeline import pdf_to_docs
        
        pdf_docs = []
        for pdf_file in pdf_files:
            try:
                docs = pdf_to_docs(pdf_file)
                pdf_docs.extend(docs)
                print(f"   📄 Processed: {pdf_file.name}")
            except Exception as e:
                print(f"   ⚠️  Error processing {pdf_file.name}: {e}")
                continue
        
        if not pdf_docs:
            print("⚠️  No documents extracted from PDF files. Skipping papers.")
            return False
        
        # Create/update ChromaDB collection
        if EMBED:
            Chroma.from_documents(
                documents=pdf_docs,
                embedding=EMBED,
                collection_name="responsible_ai_papers",
                persist_directory=str(COL_DIR)
            )
            print(f"✅ Loaded {len(pdf_docs)} documents to Papers database")
            return True
        else:
            print("⚠️  Embeddings not available. Skipping papers.")
            return False
            
    except Exception as e:
        print(f"⚠️  Error loading papers database: {e}")
        print("Continuing without papers data...")
        return False

# ───────────────────────── Interview Functions ────────────────
QUESTIONS = [
    "1. What industry does your organisation operate in?",
    "2. In which region or geographical area do you primarily serve?",
    "3. Who are your main customers or target audience?",
    "4. What specific AI use‑cases are you planning to implement?",
    "5. What key risks or organisational values should guide your approach?",
]

def predefined_interview(state: GraphState):
    """Handle predefined interview questions with skip functionality."""
    try:
        # Record user's latest answer
        if state.user_messages:
            if state.answers is None:
                state.answers = []
            
            user_response = state.user_messages[0].content.strip()
            
            # Check for skip command
            if user_response.lower() == "skip":
                print("🔄 User requested to skip to freeform interview")
                # Fill remaining answers with "Not specified"
                while len(state.answers) < 5:
                    state.answers.append("Not specified")
                # Create basic profile
                state.profile = {
                    "industry": state.answers[0] if len(state.answers) > 0 else "Not specified",
                    "region": state.answers[1] if len(state.answers) > 1 else "Not specified", 
                    "audience": state.answers[2] if len(state.answers) > 2 else "Not specified",
                    "use_cases": state.answers[3] if len(state.answers) > 3 else "Not specified",
                    "risk_hypotheses": state.answers[4] if len(state.answers) > 4 else "Not specified",
                }
                state.user_messages = []
                state.ai_messages = []
                return state
            
            state.answers.append(user_response)
            state.user_messages = []

        current_answers = state.answers if state.answers is not None else []

        if len(current_answers) >= 5:  # Finished
            ind, reg, aud, use, risk = current_answers
            state.profile = {
                "industry": ind, "region": reg, "audience": aud,
                "use_cases": use, "risk_hypotheses": risk,
            }
            state.ai_messages = []
        else:
            next_q = QUESTIONS[len(current_answers)]
            skip_hint = " (Type 'skip' to jump to freeform interview)" if len(current_answers) > 0 else ""
            state.ai_messages = [AIMessage(content=next_q + skip_hint)]
            
        return state
        
    except Exception as e:
        print(f"Error in predefined interview: {e}")
        # Fallback: create minimal profile and continue
        state.profile = {"industry": "Not specified", "region": "Not specified", 
                        "audience": "Not specified", "use_cases": "Not specified", 
                        "risk_hypotheses": "Not specified"}
        state.ai_messages = []
        return state

# Create prompt templates with error handling
try:
    freeform_prompt = ChatPromptTemplate.from_template(prompts.get("freeform_interview", "Please tell me more about your needs."))
    # Create structured LLMs
    summarizer_llm = llm.with_structured_output(UserProfileSummary)
    search_llm = llm.with_structured_output(SearchQueries)
except Exception as e:
    print(f"Warning: Error creating prompt templates: {e}")
    # Create fallback templates
    freeform_prompt = ChatPromptTemplate.from_template("Please tell me more about your organization and AI needs.")
    summarizer_llm = None
    search_llm = None

def freeform_interview(state: GraphState) -> GraphState:
    """Handle freeform interview phase."""
    try:
        # Build history on first entry
        if not state.conversation_history:
            history = []
            for i, answer in enumerate(state.answers or []):
                if i < len(QUESTIONS):
                    history.append(AIMessage(content=QUESTIONS[i]))
                    history.append(HumanMessage(content=answer))
            state.conversation_history = history
            state.conversation_history.append(AIMessage(content="Thanks for that information. Let's talk in some more detail now."))

        # Append user message to history
        if state.user_messages:
            state.conversation_history.extend(state.user_messages)
            state.user_messages = []

        # Generate next question
        history_str = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.conversation_history)
        
        try:
            next_q_prompt = freeform_prompt.format_prompt(history=history_str)
            response = llm.invoke(next_q_prompt.to_messages())
            ai_message = AIMessage(content=response.content)
        except Exception as e:
            print(f"Error generating freeform question: {e}")
            ai_message = AIMessage(content="Thank you for your responses. Let's proceed to create your strategy.")
            
        state.ai_messages = [ai_message]
        state.conversation_history.append(ai_message)
        state.freeform_question_count += 1
        
        return state
        
    except Exception as e:
        print(f"Error in freeform interview: {e}")
        state.ai_messages = [AIMessage(content="Thank you for your responses. Let's proceed to create your strategy.")]
        return state

def summarize_profile(state: GraphState) -> GraphState:
    """Summarize user profile from conversation using structured output."""
    try:
        history_str = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.conversation_history)
        
        try:
            if summarizer_llm:
                # Use structured output
                summarizer_prompt_text = prompts.get("summarizer", 
                    "Based on the following conversation, extract the user's values and their understanding of AI.")
                
                # Create the prompt for structured output
                full_prompt = f"{summarizer_prompt_text}\n\nConversation:\n{history_str}"
                
                # Get structured response
                summary_response = summarizer_llm.invoke(full_prompt)
                
                # Convert Pydantic model to dict
                new_info = {
                    "user_values": summary_response.user_values,
                    "ai_understanding": summary_response.ai_understanding
                }
                
                if state.profile:
                    state.profile.update(new_info)
                else:
                    state.profile = new_info
                    
                print("✅ Profile summarized using structured output")
                
            else:
                # Fallback to regular LLM if structured output failed to initialize
                print("⚠️  Using fallback summarization method")
                fallback_prompt = f"Based on the following conversation, extract the user's values and AI understanding as JSON:\n\nConversation:\n{history_str}\n\nRespond with JSON containing 'user_values' and 'ai_understanding' keys."
                
                response = llm.invoke(fallback_prompt)
                new_info = json.loads(response.content)
                
                if state.profile:
                    state.profile.update(new_info)
                else:
                    state.profile = new_info
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error summarizing profile: {e}")
            # Fallback: ensure profile exists with basic structure
            if state.profile is None:
                state.profile = {"user_values": "Not specified", "ai_understanding": "Not specified"}
            elif "user_values" not in state.profile:
                state.profile["user_values"] = "Not specified"
                state.profile["ai_understanding"] = "Not specified"
        
        state.ai_messages = [AIMessage(content="Thanks! I've gathered all the necessary information. Now, I'll build a draft strategy for you.")]
        return state
        
    except Exception as e:
        print(f"Error in profile summarization: {e}")
        if state.profile is None:
            state.profile = {"error": "Could not summarize profile", "user_values": "Not specified", "ai_understanding": "Not specified"}
        state.ai_messages = [AIMessage(content="Let's proceed to create your strategy.")]
        return state

# ───────────────────────── Retrieval and Strategy Functions ────────────
def retriever(state: GraphState) -> GraphState:
    """Retrieve relevant documents from available databases."""
    try:
        if state.profile is None:
            return state

        # Try to connect to databases
        risks_db = None
        papers_db = None
        
        try:
            if state.databases_available.get("risks", False) and EMBED:
                risks_db = Chroma(collection_name="mit_ai_risks", persist_directory=str(COL_DIR), embedding_function=EMBED)
        except Exception as e:
            print(f"Could not connect to risks database: {e}")
            
        try:
            if state.databases_available.get("papers", False) and EMBED:
                papers_db = Chroma(collection_name="responsible_ai_papers", persist_directory=str(COL_DIR), embedding_function=EMBED)
        except Exception as e:
            print(f"Could not connect to papers database: {e}")

        if not risks_db and not papers_db:
            print("⚠️  No databases available for retrieval. Continuing without evidence.")
            state.evidences = []
            return state

        # Generate search queries using structured output
        try:
            if search_llm:
                # Use structured output for search queries
                search_prompt_text = prompts.get("search_system", "Generate 2 short search queries relevant to this profile.")
                profile_json = json.dumps(state.profile, indent=2)
                
                # Create the prompt for structured output
                full_search_prompt = f"{search_prompt_text}\n\nUser Profile:\n{profile_json}"
                
                # Get structured response
                search_response = search_llm.invoke(full_search_prompt)
                queries = search_response.queries
                
                print(f"✅ Generated {len(queries)} search queries using structured output")
                
            else:
                # Fallback to regular LLM if structured output failed to initialize
                print("⚠️  Using fallback search query generation")
                search_prompt = ChatPromptTemplate.from_messages([
                    ("system", prompts.get("search_system", "Generate search queries")),
                    ("human", "{p}"),
                ])
                
                q = llm.invoke(search_prompt.format_prompt(p=json.dumps(state.profile, indent=2)).to_messages()).content
                queries = json.loads(q) if q.startswith('[') else [q.strip()]
                
        except Exception as e:
            print(f"Error generating search queries: {e}")
            # Fallback queries based on profile content
            fallback_queries = ["responsible AI", "AI ethics"]
            if state.profile:
                industry = state.profile.get("industry", "")
                if industry and industry != "Not specified":
                    fallback_queries = [f"AI ethics {industry}", "responsible AI implementation"]
            queries = fallback_queries

        # Search databases
        evidences = []
        for query in queries[:3]:  # Limit to 3 queries for performance
            try:
                if risks_db:
                    for doc, score in risks_db.similarity_search_with_score(query, k=3):
                        evidences.append({"content": doc.page_content, "metadata": doc.metadata, "score": score})
                if papers_db:
                    for doc, score in papers_db.similarity_search_with_score(query, k=3):
                        evidences.append({"content": doc.page_content, "metadata": doc.metadata, "score": score})
            except Exception as e:
                print(f"Error searching with query '{query}': {e}")
                continue

        # Sort and limit results
        evidences = sorted(evidences, key=lambda e: e["score"])[:8]
        state.evidences = evidences
        print(f"📋 Retrieved {len(evidences)} evidence documents")
        
        return state
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        state.evidences = []
        return state

def writer(state: GraphState):
    """Generate initial strategy draft."""
    try:
        if not state.evidences:
            state.evidences = []
            
        writer_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get("writer_system", "Write a Responsible AI strategy")),
            ("human", "PROFILE:\n{prof}"),
            ("human", "EVIDENCES:\n{ev}"),
        ])

        refs = [f"[S{i + 1}] {e.get('metadata', {}).get('title', 'Unknown')[:80]}" for i, e in enumerate(state.evidences)]
        ev_snips = "\n".join(f"[S{i + 1}] {e['content'][:250]}…" for i, e in enumerate(state.evidences))
        
        try:
            draft = llm.invoke(
                writer_prompt.format_prompt(
                    prof=json.dumps(state.profile, indent=2), 
                    ev=ev_snips
                ).to_messages()
            ).content
            
            state.strategy = draft
            if refs and any(ref.strip() for ref in refs):
                state.strategy += "\n\nReferences:\n" + "\n".join(ref for ref in refs if ref.strip())
                
        except Exception as e:
            print(f"Error generating strategy: {e}")
            state.strategy = f"# Responsible AI Strategy\n\nBased on your profile: {json.dumps(state.profile, indent=2)}\n\nA comprehensive Responsible AI strategy should address ethical considerations, risk management, and implementation guidelines specific to your organization."
        
        print("📝 Strategy draft created")
        return state
        
    except Exception as e:
        print(f"Error in writer: {e}")
        state.strategy = "Error generating strategy. Please try again."
        return state

def reviewer(state: GraphState) -> GraphState:
    """Review and improve the strategy."""
    try:
        if not state.strategy or not state.profile:
            return state

        reviewer_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get("reviewer_system", "Review and improve the strategy")),
            ("human", "PROFILE:\n{prof}"),
            ("human", "EVIDENCES:\n{ev}"),
            ("human", "CURRENT STRATEGY:\n{strat}"),
        ])

        ev_snips = "\n".join(f"[S{i + 1}] {e['content'][:250]}…" for i, e in enumerate(state.evidences or []))
        
        # Separate references from main strategy
        strategy_parts = state.strategy.split("\n\nReferences:\n")
        main_strategy_text = strategy_parts[0]
        references_text = ""
        if len(strategy_parts) > 1:
            references_text = "\n\nReferences:\n" + strategy_parts[1]

        try:
            improved_strategy = llm.invoke(
                reviewer_prompt.format_prompt(
                    prof=json.dumps(state.profile, indent=2), 
                    ev=ev_snips, 
                    strat=main_strategy_text
                ).to_messages()
            ).content

            state.strategy = improved_strategy + references_text
            
        except Exception as e:
            print(f"Error reviewing strategy: {e}")
            # Keep original strategy if review fails
            
        state.ai_messages = [AIMessage(content="✅ Your Responsible AI strategy has been completed and refined!")]
        print("🔍 Strategy reviewed and improved")
        return state
        
    except Exception as e:
        print(f"Error in reviewer: {e}")
        return state

# ───────────────────────── Graph Routing Functions ────────────
def route_interviews(state: GraphState) -> str:
    """Route to correct interview stage."""
    if state.profile is not None:
        return "freeform_interview"
    else:
        return "predefined_interview"

def after_predefined_interview_condition(state: GraphState) -> str:
    """Decide next step after predefined interview."""
    if state.profile is not None:
        return "freeform_interview"
    else:
        return "__END__"

def after_freeform_interview_condition(state: GraphState) -> str:
    """Check if freeform interview should continue."""
    try:
        # Check question limit
        if state.freeform_question_count >= 5:
            return "summarize_profile"

        # Check for stop request
        if state.user_messages:
            last_user_message = state.user_messages[-1].content.lower()
            stop_phrases = ["stop", "that's enough", "i don't want to answer", "quit", "exit"]
            if any(phrase in last_user_message for phrase in stop_phrases):
                return "summarize_profile"

        # Check if AI indicated interview is over by checking last AI message
        if state.conversation_history:
            # Get the last message in conversation history
            last_message = state.conversation_history[-1]
            if isinstance(last_message, AIMessage) and "the interview is over".lower() in last_message.content.lower():
                return "summarize_profile"
        
        return "__END__"
        
    except Exception as e:
        print(f"Error in interview condition check: {e}")
        return "summarize_profile"

# ───────────────────────── Graph Creation ────────────────
def create_workflow(databases_available: Dict[str, bool]) -> StateGraph:
    """Create the LangGraph workflow."""
    try:
        graph = StateGraph(GraphState)

        # Add all nodes
        graph.add_node("predefined_interview", predefined_interview)
        graph.add_node("freeform_interview", freeform_interview)
        graph.add_node("summarize_profile", summarize_profile)
        graph.add_node("retriever", retriever)
        graph.add_node("writer", writer)
        graph.add_node("reviewer", reviewer)

        # Set entry point
        graph.set_conditional_entry_point(
            route_interviews,
            {
                "predefined_interview": "predefined_interview",
                "freeform_interview": "freeform_interview",
            },
        )

        # Define edges
        graph.add_conditional_edges(
            "predefined_interview",
            after_predefined_interview_condition,
            {
                "freeform_interview": "freeform_interview",
                "__END__": END
            }
        )

        graph.add_conditional_edges(
            "freeform_interview",
            after_freeform_interview_condition,
            {
                "summarize_profile": "summarize_profile",
                "__END__": END,
            },
        )

        # Connect the rest of the pipeline
        graph.add_edge("summarize_profile", "retriever")
        graph.add_edge("retriever", "writer")
        graph.add_edge("writer", "reviewer")
        graph.set_finish_point("reviewer")

        return graph.compile()
        
    except Exception as e:
        print(f"Error creating workflow: {e}")
        raise

def save_strategy_to_file(strategy: str, profile: Dict[str, Any]) -> str:
    """Save the strategy to a file and return the filename."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        industry = profile.get("industry", "unknown").replace(" ", "_").lower()
        filename = f"responsible_ai_strategy_{industry}_{timestamp}.md"
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        # Prepare content
        content = f"""# Responsible AI Strategy

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## User Profile
```json
{json.dumps(profile, indent=2)}
```

## Strategy

{strategy}

---
*This strategy was generated using RAG-based AI assistance.*
"""
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"💾 Strategy saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"Error saving strategy to file: {e}")
        # Fallback: try to save with basic filename
        try:
            fallback_path = Path("responsible_ai_strategy.md")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(strategy)
            return str(fallback_path)
        except Exception as e2:
            print(f"Could not save strategy: {e2}")
            return ""

# ───────────────────────── Main Orchestration ─────────────────
async def main():
    """Main orchestration function."""
    print("🚀 Starting Responsible AI Strategy Generator")
    print("=" * 50)
    
    # Step 1: Load databases
    print("\n📂 STEP 1: Loading databases...")
    print("ℹ️  Note: ChromaDB telemetry warnings are harmless and can be ignored.")
    databases_available = {
        "risks": load_mit_risks_database(),
        "papers": load_papers_database()
    }
    
    if not any(databases_available.values()):
        print("⚠️  No databases could be loaded. Proceeding with interview only.")
    
    # Step 2-7: Run interview and strategy generation
    print("\n🗣️  STEP 2: Starting customer interview...")
    
    try:
        # Initialize workflow
        print("🔧 Initializing workflow...")
        workflow = create_workflow(databases_available)
        print("✅ Workflow created successfully")
        
        # Initialize state
        print("📋 Initializing state...")
        state = {
            "user_messages": [], 
            "answers": [], 
            "freeform_question_count": 0,
            "databases_available": databases_available
        }
        print("✅ State initialized successfully")

        # Initial invocation to get first question
        print("🚀 Starting workflow (this may take a moment)...")
        try:
            # Add timeout to prevent hanging
            state = await asyncio.wait_for(workflow.ainvoke(state), timeout=60.0)
            print("✅ Workflow invocation completed")
        except asyncio.TimeoutError:
            print("⚠️  Workflow invocation timed out. This might indicate an issue with LLM connectivity.")
            print("   Please check your OPENAI_API_KEY and internet connection.")
            return
        except Exception as e:
            print(f"❌ Error during workflow invocation: {e}")
            print("   This might be due to LLM connectivity issues or configuration problems.")
            return

        # Interview loop
        while True:
            # Display AI messages
            ai_messages = state.get("ai_messages", [])
            for m in ai_messages:
                print(f"\n🤖 AI: {m.content}")
            state["ai_messages"] = []

            # Check if strategy is complete
            if state.get("strategy"):
                print(f"\n✅ STEP 8: Strategy generation complete!")
                break

            # Get user input
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("👋 Goodbye!")
                return

            # Process user input
            state["user_messages"] = [HumanMessage(content=user_input)]
            state = await workflow.ainvoke(state)

        # Step 8: Save strategy to file
        print("\n💾 STEP 8: Saving strategy...")
        strategy = state.get("strategy", "No strategy generated")
        profile = state.get("profile", {})
        
        filepath = save_strategy_to_file(strategy, profile)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📄 Your Responsible AI strategy has been saved to: {filepath}")
        
        # Display final strategy
        print("\n" + "="*50)
        print("FINAL STRATEGY:")
        print("="*50)
        print(strategy)
        
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Stack trace:")
        traceback.print_exc()
        print("\nThe process encountered an error, but any partial results may still be useful.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
