# rag_graph.py (condensed)
"""LangGraph RAG with deterministic 5‑question interview."""

from __future__ import annotations

import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

import asyncio, json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import langgraph
from langgraph.graph import StateGraph, END
import httpx

load_dotenv()

# ───────────────────────── resources ─────────────────────────
EMBED = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en", encode_kwargs={"normalize_embeddings": True})
COL_DIR = Path("./vector_store")
risks = Chroma(collection_name="mit_ai_risks", persist_directory=str(COL_DIR), embedding_function=EMBED)
papers = Chroma(collection_name="responsible_ai_papers", persist_directory=str(COL_DIR), embedding_function=EMBED)
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.3,
    http_client=httpx.Client(proxies=None)
)


# ───────────────────────── state dataclass ───────────────────
@dataclass
class GraphState:
    ai_messages: List[AIMessage] = field(default_factory=list)
    user_messages: List[HumanMessage] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    evidences: Optional[List[Dict]] = None
    strategy: Optional[str] = None


# ───────────────────────── interview node ────────────────────
QUESTIONS = [
    "1. What industry does your organisation operate in?",
    "2. In which region or geographical area do you primarily serve?",
    "3. Who are your main customers or target audience?",
    "4. What specific AI use‑cases are you planning to implement?",
    "5. What key risks or organisational values should guide your approach?",
]


def interview(state: GraphState):
    # record user's latest answer
    if state.user_messages: 
        # Ensure state.answers is a list if it somehow became None
        if state.answers is None:
            state.answers = []
        state.answers.append(state.user_messages[0].content.strip())
        state.user_messages = []

    # Ensure state.answers is a list for the length check
    current_answers = state.answers if state.answers is not None else []

    if len(current_answers) >= 5:  # finished
        ind, reg, aud, use, risk = current_answers
        state.profile = {
            "industry": ind, "region": reg, "audience": aud,
            "use_cases": use, "risk_hypotheses": risk,
        }
        state.ai_messages = []
    else:
        next_q = QUESTIONS[len(current_answers)]
        state.ai_messages = [AIMessage(content=next_q)]
    return state


# ───────────────────────── retriever node ───────────────────
search_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 1–2 short search queries relevant to this profile. Return JSON list."),
    ("human", "{p}"),
])


def retriever(state: GraphState) -> GraphState:
    # Skip if profile not yet collected (first few ticks)
    if state.profile is None:
        return state

    search_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 1–2 short search queries relevant to this profile. Return JSON list."),
        ("human", "{p}"),
    ])

    q = llm.invoke(search_prompt.format_prompt(p=json.dumps(state.profile, indent=2)).to_messages()).content
    try:
        queries = json.loads(q)
    except json.JSONDecodeError:
        queries = [q.strip()]

    evidences: List[Dict[str, Any]] = []
    for query in queries:
        for col in (risks, papers):
            for doc, score in col.similarity_search_with_score(query, k=6):
                evidences.append({"content": doc.page_content, "metadata": doc.metadata, "score": score})
    evidences = sorted(evidences, key=lambda e: e["score"])[:8]
    state.evidences = evidences
    return state


# ───────────────────────── writer & refiner ──────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a 700‑word actionable Responsible‑AI strategy. Cite as [S1]…"),
    ("human", "PROFILE:\n{prof}"),
    ("human", "EVIDENCES:\n{ev}"),
])


def writer(state: GraphState):
    refs = [f"[S{i + 1}] {e['metadata'].get('title', '')[:80]}" for i, e in enumerate(state.evidences)]
    ev_snips = "\n".join(f"[S{i + 1}] {e['content'][:250]}…" for i, e in enumerate(state.evidences))
    draft = llm.invoke(
        writer_prompt.format_prompt(prof=json.dumps(state.profile, indent=2), ev=ev_snips).to_messages()).content
    state.strategy = draft + "\n\nReferences:\n" + "\n".join(refs)
    return state


# ───────────────────────── reviewer node ─────────────────────
reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a meticulous editor. Review the following Responsible-AI strategy and refine it for clarity, conciseness, impact, and actionable insights. Ensure it directly addresses the user's profile and the provided evidence. Maintain a professional tone and the ~700-word length. Output only the improved strategy, including the original references section if it exists."),
    ("human", "PROFILE:\n{prof}"),
    ("human", "EVIDENCES:\n{ev}"),
    ("human", "CURRENT STRATEGY:\n{strat}"),
])


def reviewer(state: GraphState) -> GraphState:
    if not state.strategy or not state.profile or not state.evidences:
        # Should not happen if graph is wired correctly
        return state

    ev_snips = "\n".join(f"[S{i + 1}] {e['content'][:250]}…" for i, e in enumerate(state.evidences))
    
    # Separate references from the main strategy text if they exist
    strategy_parts = state.strategy.split("\n\nReferences:\n")
    main_strategy_text = strategy_parts[0]
    references_text = ""
    if len(strategy_parts) > 1:
        references_text = "\n\nReferences:\n" + strategy_parts[1]

    improved_strategy = llm.invoke(
        reviewer_prompt.format_prompt(
            prof=json.dumps(state.profile, indent=2), 
            ev=ev_snips, 
            strat=main_strategy_text
        ).to_messages()
    ).content

    state.strategy = improved_strategy + references_text # Append references back
    state.ai_messages = [AIMessage(content="The Responsible AI strategy has been reviewed and refined.")] # Optional: notify user
    return state


# ───────────────────────── graph build ───────────────────────

def after_interview_condition(state: GraphState) -> str:
    """Determines whether to proceed to retrieval or end the current graph run."""
    if state.profile is not None:  # Interview complete, profile is set
        return "retriever"
    else: # Interview ongoing, current graph run effectively ends after interview node outputs question
        return "__END__"

graph = StateGraph(GraphState)

graph.add_node("interview", interview)
graph.add_node("retriever", retriever)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("interview")

# Conditional edge from interview
graph.add_conditional_edges(
    "interview",
    after_interview_condition,
    {
        "retriever": "retriever",
        "__END__": END  # Use the imported END constant here
    }
)

# graph.add_edge("interview", "retriever") # Ensure no direct edge
graph.add_edge("retriever", "writer")
graph.add_edge("writer", "reviewer")

graph.set_finish_point("reviewer") # This is the overall graph finish
workflow = graph.compile()


# ───────────────────────── CLI loop ─────────────────────────
async def chat_loop():
    initial_graph_input = GraphState() # Initialize with GraphState instance for the first call

    # Initial invocation to get the first AI message (the first question)
    # workflow.ainvoke is expected to return a dict-like representation of the state
    state: Dict[str, Any] = await workflow.ainvoke(initial_graph_input)

    while True:
        for m in state.get("ai_messages", []):
            print("AI:", m.content, "\n")
        state["ai_messages"] = []

        if state.get("strategy"):
            print("\n===== FINAL STRATEGY =====\n")
            print(state["strategy"])
            break

        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        state["user_messages"] = [HumanMessage(content=user)]
        state = await workflow.ainvoke(state)


if __name__ == "__main__":
    asyncio.run(chat_loop())
