# rag_graph.py (condensed)
"""LangGraph RAG with deterministic 5‑question interview."""

from __future__ import annotations

import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

import asyncio, json, os, yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

# ───────────────────────── load prompts ──────────────────────
def load_prompts():
    """Load prompts from YAML configuration file."""
    with open("configurations/prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_prompts()

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
    conversation_history: List[Union[AIMessage, HumanMessage]] = field(default_factory=list)
    ai_messages: List[AIMessage] = field(default_factory=list)
    user_messages: List[HumanMessage] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)  # For predefined
    profile: Optional[Dict[str, Any]] = None
    evidences: Optional[List[Dict]] = None
    strategy: Optional[str] = None
    freeform_question_count: int = 0


# ───────────────────────── interview node ────────────────────
QUESTIONS = [
    "1. What industry does your organisation operate in?",
    "2. In which region or geographical area do you primarily serve?",
    "3. Who are your main customers or target audience?",
    "4. What specific AI use‑cases are you planning to implement?",
    "5. What key risks or organisational values should guide your approach?",
]


def predefined_interview(state: GraphState):
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


# --- New nodes for free-form interview ---

freeform_prompt = ChatPromptTemplate.from_template(prompts["freeform_interview"])

completion_check_prompt = ChatPromptTemplate.from_template(prompts["completion_check"])

summarizer_prompt = ChatPromptTemplate.from_template(prompts["summarizer"])


def freeform_interview(state: GraphState) -> GraphState:
    # If this is the first time we enter freeform, we need to build history.
    if not state.conversation_history:
        history = []
        for i, answer in enumerate(state.answers):
            history.append(AIMessage(content=QUESTIONS[i]))
            history.append(HumanMessage(content=answer))
        state.conversation_history = history
        # Add a transitional message.
        state.conversation_history.append(AIMessage(content="Thanks for that information. Let's talk in some more detail now."))

    # Append user message to history (for subsequent turns)
    if state.user_messages:
        state.conversation_history.extend(state.user_messages)
        state.user_messages = []

    # Generate next question
    history_str = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.conversation_history)
    next_q_prompt = freeform_prompt.format_prompt(history=history_str)
    response = llm.invoke(next_q_prompt.to_messages())
    
    ai_message = AIMessage(content=response.content)
    state.ai_messages = [ai_message]
    state.conversation_history.append(ai_message)
    
    # Increment the question counter
    state.freeform_question_count += 1
    
    return state


def summarize_profile(state: GraphState) -> GraphState:
    history_str = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.conversation_history)
    summarizer_request = summarizer_prompt.format_prompt(history=history_str)
    response = llm.invoke(summarizer_request.to_messages())
    try:
        new_info = json.loads(response.content)
        if state.profile:
            state.profile.update(new_info)
        else: # Should not happen
            state.profile = new_info
    except json.JSONDecodeError:
        print("Error decoding profile JSON from LLM")
        if state.profile is None:
            state.profile = {}  # empty profile
    
    state.ai_messages = [AIMessage(content="Thanks! I've gathered all the necessary information. Now, I'll build a draft strategy for you.")]
    return state


# ───────────────────────── retriever node ───────────────────
search_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts["search_system"]),
    ("human", "{p}"),
])


def retriever(state: GraphState) -> GraphState:
    # Skip if profile not yet collected (first few ticks)
    if state.profile is None:
        return state

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
    ("system", prompts["writer_system"]),
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
    ("system", prompts["reviewer_system"]),
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

def route_interviews(state: GraphState) -> str:
    """Routes to the correct interview stage."""
    # If profile is created, the predefined interview is done.
    if state.profile is not None:
        return "freeform_interview"
    else:
        return "predefined_interview"

def after_predefined_interview_condition(state: GraphState) -> str:
    """Proceed to freeform interview after predefined questions, or end run to await user input."""
    if state.profile is not None:  # Predefined interview complete
        return "freeform_interview"
    else: # Predefined interview ongoing
        return "__END__"

def after_freeform_interview_condition(state: GraphState) -> str:
    """Checks if the freeform interview is complete based on several conditions."""
    # 1. Check if the question limit has been reached
    if state.freeform_question_count >= 5:
        return "summarize_profile"

    # 2. Check if the user wants to stop
    # The most recent user message is in the `user_messages` list before being appended to history
    if state.user_messages:
        last_user_message = state.user_messages[-1].content.lower()
        stop_phrases = ["stop", "that's enough", "i don't want to answer", "quit", "exit"]
        if any(phrase in last_user_message for phrase in stop_phrases):
            return "summarize_profile"

    # 3. Check if the LLM has decided to end the conversation
    if not state.conversation_history:
        return "__END__" # Should not happen in normal flow
    
    history_str = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in state.conversation_history)
    completion_check = completion_check_prompt.format_prompt(history=history_str)
    response = llm.invoke(completion_check.to_messages())
    try:
        is_complete = json.loads(response.content).get("complete", False)
        if is_complete:
            return "summarize_profile"
    except json.JSONDecodeError:
        pass # Not complete, continue interview
    
    return "__END__"


graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("predefined_interview", predefined_interview)
graph.add_node("freeform_interview", freeform_interview)
graph.add_node("summarize_profile", summarize_profile)
graph.add_node("retriever", retriever)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

# Set the entry point
graph.set_conditional_entry_point(
    route_interviews,
    {
        "predefined_interview": "predefined_interview",
        "freeform_interview": "freeform_interview",
    },
)

# Path for predefined questions
graph.add_conditional_edges(
    "predefined_interview",
    after_predefined_interview_condition,
    {
        "freeform_interview": "freeform_interview",
        "__END__": END
    }
)

# Path for freeform interview
graph.add_conditional_edges(
    "freeform_interview",
    after_freeform_interview_condition,
    {
        "summarize_profile": "summarize_profile",
        "__END__": END,
    },
)

# Connect the rest of the graph
graph.add_edge("summarize_profile", "retriever")
graph.add_edge("retriever", "writer")
graph.add_edge("writer", "reviewer")

graph.set_finish_point("reviewer") # This is the overall graph finish
workflow = graph.compile()


# ───────────────────────── CLI loop ─────────────────────────
async def chat_loop():
    print("Starting interview process...")
    # Start with a clear state, but ensure essential lists and counters are initialized
    state = {"user_messages": [], "answers": [], "freeform_question_count": 0} 

    # Initial invocation to get the first AI message (the first question)
    state = await workflow.ainvoke(state)

    while True:
        ai_messages = state.get("ai_messages", [])
        for m in ai_messages:
            # All messages in this list should be AIMessage objects
            print("AI:", m.content, "\n")
        state["ai_messages"] = []

        if state.get("strategy"):
            print("\n===== FINAL STRATEGY =====\n")
            print(state["strategy"])
            break

        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        
        # Pass the user message back into the state for the next invocation
        state["user_messages"] = [HumanMessage(content=user_input)]
        state = await workflow.ainvoke(state)


if __name__ == "__main__":
    asyncio.run(chat_loop())
