print("=" * 10 + " PolicyMatch Starting... " + "=" * 10)
import sys, threading, time, itertools
import pandas as pd
import json, os
import re
import gradio as gr
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb


# ---------------------------------------------------------------------------
# Spinner utility
# ---------------------------------------------------------------------------

def _throbber(label, stop_event):
    spinner = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{next(spinner)}  {label}   ")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(f"\r✓  {label}\n")
    sys.stdout.flush()

def loading(label):
    stop = threading.Event()
    t = threading.Thread(target=_throbber, args=(label, stop), daemon=True)
    t.start()
    return stop


# ---------------------------------------------------------------------------
# JetStream / OpenAI client
# ---------------------------------------------------------------------------

JETSTREAM_URL   = "https://llm.jetstream-cloud.org/api"
JETSTREAM_MODEL = "gpt-oss-120b"

def make_client(api_key):
    return OpenAI(base_url=JETSTREAM_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# ChromaDB helpers (lazy singletons, initialised in main)
# ---------------------------------------------------------------------------

_embed_model = None
_collection  = None
CHROMA_AVAILABLE = False


def _get_embed_model(hf_token=None):
    global _embed_model
    if _embed_model is None and CHROMA_AVAILABLE:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2", token=hf_token or None)
    return _embed_model

def _get_collection(chroma_path=None):
    global _collection
    if chroma_path is None:
        chroma_path = os.path.join(os.getcwd(), "chroma_db")
    if _collection is None and CHROMA_AVAILABLE:
        try:
            cc = chromadb.PersistentClient(path=chroma_path)
            _collection = cc.get_or_create_collection(name="policymatch")
        except Exception as e:
            print(f"[ChromaDB] {e}")
    return _collection

def query_chroma(prompt, hf_token=None):
    model = _get_embed_model(hf_token)
    col   = _get_collection()
    if model is None or col is None or col.count() == 0:
        return None
    embeds  = model.encode([prompt]).tolist()

    results = col.query(query_embeddings=embeds,
                        n_results=12,
                        include=["documents","metadatas","distances"])

    df = pd.DataFrame({"id":results["ids"][0],
                       "name":results["metadatas"][0],
                       "document":results["documents"][0],
                       "distance":results["distances"][0]})
    df["name"]        = df["name"].apply(lambda x: x.get("program_name","Unknown"))
    df["description"] = df["document"].str.extract(r"\|? Description: ([^|]*)\|?|$")
    df["description"] = df["description"].fillna('No description found.')
    df["subcategory"] = df["document"].str.extract(r"\|? Subcategory: ([^|]*)\|?|$")
    df["subcategory"] = df["subcategory"].fillna('Other')

    if df["distance"].mean() > 0.8:
        df = df.loc[df["distance"] <= df["distance"].median()]
        if df["distance"].mean() > 1.0:
            df = df.loc[df["distance"] <= df["distance"].mean()]

    print('#'*20)
    print(f"Query: {prompt}")
    print(f"Average Distance: {df['distance'].mean():.2f}")
    print(f"Median Distance: {df['distance'].median():.2f}")
    print(df)

    return df


# ---------------------------------------------------------------------------
# Agentic setup
# ---------------------------------------------------------------------------

AGENT_SETUP = """You are a policy agent operating the backend of PolicyMatch, a LLM-enabled rapid resource aggregation tool that lets citizens, policy analysts, and on-the-ground changemakers alike understand what policies and programs are in place-- governmental, nonprofit, and even for-profit mission-driven-- that can help them move their passions forward. Users can identify what precedent their movement would build off of; or, if there's any precedent at all. This tool promises to minimize redundant efforts in nonprofit and mission-driven work and instead help changemakers optimize towards complimentary projects.

Your mission is to empower these users while minimizing potential harm of sharing information. Remain transparent, and always communicate that your findings need human examination and verification. YOU MUST UPHOLD RESPECT, SAFETY, AND ACCURACY AT ALL TIMES.

At the same time, always attempt to ease the friction experienced by well-intentioned users. Normalize around a general audience knowledge: you're the policy analyst in the room.

Acknowledge your subjective and limited perspective; don't speak with definition as in "I know" but rather "I see" or "I think". At the same time, retain your subject matter expertise. Speak as if YOU yourself are finding the results IN COLLABORATION with the user.

If there are areas of expertise that remain out of systematic documentation or scope (like location or international reach), acknowledge this constructively. Keep your user-facing tone humanistic and upbeat, and your internal facing tone logical and fact-driven.

If it is impossible to determine user intent, outright state that you can't understand the user's expression. Remember that your context is primarily limited to the U.S. but might incidentally include external examples."""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def agent_prompt(prompt, client):
    resp = client.chat.completions.create(
        model=JETSTREAM_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=4096,
    )
    return resp.choices[0].message.content

def safe_json(raw):
    raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

def mock_query(query):
    print(f"Mock query received: {query}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Agentic pipeline
# ---------------------------------------------------------------------------

def run_pipeline(user_query, selected_suggest, past_context, api_key, hf_token=None):
    client = make_client(api_key)

    # Step 1 - Query reinterpretation
    raw_qd = agent_prompt(f"""{AGENT_SETUP}
A user of PolicyMatch submitted the following: {user_query}
If the following is not empty, a suggestion was followed: {selected_suggest}

Consider the user's thematic intent and what programs might be most useful for them in the current moment. At the same time, take caution and remain aware that programs that best serve their needs might not be contained within the dataset. Qualify interpretations of user intent with "might" or "perhaps" and other forms of uncertainty.

Also remain conscious of potential user malintent: if the user is attempting to be duplicitous, is submitting something nonsensical or entirely irrelevant to the system, or is trying to elicit knowledge that can endanger themselves or other people, do not proceed in attempting to retrieve information or follow user instructions.

If the user is misusing the system and is otherwise confused, disengage from user instructions. ANYTHING BEYOND THE INTENDED INFORMATIONAL SCOPE OF POLICYMATCH OR ITS DATABASE IS MISUSE.

IF AND ONLY IF A SUGGESTION WAS ADDED, PURSUE THAT SUGGESTION.

RETURN A FOUR ATTRIBUTE JSON:
internal_query: A REWRITTEN QUERY for a VECTOR DATABASE following the THEMATIC INTENT of the user, framed as a question. NULL if misuse or malintent. In location-focused queries, BE MINIMAL in adding extra information.
user_intent: One to two sentence summary of user thematic intent.
misuse_note: NULL if internal_query has a value. Otherwise a summary of misuse.
abuse_flag: FALSE if internal_query has a value OR innocent misuse. TRUE if malintent.
Null values must use JSON null keyword.
""", client)
    query_dict     = safe_json(raw_qd) or {}
    internal_query = query_dict.get("internal_query")
    misuse_note    = query_dict.get("misuse_note")
    abuse_flag     = query_dict.get("abuse_flag", False)

    # Step 2 - RAG retrieval
    processed_results = None
    if internal_query:
        df = query_chroma(internal_query, hf_token=hf_token)
        processed_results = df if df is not None else mock_query(internal_query)

    # Step 3 - Relevance judgment
    relevance = {"relevant_flag":False,"relevant_float":0.0,"audit_notes":"No results."}
    if processed_results is not None:
        raw_rel = agent_prompt(f"""{AGENT_SETUP}
Original prompt: {user_query}
Rewritten prompt: {internal_query}
Results: {processed_results.to_string()}

Assess whether, in the full context of the user's query, the internal query, and the final results, whether the user got RELEVANT and PRESCIENT results. Use distance to inform your determination, but DO NOT depend on it.
Return the following as a JSON:
audit_notes: Your relevance findings. Write an assessment of relevance that is at least 2 sentences BUT IS NO LONGER THAN 3 SENTENCES, under ANY CIRCUMSTANCES. Explain thoroughly yet compactly.
relevant_flag: TRUE if material is sufficiently relevant. FALSE otherwise.
relevant_float: A 0.00 to 1.00 float describing how relevant the material is.
""", client)
        relevance = safe_json(raw_rel) or relevance

    # Step 4 - User-facing explainer
    ui_explanation = agent_prompt(f"""{AGENT_SETUP}
Original prompt: {user_query}
Rewritten prompt: {internal_query}
Relevance judgment: {json.dumps(relevance)}
User intent: {query_dict.get('user_intent')}
Results: {processed_results.to_string() if processed_results is not None else 'None'}
Misuse note: {misuse_note}
Abuse flag: {abuse_flag}

Explain your interpretation of user intent in a way that's compact and immediately understandable, irrespective of user experience or pre-existing policy knowledge. Explain either the relevance or lack of relevance of the results the system obtained in relation to the user's prompt. Ensure your response is at least 1 sentence and, at most, is no longer than 3 sentences.

IF YOUR RESPONSE MUST ABSOLUTELY BE LONGER, USE LINE BREAKS.

Your writing will be targeted towards the user and therefore should be written in first person of consideration of that: write as if in direct conversation with the user. Your response will displayed within a UI element.

Given a failure to produce relevant results, suggest ways to rephrase or further specify the request that can help the user find what they need. Acknowledge your subjective and limited perspective; don't speak with definition as in "I know" but rather "I see" or "I think". At the same time, retain your subject matter expertise. Speak as if YOU yourself are finding the results IN COLLABORATION with the user. Provide options and frame in the "you could" "it might be helpful" context.

If the user has malintent or is misuing the system, explain why their request was denied and explain how to better use PolicyMatch. Such responses should only be 2 sentences at most and under 80 characters.

Prefer "my findings" or "what I found" over "the list" and other such overly computational language.
""", client)

    # Step 5 - Theme extraction
    new_themes = agent_prompt(f"""{AGENT_SETUP}
Original prompt: {user_query}
Rewritten prompt: {internal_query}
Explanation: {ui_explanation}
Misuse note: {misuse_note}
User intent: {query_dict.get('user_intent')}

What are themes that seem to be emerging? LIST ONLY THREE as a hanging phrase, MAKE THEM SPECIFIC TO THE USER'S CONTEXT AS POSSIBLE, and express them as a single string value demarcated by commas:
'X, Y, Z'. Do not steer away from this format.

If there is misuse, return a blank string.
""", client).strip()

    # Step 6 - Rolling memory update
    new_past_context = agent_prompt(f"""{AGENT_SETUP}
Past context (empty means new conversation): {past_context}
Recent themes: {new_themes}
Most recent prompt: {user_query}
Misuse note: {misuse_note}
User intent: {query_dict.get('user_intent')}
Recent results: {processed_results.to_string() if processed_results is not None else 'None'}

What policy topics might the user be exploring, or what sort of research are they trying to execute? What is the system providing? Using this history of themes, summarize their exploration in exactly four sentences: no more, no less. Return only those sentences. If there is misuse, return a blank string.
""", client).strip()

    # Step 7 - Follow-up suggestion chips
    raw_sug = agent_prompt(f"""{AGENT_SETUP}
Past context: {past_context}
Recent themes: {new_themes}
Most recent prompt: {user_query}
Misuse note: {misuse_note}
User intent: {query_dict.get('user_intent')}
Recent results: {processed_results.to_string() if processed_results is not None else 'None'}
Last message to user: {ui_explanation}

Suggest THREE and ONLY THREE short but relevant ways that the user can explore further. That is, if the user was looking into workforce development programming, possibly suggest: 'Focus on young adults' THEY MUST BE ACTIONS THAT CAN BE PERFORMED WITHIN THE BOUNDS OF THE POLICYMATCH PLATFORM, SUCH AS THEMATIC SPECIFICATION OR KEYWORD USE.

Each suggestion should be written as a hanging imperative clause without ending punctuation. Return as a JSON of three suggestions that can be loaded as a Pandas Series.

If there is misuse, redirect the user towards ways to APPROPRIATELY use the system. RETURN ONLY A JSON ARRAY OF THREE SUGGESTIONS.
""", client)
    sug_list    = safe_json(raw_sug) or []
    suggestions = list(sug_list)[:3]
    while len(suggestions) < 3:
        suggestions.append("")

    try:
        print(f"AGENT: {new_past_context}")
    except KeyError:
        pass

    return {
        "query_dict":        query_dict,
        "processed_results": processed_results,
        "relevance":         relevance,
        "ui_explanation":    ui_explanation,
        "new_themes":        new_themes,
        "past_context":      new_past_context,
        "suggestions":       suggestions,
    }


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "Local":     ("#ffe0de","#9b4a4b"),
    "State":     ("#e3e8f5","#3a4a6e"),
    "Federal":   ("#e8f5e3","#3a6e4a"),
    "Nonprofit": ("#f5f0e3","#6e5a3a"),
    "For-Profit":("#f0e3f5","#5a3a6e"),
}
ID_TYPE_MAP = {"NY":"Local","PA":"Local","NYC":"Local",
               "MA":"State","CT":"State","NJ":"State","FED":"Federal"}

def type_badge(ptype):
    bg, fg = TYPE_COLORS.get(ptype, ("#e3e2e0","#45474d"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
            f'font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">{ptype}</span>')

def render_card_from_row(row):
    name   = row.get("name","Unknown")
    subcat = row.get("subcategory","")
    desc   = row.get("description","")
    dist   = row.get("distance",None)
    pid    = str(row.get("id",""))
    prefix = pid.split("-")[0] if "-" in pid else ""
    ptype  = ID_TYPE_MAP.get(prefix,"")
    rel_bar = ""
    score = 0
    if dist is not None:
        score = max(0, min(1, 1 - float(dist)))
        pct   = int(score * 100)
        color = "#22c55e" if score > 0.7 else "#f59e0b" if score > 0.3 else "#ef4444"
        rel_bar = (f'<div style="margin-top:8px;">'
                   f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;margin-bottom:2px;">'
                   f'<span>Relevance</span><span>{pct}%</span></div>'
                   f'<div style="background:#f1f5f9;border-radius:99px;height:4px;">'
                   f'<div style="background:{color};width:{pct}%;height:4px;border-radius:99px;"></div></div></div>')
    safe_name = name.replace("'","\\'").replace('"',"&quot;")
    badge = type_badge(ptype) if ptype else "<span></span>"
    return f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px;
box-shadow:0 4px 16px -6px rgba(0,0,0,.06);display:flex;flex-direction:column;gap:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    {badge}
    <button onclick="(function(){{var d=JSON.stringify({{id:'{pid}',name:'{safe_name}',jurisdiction:''}});var el=document.getElementById('bin_signal');if(el){{el.value=d;el.dispatchEvent(new Event('input',{{bubbles:true}}));}}}})();"
    style="background:none;border:none;cursor:pointer;padding:4px;" title="Save to bin">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2">
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
      </svg>
    </button>
  </div>
  <div style="font-size:15px;font-weight:700;color:#0f172a;line-height:1.3;">{name}</div>
  <div style="font-size:10px;font-weight:600;color:#475569;line-height:1.3;">{subcat or "Other"}</div>
  <div style="font-size:13px;color:#475569;line-height:1.5;">{desc or "No description available."}</div>
  <div style="font-size:10px;font-weight:100;font-style:italic;color:#475569;line-height:1.3;">Relevance: {score:.2f}</div>
  {rel_bar}
</div>"""

def render_cards_grid(df):
    if df is None or len(df) == 0:
        return ('<div style="color:#94a3b8;text-align:center;padding:48px;font-size:14px;">'
                'No matching policies or programs found. Try using different keywords.</div>')
    rows  = df.to_dict("records")
    left  = "".join(render_card_from_row(r) for r in rows[::2])
    right = "".join(render_card_from_row(r) for r in rows[1::2])
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">'
            f'<div style="display:flex;flex-direction:column;gap:18px;">{left}</div>'
            f'<div style="display:flex;flex-direction:column;gap:18px;">{right}</div></div>')

def render_bin(items):
    if not items:
        return ('<div style="border:2px dashed #e2e8f0;border-radius:8px;padding:28px;'
                'text-align:center;color:#cbd5e1;margin-top:8px;">'
                '<div style="font-size:22px;margin-bottom:6px;">+</div>'
                '<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;">Add to bin</div></div>')
    cards = "".join(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-bottom:10px;">'
        f'<span style="font-size:9px;font-weight:700;color:#94a3b8;">REF: #{item.get("id","")}</span>'
        f'<div style="font-size:11px;font-weight:700;color:#1e293b;margin:2px 0;">{item.get("name","")}</div>'
        f'<div style="font-size:9px;color:#94a3b8;">{item.get("jurisdiction","")}</div></div>'
        for item in items)
    if len(items) < 3:
        cards += ('<div style="border:2px dashed #e2e8f0;border-radius:8px;padding:20px;'
                  'text-align:center;color:#cbd5e1;">'
                  '<div style="font-size:18px;margin-bottom:4px;">+</div>'
                  '<div style="font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;">Add to bin</div></div>')
    return cards

def results_header_html(n):
    return ('<div style="margin:20px 0 14px;display:flex;align-items:baseline;justify-content:space-between;">'
            '<span style="font-size:22px;font-weight:800;color:#051125;letter-spacing:-.02em;">Matching Policy Resources</span>'
            f'<span style="font-size:12px;color:#94a3b8;font-weight:600;">Showing {n} results</span></div>')

def export_bin_to_csv(bin_items):
    import csv
    if not bin_items:
        return None
    path = "/tmp/policymatch_export.csv"
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","name","jurisdiction"])
        w.writeheader()
        for item in bin_items:
            w.writerow({k: item.get(k,"") for k in ["id","name","jurisdiction"]})
    return path


# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600;700;800&display=swap');
* {
  font-family: 'Open Sans',
  sans-serif !important;
  box-sizing: border-box;
}

.generating {
    border-color: #3a5a9b !important;
    box-shadow: 0 0 0 2px #3a5a9b33 !important;
}

body,
.gradio-container {
  background: #f8fafc !important;
  margin: 0 !important;
  padding: 0 !important;
}

.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
}

footer {
  display: none !important;
}

.primary-btn button {
  background: #051125 !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  transition: opacity .2s !important;
}

.primary-btn button:hover {
  opacity: 0.85 !important;
}

.chip-btn button {
  background: transparent !important;
  border: 1px solid #051125 !important;
  color: #051125 !important;
  border-radius: 99px !important;
  font-size: 12px !important;
  padding: 4px 14px !important;
  transition: all .15s !important;
}

.chip-btn button:hover {
  background: #051125 !important;
  color: #fff !important;
}

.export-btn button {
  background: #0f172a !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  width: 100% !important;
}

::-webkit-scrollbar {
  width: 4px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #e2e8f0;
  border-radius: 10px;
}

#pm-nav {
    background: #eee;
    border-bottom: 3px solid #e2e8f0;
    padding: 0 28px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 60px;
}
#pm-nav-logo { display:flex;
                align-items:center;
                justify-content: center;
                flex:0 0 auto; }
#pm-nav-key  { flex:2;
              display:flex;
              flex-direction:column;
              align-items:flex-end;
              justify-content:center;
              gap:4px;
              padding: 8px 0; }
#pm-nav-key .wrap { gap:2px !important; }
#pm-nav-key label { font-size:10px !important; color:#94a3b8 !important;
                    margin-bottom:0 !important; }
#pm-nav-key input { font-size:12px !important; border-radius:8px !important;
                    border:1px solid #e2e8f0 !important;
                    background:#f8fafc !important; padding:5px 12px !important; }
"""

LOGO_SVG_HTML = '<svg  height="64" version="1.0" viewBox="0 0 850 142" xmlns="http://www.w3.org/2000/svg" zoomAndPan="magnify"><defs><clipPath id="f"><path d="m38 0.15h111v118h-111z"/></clipPath><clipPath id="b"><path d="m0.65 0.15h100v118h-100z"/></clipPath><clipPath id="d"><path d="m117 5.1h31v29h-31z"/></clipPath><clipPath id="k"><rect width="150" height="120"/></clipPath><clipPath id="a"><path d="m202 34h54v66h-54z"/></clipPath><clipPath id="e"><rect width="256" height="101"/></clipPath><clipPath id="h"><path d="m31 16h79v79h-79z"/></clipPath><clipPath id="i"><path d="m71 16c-22 0-40 18-40 40 0 22 18 40 40 40s40-18 40-40c0-22-18-40-40-40z"/></clipPath><clipPath id="j"><path d="m465 23h370v96h-370z"/></clipPath><clipPath id="c"><rect width="371" height="96"/></clipPath><clipPath id="g"><path d="m0.46 24h111v110h-111z"/></clipPath><clipPath id="l"><rect width="836" height="135"/></clipPath></defs><g transform="translate(1 -7.9e-15)"><g clip-path="url(#l)"><g transform="translate(13 14)"><g clip-path="url(#k)"><g clip-path="url(#f)"><path d="m132 65v6.3h-5.5v-6.3zm-5.5 16v-6.3h5.5v6.3zm0 9.2v-6.3h5.5v6.3zm-1.5 2.9h8.4c0.82 0 1.5-0.66 1.5-1.5v-28c0-0.81-0.66-1.5-1.5-1.5h-8.4c-0.81 0-1.5 0.66-1.5 1.5v28c0 0.82 0.66 1.5 1.5 1.5zm-28-28v4.5h-5.9v-4.5zm-5.9 12v-4.5h5.9v4.5zm-1.5 2.9h8.9c0.81 0 1.5-0.66 1.5-1.5v-15c0-0.82-0.66-1.5-1.5-1.5h-8.9c-0.81 0-1.5 0.66-1.5 1.5v15c0 0.81 0.66 1.5 1.5 1.5zm-28-15v6.3h-5.5v-6.3zm-5.5 16v-6.3h5.5v6.3zm0 9.2v-6.3h5.5v6.3zm-1.5 2.9h8.4c0.81 0 1.5-0.66 1.5-1.5v-28c0-0.81-0.66-1.5-1.5-1.5h-8.4c-0.82 0-1.5 0.66-1.5 1.5v28c0 0.82 0.66 1.5 1.5 1.5zm92-41-1.2 3.4c-0.13 0.4-0.46 0.63-0.88 0.63h-100c-0.42 0-0.75-0.23-0.88-0.63l-1.2-3.4 0.0039-0.023-1.2-0.86 1.2 0.85h104l0.023 0.012 1.2-0.86-1.2 0.88zm-6.3 47v-40h-21v38c1.1 0.36 1.9 1.3 2 2.5zm-9.2 17v-5.2c0-0.81-0.66-1.5-1.5-1.5h-3.8v-5.2c0-0.81-0.66-1.5-1.5-1.5h-2.7v-0.67h19v14zm-83-14h19v0.67h-2.7c-0.82 0-1.5 0.66-1.5 1.5v5.2h-3.8c-0.82 0-1.5 0.66-1.5 1.5v5.2h-9.2zm29-5.6h-5.1v-38h5.1zm31 0.15v-38h-29v38c1.2 0.39 2.1 1.5 2.1 2.8v3.3h3.2v-16h-0.92c-0.82 0-1.5-0.66-1.5-1.5 0-0.81 0.66-1.5 1.5-1.5h20c0.81 0 1.5 0.66 1.5 1.5 0 0.82-0.66 1.5-1.5 1.5h-0.92v16h3.2v-3.3c0-1.3 0.86-2.4 2.1-2.8zm8.1-0.15h-5.1v-38h5.1zm-7.2 6.2v-3.3c0-0.012 0.02-0.027 0.027-0.027h9.2c0.012 0 0.027 0.016 0.027 0.027v3.3zm-14 0v-16h4.6v16zm-3-16v16h-4.6v-16zm-23 13c0-0.012 0.016-0.027 0.027-0.027h9.2s0.027 0.016 0.027 0.027v3.3h-9.2zm53 10v-3.8h-57v3.8zm5.2 6.7v-3.8h-68v3.8zm-59-58h-21v40h19c0.14-1.2 0.95-2.1 2-2.5zm-5.6-14c0-0.33 0.28-0.6 0.6-0.6h61c0.33 0 0.61 0.27 0.61 0.6v3.7h-62zm25-25c-0.87 1.2-1.7 2.7-2.4 4.5-1.8 4.5-2.8 10-3 17h-12c0.56-10 7.8-19 17-21zm5.9-0.76c-3.8 0-8.1 9-8.3 22h17c-0.26-13-4.5-22-8.3-22zm23 22h-12c-0.13-6.3-1.2-12-3-17-0.7-1.8-1.5-3.3-2.4-4.5 9.6 2.5 17 11 17 21zm-22-38h8l-1.5 1.4c-0.28 0.28-0.44 0.66-0.44 1.1 0 0.39 0.16 0.77 0.44 1.1l1.5 1.4h-8zm53 46c-0.57-0.79-1.4-1.2-2.4-1.2h-18v-3.7c0-2-1.6-3.5-3.6-3.5h-4c-0.65-13-11-24-25-25v-4.9h12c0.6 0 1.1-0.36 1.4-0.92 0.23-0.55 0.094-1.2-0.34-1.6l-3-2.9 3-2.9c0.43-0.42 0.56-1.1 0.34-1.6-0.23-0.55-0.77-0.91-1.4-0.91h-13c-0.81 0-1.5 0.66-1.5 1.5v14c-13 0.75-24 12-25 25h-4c-2 0-3.6 1.6-3.6 3.6v3.7h-18c-0.97 0-1.9 0.45-2.4 1.2-0.57 0.79-0.71 1.8-0.41 2.7l1.2 3.4c0.54 1.6 2 2.6 3.7 2.6h1.3v59c0 0.81 0.66 1.5 1.5 1.5h95c0.82 0 1.5-0.66 1.5-1.5v-59h1.3c1.7 0 3.1-1 3.7-2.6l1.2-3.4c0.31-0.92 0.16-1.9-0.4-2.7z" fill="#12100b" fill-rule="evenodd"/></g><g clip-path="url(#b)"><path d="m59 0.16c19 0.11 35 13 40 32 4.6 18-4.4 38-20 46-9 5-19 6.3-29 4.5-3.1-0.56-6.1-1.5-9-2.8-0.76-0.34-1.1-0.19-1.6 0.46-7.7 11-16 22-23 33-0.87 1.2-1.7 2.5-2.6 3.7-1.5 2-3.6 2.4-5.6 1-1.9-1.3-3.7-2.5-5.5-3.9-2.3-1.7-2.7-3.9-1-6.2 4.8-6.5 9.6-13 14-19 4.1-5.5 8.1-11 12-17 0.52-0.7 0.5-1.1-0.1-1.7-13-14-16-37-3.6-53 6.9-9.3 16-15 28-16 1.3-0.19 2.6-0.3 3.8-0.38 1.1-0.062 2.1-0.012 3.2-0.012zm37 41c-0.1-21-17-37-37-37-22-0.035-38 17-38 36-0.17 19 15 38 38 38 20-0.15 37-16 37-37zm-86 72c0.34-0.12 0.46-0.39 0.63-0.63 8.1-11 16-23 24-34 0.49-0.68 1.3-1.4 1.3-2.1-0.051-0.7-1.2-1.1-1.8-1.6-1.9-1.5-1.9-1.4-3.4 0.52-5.7 7.8-11 16-17 23-2.7 3.6-5.3 7.3-8 11-0.43 0.57-0.39 0.88 0.22 1.3 1.1 0.74 2.2 1.6 3.3 2.4 0.18 0.12 0.37 0.23 0.53 0.34z" fill="#040404" fill-rule="evenodd"/></g><path d="m58 75c-19-0.12-34-16-34-35 0.14-19 15-34 34-34 19-0.027 34 15 34 35-0.11 19-16 34-34 34zm0.23-3.8c16-0.078 30-14 30-31-0.062-17-13-30-30-30-17 0.047-30 13-30 30 0.082 17 14 31 30 31z" fill="#040404" fill-rule="evenodd"/><path d="m57 13c4.1-0.012 7.4 0.62 11 1.9 1.2 0.48 1.7 1.4 1.3 2.4-0.3 0.95-1.4 1.4-2.5 1.2-0.32-0.082-0.62-0.21-0.94-0.32-5.5-1.8-11-1.8-16 0.17-4.4 1.6-8 4.2-11 8.1-0.84 1.2-1.9 1.5-2.9 0.86-1-0.67-1.2-1.9-0.31-3.1 3.6-5 8.4-8.2 14-10 2.6-0.8 5.3-1.2 7.4-1.2z" fill="#040404" fill-rule="evenodd"/><path d="m83 40c-0.055 4.4-1 8.7-3.1 13-0.16 0.29-0.32 0.59-0.52 0.84-0.72 0.92-1.8 1.1-2.6 0.59-0.87-0.56-1.2-1.6-0.57-2.6 1.2-2 2-4.2 2.5-6.4 0.66-2.9 0.73-5.8 0.3-8.7-0.19-1.3 0.38-2.3 1.4-2.4 1.1-0.2 2.1 0.55 2.3 1.9 0.2 1.4 0.37 2.8 0.32 4.3z" fill="#040404" fill-rule="evenodd"/><path d="m76 28c-1.2 0.012-2-0.79-2-1.9 0.0078-1 0.93-1.9 2-1.9 1.1 0.0039 1.9 0.83 2 1.8 0.031 1.1-0.84 1.9-1.9 1.9z" fill="#040404" fill-rule="evenodd"/><path d="m72 60c0.023 1-0.82 1.9-1.8 2-1.1 0.055-2-0.77-2.1-1.8-0.059-1 0.86-2 1.9-2 1.1-0.027 2 0.82 2 1.9z" fill="#040404" fill-rule="evenodd"/><g clip-path="url(#d)"><path d="m123 5.2-2.3 4.2-4.2 2.4v1.3l4.2 2.4 2.3 4.2h1.3l2.3-4.2 4.2-2.4v-1.3l-4.2-2.4-2.3-4.2zm0.65 3.3 1.4 2.5 2.5 1.4-2.5 1.4-1.4 2.5-1.4-2.5-2.5-1.4 2.5-1.4zm14 2.5-3.3 6.2-6.1 3.3v1.3l6.1 3.3 3.3 6.2h1.3l3.3-6.2 6.1-3.3v-1.3l-6.1-3.3-3.3-6.2zm0.66 3.3 2.3 4.4 4.3 2.4-4.3 2.4-2.3 4.4-2.3-4.4-4.4-2.4 4.4-2.4zm-13 11-1.4 2.3-2.2 1.4v1.2l2.2 1.4 1.4 2.3h1.2l1.4-2.3 2.2-1.4v-1.2l-2.2-1.4-1.4-2.3zm0.61 3.1 0.48 0.77 0.75 0.48-0.75 0.48-0.48 0.77-0.47-0.77-0.76-0.48 0.76-0.48z"/></g></g></g><g transform="translate(211 20)"><g clip-path="url(#e)"><g><g transform="translate(.76 76)"><path d="m30 0.55c-0.42 0-1.1-0.031-2.1-0.094s-2-0.12-3-0.19c-1-0.055-2-0.11-3-0.17s-1.7-0.094-2.1-0.094h-5.4c-0.43 0-1.1 0.031-2.1 0.094s-2 0.12-3 0.17c-1 0.062-2 0.12-3 0.19s-1.7 0.094-2.1 0.094l-0.19-0.36 0.28-3.5 0.45-0.36c0.73 0 1.6-0.047 2.5-0.14s1.7-0.32 2.3-0.69c0.48-0.36 0.93-0.94 1.4-1.7 0.43-0.79 0.73-2 0.91-3.5 0-0.41 0.016-0.91 0.047-1.5 0.031-0.58 0.062-1.3 0.094-2.1s0.047-1.9 0.047-3.1v-4.5-20-4.1c0-1.5-0.062-3.1-0.19-4.6-0.12-1.6-0.42-2.7-0.88-3.5-0.45-0.76-0.91-1.3-1.4-1.7-0.54-0.49-1.1-0.76-1.7-0.81-0.57-0.062-1.6-0.094-3.1-0.094l-0.45-0.38-0.28-3.4 0.19-0.38c0.43 0 0.98 0.031 1.7 0.094 0.7 0.062 1.4 0.12 2.2 0.19 0.76 0.062 1.5 0.12 2.3 0.19 0.79 0.055 1.5 0.078 2.2 0.078h6.9c0.66 0 1.4-0.016 2.3-0.047 0.85-0.031 1.7-0.047 2.6-0.047s1.7-0.0078 2.4-0.031c0.73-0.031 1.3-0.047 1.8-0.047 4.1 0 7.5 0.41 10 1.2 2.8 0.81 5.1 1.9 6.9 3.4 1.8 1.4 3 3.2 3.9 5.2 0.82 2 1.2 4.3 1.2 6.7 0 4.1-0.73 7.4-2.2 10-1.4 2.6-3.3 4.7-5.6 6.2-2.3 1.5-4.8 2.6-7.6 3.1-2.8 0.57-5.6 0.86-8.2 0.86h-5.1v3.8c0 1.3 0.0078 2.6 0.031 3.9 0.031 1.2 0.062 2.3 0.094 3.3s0.078 1.7 0.14 2.2c0.12 1.8 0.43 3 0.91 3.7 0.49 0.69 0.91 1.2 1.3 1.5 0.55 0.48 1.1 0.75 1.7 0.81 0.57 0.062 1.6 0.094 3.2 0.094l0.36 0.36 0.28 3.5zm-3.2-30c5.1 0 8.7-1.1 11-3.5 2.2-2.3 3.3-5.5 3.3-9.5 0-3.3-1.3-5.7-3.8-7.3-2.5-1.6-5.9-2.4-10-2.4h-4.3c-0.19 2.7-0.31 4.7-0.38 6-0.055 1.3-0.078 2.5-0.078 3.5v9.9c0 0.79 0.023 1.4 0.078 1.9 0.062 0.45 0.21 0.78 0.45 0.98 0.25 0.21 0.6 0.34 1 0.41 0.46 0.062 1.1 0.094 1.9 0.094z"/></g><g transform="translate(52 76)"><path d="m26-43c3.6 0 6.7 0.62 9.5 1.9 2.8 1.2 5.1 2.9 7 5 1.9 2.1 3.4 4.5 4.4 7.2 1 2.7 1.5 5.5 1.5 8.4 0 2.8-0.46 5.5-1.4 8.1-0.91 2.6-2.3 4.9-4.2 6.9s-4.2 3.6-7 4.8-6 1.8-9.7 1.8c-3.8 0-7.1-0.64-10-1.9s-5.2-2.9-7-5c-1.8-2.1-3.2-4.4-4.1-7-0.91-2.6-1.4-5.3-1.4-8.1 0-3 0.52-5.8 1.5-8.4 1-2.7 2.5-5 4.4-7 1.9-2 4.2-3.6 7-4.8 2.8-1.2 5.9-1.8 9.4-1.8zm0.078 38c1.9 0 3.5-0.44 5-1.3 1.4-0.88 2.6-2 3.6-3.5 0.97-1.4 1.7-3.1 2.1-4.9 0.46-1.8 0.69-3.7 0.69-5.7 0-1.9-0.26-3.8-0.78-5.7-0.51-1.9-1.3-3.6-2.3-5.1-1-1.5-2.2-2.8-3.7-3.7-1.4-0.97-3.1-1.5-5-1.5-1.9 0-3.6 0.45-5 1.4-1.4 0.91-2.6 2.1-3.5 3.5-0.91 1.4-1.6 3.1-2 4.9-0.45 1.8-0.67 3.7-0.67 5.5 0 1.9 0.25 3.9 0.77 5.7 0.52 1.9 1.3 3.6 2.3 5.1 1 1.5 2.2 2.8 3.7 3.7 1.5 0.94 3.1 1.4 4.9 1.4z"/></g><g transform="translate(103 76)"><path d="m25 0.36c-0.42 0-1.1-0.031-1.9-0.094-0.84-0.055-1.8-0.094-2.7-0.12s-1.9-0.062-2.7-0.094c-0.84-0.031-1.5-0.047-1.9-0.047h-4.9c-0.43 0-1.1 0.016-2 0.047s-1.8 0.062-2.7 0.094-1.9 0.07-2.7 0.12c-0.88 0.062-1.5 0.094-2 0.094l-0.078-0.36 0.17-3.1 0.28-0.36c0.91 0 1.8-0.039 2.8-0.12 0.97-0.094 1.7-0.38 2.2-0.88 0.43-0.41 0.73-0.96 0.91-1.6 0.19-0.66 0.34-1.6 0.45-2.8 0.062-0.43 0.11-0.85 0.14-1.3 0.031-0.43 0.062-0.88 0.094-1.4 0.031-0.49 0.047-1.1 0.047-1.7v-2.5-35-2.8c0-0.7-0.016-1.3-0.047-1.8-0.031-0.49-0.062-0.93-0.094-1.3-0.031-0.39-0.078-0.86-0.14-1.4-0.12-0.6-0.3-1.2-0.55-1.7-0.24-0.55-0.45-0.91-0.62-1.1-0.61-0.79-1.4-1.2-2.3-1.4-0.91-0.12-1.8-0.19-2.8-0.19l-0.36-0.36v-3.1l0.36-0.27h4.6c1.4 0 2.9-0.016 4.4-0.047 1.6-0.031 3-0.13 4.2-0.31l3.2-0.45 0.55 1.7c-0.12 0.24-0.23 0.62-0.33 1.1-0.086 0.51-0.14 1.1-0.17 1.6-0.031 0.57-0.062 1.2-0.094 1.7-0.031 0.57-0.047 1-0.047 1.4 0 0.48-0.016 0.92-0.047 1.3-0.031 0.4-0.047 0.85-0.047 1.4 0 0.51-0.016 1.2-0.047 2-0.023 0.78-0.031 1.8-0.031 2.9v35c0 1.2 0.0078 2.2 0.031 3 0.031 0.76 0.062 1.4 0.094 2 0.031 0.54 0.062 1 0.094 1.5 0.031 0.45 0.078 0.95 0.14 1.5 0.11 1.3 0.25 2.2 0.41 2.9 0.16 0.64 0.47 1.2 0.95 1.6 0.48 0.49 1.2 0.78 2 0.88 0.88 0.086 1.8 0.12 2.7 0.12l0.45 0.45 0.19 3z"/></g><g transform="translate(130 76)"><path d="m26 0.36c-0.42 0-1.1-0.031-1.9-0.094-0.84-0.055-1.8-0.094-2.7-0.12s-1.9-0.062-2.7-0.094c-0.84-0.031-1.5-0.047-1.9-0.047h-4.9c-0.43 0-1.1 0.016-2 0.047s-1.8 0.062-2.7 0.094-1.9 0.07-2.7 0.12c-0.88 0.062-1.5 0.094-2 0.094l-0.078-0.36 0.17-3.1 0.28-0.36c0.91 0 1.8-0.039 2.8-0.12 0.97-0.094 1.7-0.38 2.2-0.88 0.43-0.41 0.73-0.96 0.91-1.6 0.19-0.66 0.34-1.6 0.45-2.8 0.062-0.43 0.11-0.85 0.14-1.3 0.031-0.43 0.062-0.88 0.094-1.4 0.031-0.49 0.047-1.1 0.047-1.7v-2.5-10-2.8c0-0.7-0.016-1.3-0.047-1.8-0.031-0.49-0.062-0.93-0.094-1.3-0.031-0.39-0.078-0.86-0.14-1.4-0.12-0.61-0.3-1.2-0.55-1.7-0.24-0.54-0.45-0.91-0.62-1.1-0.61-0.78-1.4-1.2-2.3-1.3-0.91-0.12-1.8-0.19-2.8-0.19l-0.38-0.38v-3.1l0.38-0.28h4.6c1.4 0 2.9-0.0078 4.4-0.031 1.6-0.031 3-0.14 4.2-0.33l3.2-0.45 0.55 1.7c-0.12 0.24-0.23 0.62-0.33 1.1-0.086 0.51-0.14 1.1-0.17 1.6-0.031 0.57-0.062 1.1-0.094 1.7-0.031 0.57-0.047 1-0.047 1.4 0 0.49-0.016 0.93-0.047 1.3-0.023 0.39-0.031 0.84-0.031 1.4 0 0.51-0.016 1.2-0.047 2-0.031 0.78-0.047 1.7-0.047 2.9v10c0 1.2 0.016 2.2 0.047 3 0.031 0.76 0.055 1.4 0.078 2 0.031 0.54 0.062 1 0.094 1.5 0.031 0.45 0.078 0.95 0.14 1.5 0.12 1.3 0.26 2.2 0.41 2.9 0.16 0.64 0.47 1.2 0.95 1.6 0.49 0.49 1.2 0.78 2 0.88 0.88 0.086 1.8 0.12 2.7 0.12l0.45 0.45 0.19 3zm-18-56c0-0.72 0.15-1.4 0.45-2.2 0.3-0.73 0.72-1.4 1.3-2 0.55-0.57 1.2-1 1.9-1.4 0.7-0.33 1.4-0.5 2.2-0.5 1.5 0 2.9 0.62 4.1 1.9 1.2 1.2 1.9 2.6 1.9 4.1 0 0.79-0.17 1.5-0.5 2.2-0.34 0.7-0.79 1.3-1.4 1.9-0.57 0.54-1.2 0.96-2 1.3-0.73 0.3-1.5 0.45-2.2 0.45-1.6 0-2.9-0.57-4.1-1.7-1.1-1.1-1.7-2.5-1.7-4.1z"/></g><g transform="translate(158 76)"><path d="m38-29-0.36-0.27c0-1.4-0.41-2.5-1.2-3.4-0.81-0.88-1.8-1.6-3-2-1.1-0.49-2.4-0.82-3.7-1-1.3-0.19-2.4-0.28-3.4-0.28-2.2 0-4.1 0.43-5.7 1.3-1.6 0.84-3 2-4 3.4-1 1.4-1.8 3-2.3 4.8-0.51 1.8-0.77 3.7-0.77 5.6 0 2.1 0.29 4 0.86 5.9 0.57 1.9 1.4 3.5 2.6 5 1.2 1.4 2.6 2.6 4.4 3.5 1.8 0.88 3.9 1.3 6.3 1.3 2.1 0 4.2-0.21 6.5-0.64 2.3-0.43 4.4-1.1 6.2-2.1l1.5 2.2-1.4 2.5c-0.67 0.49-1.5 1-2.6 1.5-1.1 0.51-2.2 1-3.6 1.5-1.3 0.45-2.8 0.82-4.3 1.1-1.5 0.3-3.1 0.45-4.6 0.45-3 0-5.8-0.53-8.5-1.6-2.7-1.1-5-2.5-7.1-4.5-2-1.9-3.6-4.2-4.9-6.9-1.2-2.7-1.8-5.7-1.8-8.9 0-3.3 0.6-6.4 1.8-9.1 1.2-2.7 2.9-5.1 5-7 2.1-1.9 4.6-3.4 7.4-4.5 2.8-1.1 5.8-1.6 9-1.6 1.4 0 2.9 0.11 4.4 0.33 1.5 0.21 2.9 0.46 4.3 0.77 1.4 0.3 2.6 0.65 3.7 1 1.1 0.4 2 0.77 2.6 1.1l0.28 0.83-1 9.7z"/></g></g><g clip-path="url(#a)"><g><g transform="translate(203 76)"><path d="m27 11c-0.73 1.8-1.7 3.5-2.9 4.9-1.2 1.5-2.5 2.8-4 3.9-1.5 1.1-3.1 2-4.8 2.7-1.7 0.7-3.4 1.2-5.2 1.4-0.49 0.12-1.1 0.19-1.7 0.19h-2.8l-0.62-0.19-0.28-4.6 0.36-0.45h1.3c0.72 0 1.5-0.078 2.3-0.23 0.85-0.15 1.7-0.43 2.5-0.86 1.8-0.84 3.4-2.1 4.8-3.6 1.5-1.6 2.5-3.1 3-4.5l2.5-6.8-7.3-17c-0.3-0.66-0.79-1.8-1.5-3.5-0.67-1.7-1.4-3.6-2.3-5.6-0.84-2-1.7-4-2.7-6-0.94-2-1.8-3.7-2.6-5-0.61-0.97-1.3-1.7-2-2.3-0.76-0.54-1.8-0.88-3-1l-0.53-0.45v-3.1l0.36-0.36c1 0.12 2.1 0.25 3.3 0.38 1.1 0.12 2.2 0.17 3.2 0.17h7.2c0.91 0 2-0.055 3.3-0.17 1.3-0.12 2.5-0.25 3.6-0.38l0.36 0.36v2.9l-0.36 0.38c-1.2 0.12-2.1 0.46-3 1-0.81 0.57-1.2 1.4-1.2 2.5 0 0.73 0.25 1.9 0.77 3.6 0.51 1.7 1.1 3.4 1.8 5.3 0.7 1.9 1.4 3.6 2 5.2 0.63 1.6 1 2.7 1.2 3.2l3.9 11 4.5-11c0.41-0.97 0.93-2.2 1.5-3.7 0.61-1.5 1.2-3 1.8-4.7 0.57-1.6 1.1-3.2 1.5-4.8 0.43-1.6 0.64-3 0.64-4.2 0-1.3-0.45-2.2-1.4-2.7-0.91-0.46-1.9-0.74-2.9-0.86l-0.36-0.38v-2.9l0.36-0.36c1.2 0.24 2.2 0.39 3.3 0.45 1 0.062 2 0.094 2.9 0.094h7.4c1 0 2-0.055 3-0.17 1-0.12 2.1-0.25 3.2-0.38l0.36 0.36v3.1l-0.55 0.45c-1 0.12-1.9 0.43-2.7 0.92-0.75 0.48-1.5 1.3-2.1 2.3-0.91 1.5-1.9 3.3-3 5.6-1.1 2.3-2.2 4.5-3.3 6.8-1.1 2.3-2 4.4-2.9 6.3-0.84 1.9-1.4 3.2-1.6 3.9z"/></g></g></g></g></g><g clip-path="url(#h)"><g clip-path="url(#i)"><path transform="matrix(.75 0 0 .75 31 16)" d="m53 2e-3c-29 0-53 24-53 53s24 53 53 53c29 0 53-24 53-53s-24-53-53-53z" fill="none" stroke="#000" stroke-width="20"/></g></g><g clip-path="url(#j)"><g transform="translate(465 23)"><g clip-path="url(#c)"><g><g transform="translate(.76 76)"><path d="m57 0v-38h-13v-13h13v-13h13v64zm-51 0v-64h13v13h13v13h-13v38zm25-25v-13h13v13z"/></g><g transform="translate(77 76)"><path d="m6.4 0v-51h13v25h38v-25h13v51h-13v-13h-38v13zm13-51v-13h38v13z"/></g><g transform="translate(147 76)"><path d="m32 0v-51h-25v-13h64v13h-25v51z"/></g><g transform="translate(217 76)"><path d="m19-51v38h-13v-38zm0 51v-13h51v13zm0-51v-13h51v13z"/></g><g transform="translate(293 76)"><path d="m6.4 0v-64h13v25h38v-25h13v64h-13v-25h-38v25z"/></g></g></g></g></g><g clip-path="url(#g)"><path transform="matrix(.44 -.61 .61 .44 15 123)" d="m-2.9e-4 7.5 63 0.0018" fill="none" stroke="#000" stroke-width="15"/></g></g></g></svg>'


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="PolicyMatch | Institutional Clarity") as demo:

        past_context_state     = gr.State("")
        selected_suggest_state = gr.State("")
        collection_bin_state   = gr.State([])
        current_results_state  = gr.State(None)

        with gr.Row(elem_id="pm-nav", equal_height=True):
            with gr.Column(scale=1, min_width=0, elem_id="pm-nav-logo"):
                gr.HTML(LOGO_SVG_HTML)
            with gr.Column(scale=1, min_width=260, elem_id="pm-nav-key"):
                api_key_input = gr.Textbox(
                    label="JetStream API Key",
                    placeholder="Paste JetStream2 key here.",
                    type="password",
                    value=os.environ.get("IU_KEY", ""),
                    container=True,
                )
            with gr.Column(scale=1, min_width=260, elem_id="pm-nav-key"):
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token (Optional)",
                    placeholder="Paste Hugging Face token here.",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    container=True,
                )

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=260):

                gr.HTML("""
                <div style="padding:12px 0 4px;">
                  <div style="font-size:15px;font-weight:800;color:#0f172a;">
                    Agent Dialogue</div>
                  <div style="font-size:10px;color:#94a3b8;font-weight:600;
                       text-transform:uppercase;letter-spacing:.08em;margin-top:2px;">
                    MINIMUM VIABLE PRODUCT USING JETSTREAM2</div>
                </div>""")

                with gr.Group():
                    chatbot = gr.Chatbot(
                        value=[
                            {"role": "assistant",
                             "content": "Hi! I'm your personal policy analyst! Type your interest in the chat input below. I'll try and pull up something relevant from the database."}
                        ],
                        label="",
                        height=400,
                        show_label=False,
                    )
                    with gr.Row():
                        chip1 = gr.Button("Policy around gun violence",
                                          size="sm", elem_classes=["chip-btn"])
                        chip2 = gr.Button("After-school programming",
                                          size="sm", elem_classes=["chip-btn"])
                    with gr.Row():
                        chip3 = gr.Button("Focus on housing policy",
                                          size="sm", elem_classes=["chip-btn"])
                    with gr.Row():
                        search_input = gr.Textbox(
                            placeholder="Enter prompt here...",
                            show_label=False, lines=1, scale=8, container=False,
                        )
                        search_btn = gr.Button("Search", scale=2,
                                               elem_classes=["primary-btn"])
                    status_msg = gr.HTML("")

                gr.HTML("""
                <div style="margin-top:8px;
                padding-top:8px;
                border-top:1px solid #e2e8f0;
                     display:flex;flex-direction:column;gap:4px;">
                  <div style="display:flex;align-items:center;gap:8px;padding:2px 0;
                       cursor:pointer;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                         stroke="#94a3b8" stroke-width="2">
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                      <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span style="font-size:12px;color:#94a3b8;">Support</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:8px;padding:2px 0;
                       cursor:pointer;">
                    <svg width="14" height="14" viewBox="-2 -4 24 24" fill="none"
                         stroke="#94a3b8" stroke-width="2">
                      <path d='M3.636 7.208L10 13.572l6.364-6.364a3 3 0 1 0-4.243-4.243L10 5.086l-2.121-2.12a3 3 0 0 0-4.243 4.242zM9.293 1.55l.707.707.707-.707a5 5 0 1 1 7.071 7.071l-7.07 7.071a1 1 0 0 1-1.415 0l-7.071-7.07a5 5 0 1 1 7.07-7.071z'/>
                    </svg>
                    <span style="font-size:12px;color:#94a3b8;">CMU Heinz College</span>
                  </div> """)

            with gr.Column(scale=2):
                results_header = gr.HTML(results_header_html(0))
                results_html   = gr.HTML(
                    '<div style="color:#94a3b8;text-align:center;padding:48px;'
                    'font-size:14px;">Submit a query above to explore policies.</div>'
                )
                bin_signal = gr.Textbox(visible=False, elem_id="bin_signal")

        SEARCH_OUTPUTS = [
            chatbot, current_results_state, results_html, results_header,
            chip1, chip2, chip3,
            past_context_state, selected_suggest_state,
            search_input, status_msg,
        ]

        def run_search(query, history, api_key, hf_token, past_ctx, sel_suggest):
            if not query.strip():
                yield (history, None, render_cards_grid(None), results_header_html(0),
                       gr.update(), gr.update(), gr.update(), past_ctx, "",
                       gr.update(value=""), gr.update(value=""))
                return

            thinking = list(history) + [
                {"role": "user",      "content": query},
                {"role": "assistant", "content": "Searching the policy database..."},
            ]
            yield (thinking, None, render_cards_grid(None), results_header_html(0),
                   gr.update(value="Rewriting query..."),
                   gr.update(value="Running RAG retrieval..."),
                   gr.update(value="Assessing relevance..."),
                   past_ctx, "", gr.update(value=""), gr.update(value="Searching..."))

            hf_token = hf_token.strip() if hf_token else None

            if api_key and api_key.strip():
                try:
                    result = run_pipeline(query, sel_suggest, past_ctx, api_key.strip(),
                                          hf_token=hf_token)
                except Exception as e:
                    err_hist = list(history) + [
                        {"role": "user",      "content": query},
                        {"role": "assistant", "content": f"Pipeline error: {e}"},
                    ]
                    yield (err_hist, None, render_cards_grid(None), results_header_html(0),
                           gr.update(value="Retry"), gr.update(value=""),
                           gr.update(value=""), past_ctx, "",
                           gr.update(value=""), gr.update(value=""))
                    return
            else:
                df  = mock_query(query)
                sug = [
                    "Specify a geographic area",
                    "Filter by policy type",
                    "Narrow by decade",
                ]
                result = {
                    "processed_results": df,
                    "ui_explanation": "Please enter a JetStream2 API key.",
                    "past_context": past_ctx,
                    "suggestions":  sug,
                }

            df       = result.get("processed_results")
            ui_expl  = result.get("ui_explanation", "")
            new_past = result.get("past_context", past_ctx)
            sug      = result.get("suggestions", ["", "", ""])
            n        = len(df) if df is not None else 0

            final_hist = list(history) + [
                {"role": "user",      "content": query},
                {"role": "assistant", "content": ui_expl},
            ]
            yield (final_hist, df, render_cards_grid(df), results_header_html(n),
                   gr.update(value=sug[0] if len(sug) > 0 else ""),
                   gr.update(value=sug[1] if len(sug) > 1 else ""),
                   gr.update(value=sug[2] if len(sug) > 2 else ""),
                   new_past, "", gr.update(value=""), gr.update(value=""))

        SEARCH_INPUTS = [
            search_input, chatbot, api_key_input, hf_token_input,
            past_context_state, selected_suggest_state,
        ]

        search_btn.click(run_search,    inputs=SEARCH_INPUTS, outputs=SEARCH_OUTPUTS)
        search_input.submit(run_search, inputs=SEARCH_INPUTS, outputs=SEARCH_OUTPUTS)

        def chip_click(chip_text, history, api_key, hf_token, past_ctx, *_):
            yield from run_search(chip_text, history, api_key, hf_token, past_ctx, chip_text)

        CHIP_COMMON = [chatbot, api_key_input, hf_token_input,
                       past_context_state, selected_suggest_state]
        chip1.click(chip_click, inputs=[chip1] + CHIP_COMMON, outputs=SEARCH_OUTPUTS)
        chip2.click(chip_click, inputs=[chip2] + CHIP_COMMON, outputs=SEARCH_OUTPUTS)
        chip3.click(chip_click, inputs=[chip3] + CHIP_COMMON, outputs=SEARCH_OUTPUTS)

        def add_to_bin_fn(signal_val, bin_items):
            if not signal_val:
                return bin_items
            try:
                data = json.loads(signal_val)
                ids  = [x["id"] for x in bin_items]
                if data["id"] in ids:
                    bin_items = [x for x in bin_items if x["id"] != data["id"]]
                elif len(bin_items) < 6:
                    bin_items = bin_items + [data]
            except Exception:
                pass
            return bin_items

        bin_signal.change(add_to_bin_fn,
                          inputs=[bin_signal, collection_bin_state],
                          outputs=[collection_bin_state])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global CHROMA_AVAILABLE

    _s = loading("Loading dependencies...")
    _s.set(); time.sleep(0.12)

    chroma_path = os.path.join(os.getcwd(), "chroma_db")
    _s = loading("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(name="policymatch")
    _s.set(); time.sleep(0.12)

    n_docs = collection.count()
    print(f"✓  {n_docs} documents in collection")
    CHROMA_AVAILABLE = n_docs > 0

    demo = build_demo()

    _s = loading("PolicyMatch (Ctrl+C to Stop) |")
    try:
        print("\n======= PolicyMatch is now running. Press Ctrl+C twice to stop at any time. =======")
        demo.launch(share=False, debug=True, css=CUSTOM_CSS, theme=gr.themes.Soft())
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down PolicyMatch...")
        demo.close()
        print("========= Shutdown complete! PolicyMatch is now closed. =========")


if __name__ == "__main__":
    main()
