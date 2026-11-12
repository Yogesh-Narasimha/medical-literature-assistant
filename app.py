"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st

from src.agent.graph import run_agent

st.set_page_config(page_title="Medical Literature Assistant", page_icon="💊")
st.title("Medical Literature Assistant")
st.caption("Answers come only from FDA drug labels and PubMed abstracts, with citations. "
           "Learning project, not medical advice.")

examples = ["Can I take ibuprofen with warfarin?", "What are the side effects of metformin?",
            "What is the boxed warning for fluoxetine?"]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex

question = st.text_input("Ask a medical question", key="q")

if question:
    with st.spinner("Agent is thinking and searching..."):
        r = run_agent(question)
    st.subheader("Answer")
    st.write(r["answer"])
    if r["invalid"]:
        st.warning(f"The model cited non-existent sources {r['invalid']}.")
    elif r["answered"] and not r["cited"]:
        st.warning("This answer has no citations, so treat it as unverified.")
    if r["uncited"]:
        st.warning("Statements below have no source and should be treated as unverified:\n\n"
                   + "\n".join(f"- {s}" for s in r["uncited"]))
    if r["cited"]:
        st.subheader("Sources")
        for n in r["cited"]:
            s = r["sources"][n - 1]
            with st.expander(f"[{n}] {s['title']} — {s['section']}"):
                st.write(s["text"])
                if s["url"]:
                    st.markdown(f"[Open source]({s['url']})")
    with st.expander("Agent steps (ReAct trace)"):
        for t in r["trace"]:
            st.code(f"{t['tool']}({t['args']}) -> {t['n_results']} sources")
            if t.get("note"):
                st.caption(f"Corrective RAG: {t['note']}")
