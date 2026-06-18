from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from streamlit_orchestration import PIPELINE_STEP_NAMES, get_pipeline_step_agents

PIPELINE_PRESET_STEPS: dict[str, list[str]] = {
    "rag_presentation": [
        "📚 RAG Search (RAG SubAgent)",
        "📊 Create Presentation (Presenter)",
    ],
    "data_presentation": [
        "📊 Data Analysis (Data Scientist)",
        "📊 Create Presentation (Presenter)",
    ],
    "search_write": [
        "🌐 Web Search (Websearch)",
        "📝 Write Content (Writer)",
    ],
}


def render_sidebar(
    *,
    st: Any,
    session_state: Any,
    project_dir: Path,
    load_chat_sessions: Callable[..., Any],
    save_current_session: Callable[..., Any],
    new_chat_session: Callable[[], Any],
    switch_to_session: Callable[[str], Any],
    delete_session: Callable[[str], Any],
) -> str:
    """Render the Streamlit sidebar and return the active mode key."""
    with st.sidebar:
        st.markdown(
            """
        **Developed by**  
        👤 *Ven Seyhah*
        ## 🚀 Orchestration      
        """
        )

        mode_labels = ["💬 Agent", "📚 RAG", "🔬 Literature", "📊 Data Analysis"]
        mode_keys = ["main", "rag", "literature", "data_analysis"]
        mode_idx = mode_keys.index(session_state.active_mode) if session_state.active_mode in mode_keys else 0
        chosen_label = st.radio(
            "Navigation",
            mode_labels,
            index=mode_idx,
            label_visibility="collapsed",
            key="sidebar_mode_radio",
        )
        new_mode = mode_keys[mode_labels.index(chosen_label)]
        if new_mode != session_state.active_mode:
            session_state.active_mode = new_mode
            st.rerun()

        st.divider()

        load_chat_sessions(merge_only=True)

        if st.button("➕ New Chat", width="stretch", key="new_chat_btn"):
            new_chat_session()
            st.rerun()

        saved_sessions = session_state.chat_sessions
        if saved_sessions:
            with st.expander(f"💬 Chat History ({len(saved_sessions)})", expanded=False):
                for sid, info in sorted(
                    saved_sessions.items(),
                    key=lambda item: item[1].get("created", ""),
                    reverse=True,
                ):
                    is_active = sid == session_state.thread_id
                    label = info.get("name", sid[:20])
                    date_str = info.get("created", "")
                    col_btn, col_del = st.columns([5, 1])
                    with col_btn:
                        btn_label = f"{'▶ ' if is_active else ''}{label}"
                        if date_str:
                            btn_label += f"  _{date_str}_"
                        if st.button(
                            btn_label,
                            key=f"switch_{sid}",
                            width="stretch",
                            disabled=is_active,
                        ):
                            switch_to_session(sid)
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{sid}"):
                            delete_session(sid)
                            if is_active:
                                new_chat_session()
                            st.rerun()

        st.divider()

        with st.expander("🔀 Mission Pipeline", expanded=bool(session_state.pipeline_steps)):
            st.caption("Sequential workflow: each step delegates to a specialist in order.")
            steps = session_state.pipeline_steps
            if steps:
                for idx, step_name in enumerate(steps):
                    c_num, c_lbl, c_up, c_dn, c_rm = st.columns([0.4, 4, 0.6, 0.6, 0.6])
                    c_num.markdown(f"**{idx + 1}.**")
                    c_lbl.markdown(step_name)
                    with c_up:
                        if idx > 0 and st.button("⬆", key=f"pipe_up_{idx}"):
                            steps[idx - 1], steps[idx] = steps[idx], steps[idx - 1]
                            st.rerun()
                    with c_dn:
                        if idx < len(steps) - 1 and st.button("⬇", key=f"pipe_dn_{idx}"):
                            steps[idx], steps[idx + 1] = steps[idx + 1], steps[idx]
                            st.rerun()
                    with c_rm:
                        if st.button("✕", key=f"pipe_rm_{idx}"):
                            steps.pop(idx)
                            st.rerun()
                st.caption("Flow: " + " → ".join(get_pipeline_step_agents(steps)))
            else:
                st.info("No pipeline configured.")

            add_col, btn_col = st.columns([4, 1])
            with add_col:
                new_step = st.selectbox(
                    "Add step",
                    [""] + PIPELINE_STEP_NAMES,
                    index=0,
                    key="pipeline_add_step_select",
                    label_visibility="collapsed",
                )
            with btn_col:
                if st.button("➕", key="pipe_add_btn") and new_step:
                    session_state.pipeline_steps.append(new_step)
                    st.rerun()

            st.markdown("**Presets:**")
            p1, p2, p3 = st.columns(3)
            with p1:
                if st.button("🔍→📊", key="preset_rag_pres", help="RAG → Presentation"):
                    session_state.pipeline_steps = PIPELINE_PRESET_STEPS["rag_presentation"][:]
                    st.rerun()
            with p2:
                if st.button("📊→📊", key="preset_data_pres", help="Data → Presentation"):
                    session_state.pipeline_steps = PIPELINE_PRESET_STEPS["data_presentation"][:]
                    st.rerun()
            with p3:
                if st.button("🌐→📝", key="preset_search_write", help="Search → Write"):
                    session_state.pipeline_steps = PIPELINE_PRESET_STEPS["search_write"][:]
                    st.rerun()
            if steps and st.button("🗑️ Clear Pipeline", key="pipe_clear"):
                session_state.pipeline_steps = []
                st.rerun()

        st.divider()

        act1, act2 = st.columns(2)
        with act1:
            if st.button("🗑️ Clear Chat", width="stretch", key="clear_chat_btn"):
                if session_state.messages:
                    save_current_session()
                new_chat_session()
                st.rerun()
        with act2:
            if st.button("🗑️ Clear Plots", width="stretch", key="clear_plots_btn"):
                plots_dir = project_dir / "generated_plots"
                if plots_dir.exists():
                    import shutil

                    shutil.rmtree(plots_dir)
                    plots_dir.mkdir(exist_ok=True)
                    st.rerun()

        with st.expander("🛠️ Advanced Tools", expanded=False):
            if st.button("📦 Export Agent Package", key="export_package_btn", width="stretch"):
                try:
                    from package_agent import package_agent

                    zip_path = package_agent()
                    with open(zip_path, "rb") as zf:
                        st.download_button(
                            label=f"📥 Download {zip_path.name}",
                            data=zf.read(),
                            file_name=zip_path.name,
                            mime="application/zip",
                            key="download_package_btn",
                        )
                except Exception as e:
                    st.error(f"Package export failed: {e}")
            st.markdown("**Agent Protocol Server**")
            st.code("uvicorn server:app --port 2024", language="bash")
            st.caption("Exposes the agent via Agent Protocol API")
            st.markdown("**Ralph Mode (Autonomous)**")
            st.code('python ralph_mode.py "Your goal here"', language="bash")
            st.caption("Runs the agent autonomously in a loop")

        if session_state.messages:
            st.download_button(
                label="📥 Download Chat",
                data="\n".join(
                    f"**{message['role'].upper()}**: {message['content']}"
                    for message in session_state.messages
                ),
                file_name="chat_history.md",
                mime="text/markdown",
                width="stretch",
                key="download_chat_btn",
            )

    return session_state.active_mode
