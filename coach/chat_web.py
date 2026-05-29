"""
chat_web.py — Streamlit chat UI on top of the Chat module.

Thin presentation layer. All routing, parsing, and agent logic lives in
coach/chat.py. The web app:
  - captures stdout from Chat.handle() and renders it as chat bubbles
  - persists chat history into state.json so a browser refresh keeps the conversation
  - renders a live "Current Program" table whenever there's an active block

Run with:
    streamlit run coach/chat_web.py
or:
    python main.py web
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import streamlit as st

from coach import blocks, guardrails, loading
from coach.chat import Chat


STATE_PATH = os.path.join("data", "state.json")


# --- helpers ---------------------------------------------------------------

def _looks_like_trace(text: str) -> bool:
    markers = ("[PERCEIVE", "[REASON", "[GUARD", "[DECIDE", "[ACT",
               "Block:", "Lifter:", "Training maxes:")
    return any(m in text for m in markers)


def _render_response(text: str) -> str:
    text = text.strip()
    if not text:
        return "_(no output)_"
    if _looks_like_trace(text):
        return f"```\n{text}\n```"
    return text


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            buf.write(f"\n(error: {type(e).__name__}: {e})")
    return buf.getvalue()


def _week_rows(state: dict, week: int, in_deload: bool = False) -> list[dict]:
    """Compute all exercise rows for a given week of the active block,
    flattened across days. Each row has Day, Exercise, Role, Sets×Reps,
    Weight, %TM, RPE, Plates/side."""
    block = state["block"]
    if not block.get("type"):
        return []
    plan_days = block.get("weekly_plan", {}).get("days", [])
    out = []
    for day in plan_days:
        day_label = day.get("label", "Day")
        for ex in day.get("exercises", []):
            tm = (state["lifts"].get(ex.get("lift"), {}).get("training_max")
                  if ex.get("lift") in ("squat", "bench", "deadlift") else None)
            c = blocks.compute_exercise_load(
                ex, tm, block["type"], week, in_deload,
                duration_weeks=block.get("duration_weeks"),
                readiness=state.get("readiness", 1.0))
            show_load = c["top_weight"] is not None
            wt_str = f"{c['top_weight']} kg" if show_load else "RPE-only"
            plates = loading.plate_breakdown(c["top_weight"]) if show_load else []
            plates_str = " + ".join(str(p) for p in plates) if plates else "—"
            pct_str = f"{int(c['intensity_pct']*100)}%" if show_load else "—"
            out.append({
                "Day": day_label,
                "Exercise": ex["name"],
                "Role": ex.get("role", "primary"),
                "Sets × Reps": f"{c['sets']} × {c['reps']}",
                "Weight": wt_str,
                "% TM": pct_str,
                "RPE": c["rpe_cap"],
                "Plates/side": plates_str,
            })
    return out


# --- panels ----------------------------------------------------------------

def _sidebar(chat: Chat) -> None:
    sb = st.sidebar
    sb.title("🏋️ Coach")
    sb.caption("Auto-regulating powerlifting agent")

    s = chat.agent.state
    lifter = s["lifter"]
    if not lifter.get("name"):
        sb.info("Set up the basics, then chat with the coach to diagnose "
                "your first block.")
        name = sb.text_input("Name", "Sam")
        sb.markdown("**Current 1RM (kg)**")
        squat = sb.number_input("Squat", 20.0, 500.0, 150.0, step=2.5)
        bench = sb.number_input("Bench", 20.0, 400.0, 100.0, step=2.5)
        dead = sb.number_input("Deadlift", 20.0, 500.0, 190.0, step=2.5)
        bw = sb.number_input("Bodyweight (kg)", 40.0, 200.0, 85.0, step=0.5)
        exp = sb.selectbox("Experience",
                           ["novice", "intermediate", "advanced"],
                           index=1)
        if sb.button("Start program", type="primary"):
            chat.agent.init_program(
                name,
                {"squat": squat, "bench": bench, "deadlift": dead},
                bodyweight_kg=bw, experience=exp,
            )
            chat.agent.state["chat_history"] = []
            chat.agent._save()
            st.session_state.history = []
            st.rerun()
    else:
        sb.markdown(f"**{lifter['name']}**")
        meta_bits = []
        if lifter.get("bodyweight_kg"):
            meta_bits.append(f"{lifter['bodyweight_kg']}kg")
        if lifter.get("experience"):
            meta_bits.append(lifter["experience"])
        if meta_bits:
            sb.caption(" · ".join(meta_bits))

        block = s["block"]
        if block.get("type"):
            week_label = ("DELOAD" if block.get("in_deload")
                          else f"week {block['week']}/{block['duration_weeks']}")
            sb.markdown(f"**Block:** `{block['label']}` · {week_label}")
            if block.get("rationale"):
                sb.caption(f"_{block['rationale']}_")
            if block.get("focus_lifts"):
                sb.caption(f"Focus: {', '.join(block['focus_lifts'])}")
        else:
            sb.warning("No block yet. Chat with the coach — they'll "
                       "diagnose and propose one.")
        sb.markdown(f"readiness: `{s['readiness']}`")
        sb.divider()
        sb.markdown("**Training maxes (kg)**")
        for lift in ("squat", "bench", "deadlift"):
            tm = s["lifts"][lift]["training_max"]
            sb.markdown(f"- {lift}: `{tm}`")
        if s.get("block_history"):
            sb.divider()
            sb.markdown(f"**Past blocks** ({len(s['block_history'])})")
            for h in s["block_history"][-3:]:
                sb.caption(f"`{h['type']}` · {h['duration_weeks']}wk · "
                           f"{h['outcome'][:60]}")

    sb.divider()
    llm = "🟢 OpenAI" if os.getenv("OPENAI_API_KEY") else "⚪ offline"
    sb.caption(f"LLM: {llm}")
    st.session_state.show_trace = sb.checkbox(
        "Show LLM tool-call trace", value=st.session_state.get("show_trace", False),
        help="Debug view: every tool the LLM called this turn, with its result.")
    if sb.button("Clear chat"):
        chat.agent.state["chat_history"] = []
        chat.agent._save()
        st.session_state.history = []
        st.rerun()
    if sb.button("⚠️ Reset program (wipe all data)"):
        if os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)
        st.session_state.chat = Chat(STATE_PATH)
        st.session_state.history = []
        st.rerun()


_ROLE_BADGE = {
    "primary":   ("🔴", "PRIMARY"),
    "secondary": ("🟡", "SECONDARY"),
    "accessory": ("🟢", "ACCESSORY"),
}


def _render_day_card(state: dict, day: dict, block_type: str,
                     week: int, in_deload: bool,
                     duration_weeks: int | None = None) -> None:
    """Render a single training day as a bordered card with role-grouped
    exercises — way more readable than a plain dataframe row.

    Rest days (label like 'Wednesday — Rest', empty exercises) render as a
    compact rest marker so the lifter sees the full week at a glance."""
    label = day.get("label", "Day")
    # Rest day: compact marker, no exercise rendering.
    if guardrails._is_rest_day(day):
        with st.container(border=True):
            st.markdown(f"### 🛌 {label}")
            st.caption("Rest / recovery day — no programmed work.")
        return
    with st.container(border=True):
        st.markdown(f"### 📍 {label}")
        # Render exercises grouped by role in canonical order
        order = ("primary", "secondary", "accessory")
        grouped: dict[str, list[dict]] = {r: [] for r in order}
        for ex in day.get("exercises", []):
            role = ex.get("role", "primary")
            grouped.setdefault(role, []).append(ex)

        for role in order:
            items = grouped.get(role) or []
            if not items:
                continue
            badge, header = _ROLE_BADGE[role]
            st.markdown(f"**{badge} {header}**")
            for ex in items:
                tm = (state["lifts"].get(ex.get("lift"), {}).get("training_max")
                      if ex.get("lift") in ("squat", "bench", "deadlift") else None)
                c = blocks.compute_exercise_load(
                    ex, tm, block_type, week, in_deload,
                    duration_weeks=duration_weeks,
                    readiness=state.get("readiness", 1.0))
                # Hide kg / %TM / plates when intensity_pct signals a
                # non-bar exercise (DB lateral raise, cable pushdown,
                # machine). The LLM sets intensity_pct=0 in that case.
                show_load = c["top_weight"] is not None
                if show_load:
                    load_str = (f"@ **{c['top_weight']} kg**  ·  "
                                f"{int(c['intensity_pct']*100)}% TM")
                else:
                    load_str = "(use a weight that hits target RPE)"
                # Main exercise line
                col_left, col_right = st.columns([5, 1])
                with col_left:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{ex['name']} &nbsp;—&nbsp; "
                        f"{c['sets']} × {c['reps']} {load_str}",
                        unsafe_allow_html=True,
                    )
                with col_right:
                    st.markdown(f"`RPE {c['rpe_cap']}`")
                # Sub-line: plates + rationale
                sub_bits = []
                if show_load:
                    plates = loading.plate_breakdown(c["top_weight"])
                    if plates:
                        sub_bits.append(
                            "plates/side: " + " · ".join(str(p) for p in plates))
                if ex.get("rationale"):
                    sub_bits.append(f"_{ex['rationale']}_")
                if sub_bits:
                    st.markdown(
                        "&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#888; "
                        "font-size:0.85em'>" + "  ·  ".join(sub_bits) + "</span>",
                        unsafe_allow_html=True,
                    )


_BLOCK_BADGE = {
    "volume":    ("🟦", "Volume"),
    "technique": ("🟪", "Technique"),
    "strength":  ("🟧", "Strength"),
    "peaking":   ("🟥", "Peaking"),
}


def _season_summary_strip(chat: Chat) -> None:
    """Tiny one-line season strip rendered above the chat tab — meet date,
    block badges, current marker. Full detail lives on the Season Plan tab."""
    state = chat.agent.state
    sp = state.get("season_plan", {}) or {}
    history = state.get("block_history", []) or []
    upcoming = sp.get("blocks", []) or []
    active = state.get("block", {})
    has_active = bool(active.get("type"))
    if not history and not upcoming and not has_active:
        return
    chips: list[str] = []
    for h in history:
        b, n = _BLOCK_BADGE.get(h.get("type", ""), ("⚪", h.get("type", "?")))
        chips.append(f"{b}~~{n}~~")
    if has_active:
        b, n = _BLOCK_BADGE.get(active.get("type", ""),
                                 ("⚪", active.get("type", "?")))
        chips.append(f"**▶️ {b} {n}**")
    for u in upcoming:
        b, n = _BLOCK_BADGE.get(u.get("block_type", ""),
                                 ("⚪", u.get("block_type", "?")))
        chips.append(f"{b} {n}")
    line = "  →  ".join(chips)
    meet = sp.get("competition_date")
    if meet:
        line = f"🗓️ **{meet}**  ·  " + line
    st.caption(line + "  ·  see full timeline on the **Season Plan** tab.")


def _season_panel(chat: Chat) -> None:
    """FULL macrocycle view — rendered on the dedicated Season Plan tab.
    Past blocks (with outcomes), current block (with progress), and
    tentative future blocks toward the competition."""
    state = chat.agent.state
    sp = state.get("season_plan", {}) or {}
    history = state.get("block_history", []) or []
    upcoming = sp.get("blocks", []) or []
    comp_date = sp.get("competition_date")
    active = state.get("block", {})
    has_active = bool(active.get("type"))

    if not history and not upcoming and not has_active:
        st.info("No season plan yet — chat with the coach about your "
                "competition date and target lifts. They'll lay out a "
                "macrocycle (volume → strength → … → peaking) ending on "
                "your meet date.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.metric("Competition", comp_date or "—")
    with cols[1]:
        st.metric("Blocks completed", len(history))
    with cols[2]:
        st.metric("Blocks remaining",
                  len(upcoming) + (1 if has_active else 0))
    if sp.get("competition_lifts"):
        cl = sp["competition_lifts"]
        targets = " · ".join(f"**{lf}** {cl.get(lf)} kg"
                             for lf in ("squat", "bench", "deadlift")
                             if cl.get(lf))
        if targets:
            st.caption(f"🎯 target lifts: {targets}")

    st.divider()

    # ---- Past blocks -------------------------------------------------------
    if history:
        st.subheader("✅ Completed")
        for i, h in enumerate(history, 1):
            badge, name = _BLOCK_BADGE.get(h.get("type", ""),
                                           ("⚪", h.get("type", "?")))
            header = (f"✅ **{badge} {name}**  ·  "
                      f"{h.get('duration_weeks', '?')}wk  ·  "
                      f"{h.get('started_at', '?')} → "
                      f"{h.get('completed_at', '?')}")
            with st.expander(header, expanded=False):
                if h.get("rationale"):
                    st.markdown(f"**Why:** {h['rationale']}")
                if h.get("focus_lifts"):
                    st.markdown(f"**Focus:** {', '.join(h['focus_lifts'])}")
                if h.get("outcome"):
                    st.success(f"**Outcome:** {h['outcome']}")
                # Past programs are persisted in the history record now.
                weekly_plan = h.get("weekly_plan") or {}
                plan_days = weekly_plan.get("days", [])
                if plan_days:
                    st.markdown("**Program that was run:**")
                    for day in plan_days:
                        _render_day_card(state, day, h.get("type", "volume"),
                                         week=1, in_deload=False,
                                         duration_weeks=h.get("duration_weeks"))
                else:
                    st.caption("(this block was archived before per-block "
                               "program persistence — no detailed plan stored)")

    # ---- Current block -----------------------------------------------------
    if has_active:
        st.subheader("▶️ Currently active")
        badge, name = _BLOCK_BADGE.get(active.get("type", ""),
                                       ("⚪", active.get("type", "?")))
        wk = ("DELOAD" if active.get("in_deload")
              else f"week {active.get('week', '?')}/"
                   f"{active.get('duration_weeks', '?')}")
        header = (f"▶️ **{badge} {name}**  ·  {wk}  ·  "
                  f"started {active.get('started_at', '?')}")
        with st.expander(header, expanded=True):
            if active.get("rationale"):
                st.markdown(f"**Why:** {active['rationale']}")
            if active.get("focus_lifts"):
                st.markdown(f"**Focus:** {', '.join(active['focus_lifts'])}")
            plan_days = active.get("weekly_plan", {}).get("days", [])
            if plan_days:
                st.markdown("**This week's program:**")
                for day in plan_days:
                    _render_day_card(state, day, active["type"],
                                     active.get("week", 1),
                                     active.get("in_deload", False),
                                     duration_weeks=active.get("duration_weeks"))
            else:
                st.caption("(no weekly plan structured yet)")

    # ---- Future blocks -----------------------------------------------------
    if upcoming:
        st.subheader("🔜 Coming up")
        for i, u in enumerate(upcoming, 1):
            bt = u.get("block_type", "?")
            badge, name = _BLOCK_BADGE.get(bt, ("⚪", bt))
            ps = u.get("planned_start") or "tbd"
            # Plain card — NO expander. Upcoming blocks are intentionally
            # high-level (type only); the exact exercises and loads get
            # built fresh when the block becomes active, adapted to how
            # the previous block actually went.
            with st.container(border=True):
                cols = st.columns([1, 4, 3])
                with cols[0]:
                    st.markdown(f"<span style='font-size:1.5em'>🔜</span><br>"
                                f"<span style='color:#888'>#{i}</span>",
                                unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"**{badge} {name}**  "
                                f"_({u.get('duration_weeks', '?')}wk)_")
                    st.caption(f"planned start: {ps}")
                with cols[2]:
                    if u.get("rationale"):
                        st.caption(u["rationale"])
                    if u.get("focus_lifts"):
                        st.caption("focus: " + ", ".join(u["focus_lifts"]))
        st.caption("ℹ️ Upcoming blocks show **type only** — exercises and "
                   "loads are built fresh per block, adapting to how the "
                   "previous block went (e.g. \"tricep struggled during "
                   "tempo bench → next block uses close-grip\", \"DL "
                   "topsets too heavy → next block drops intensity\").")
    elif has_active:
        # Active block exists but no upcoming blocks queued — most likely
        # the LLM committed the first block but skipped propose_season.
        # Prominent CTA so the lifter knows what to ask for next.
        st.warning(
            "🔜 **No upcoming blocks planned yet.** Your full macrocycle "
            "(the sequence of blocks leading up to your competition) "
            "hasn't been laid out. Ask the coach in chat:\n\n"
            "> *\"Lay out the full macrocycle from now until my "
            "competition — all the blocks I'll do, with exercise outlines "
            "for each.\"*\n\n"
            "The coach should respond by calling `propose_season`, which "
            "will populate this tab with click-to-expand previews for "
            "every upcoming block.")


def _program_panel(chat: Chat) -> None:
    """Always-visible block view: tabs for THIS WEEK + FULL BLOCK (all weeks).
    Pulls everything from the LLM-defined weekly_plan and applies the
    block's per-week wave for the numbers."""
    state = chat.agent.state
    block = state["block"]
    if not block.get("type"):
        return
    if not block.get("weekly_plan", {}).get("days"):
        st.warning("Block committed but no weekly plan yet — ask the coach "
                   "to design one in chat.")
        return

    duration = block.get("duration_weeks") or 1
    current_week = block.get("week", 1)
    in_deload = block.get("in_deload", False)
    wk_mod = blocks.week_modifier(block["type"], current_week)
    current_label = ("DELOAD" if in_deload
                     else f"Week {current_week}/{duration} — {wk_mod['label']}")

    st.markdown(f"### 📋 {block['label']} · {current_label}")
    if block.get("rationale"):
        st.caption(f"💡 {block['rationale']}")
    if block.get("focus_lifts"):
        st.caption(f"🎯 Focus: {', '.join(block['focus_lifts'])}")

    tabs = st.tabs(["This Week", "Full Block"])

    # ---- This Week tab: day-cards with role-grouped exercises --
    with tabs[0]:
        plan_days = block.get("weekly_plan", {}).get("days", [])
        if not plan_days:
            st.info("No exercises in the weekly plan.")
        else:
            for day in plan_days:
                _render_day_card(state, day, block["type"],
                                 current_week, in_deload,
                                 duration_weeks=block.get("duration_weeks"))

    # ---- Full Block tab: nested week-tabs, each rendered as day-cards.
    # Same readable format as the "This Week" tab — switch between weeks
    # to see how the wave reshapes intensity and volume across the block.
    with tabs[1]:
        plan_days = block.get("weekly_plan", {}).get("days", [])
        if not plan_days:
            st.info("No exercises in the weekly plan.")
        else:
            if in_deload:
                st.info("Currently on a deload week. The tabs below show "
                        "the productive weeks of this block.")
            week_labels = [f"Week {wk}" for wk in range(1, duration + 1)]
            week_tabs = st.tabs(week_labels)
            for wk_idx, week_tab in enumerate(week_tabs):
                wk = wk_idx + 1
                with week_tab:
                    mod = blocks.week_modifier(block["type"], wk)
                    reps, rpe = blocks.top_set_for_week(
                        block["type"], wk, duration_weeks=duration)
                    st.caption(f"📈 {mod['label']}  ·  "
                               f"top set ≈ {reps} rep(s) @ RPE {rpe}")
                    for day in plan_days:
                        _render_day_card(state, day, block["type"], wk,
                                         in_deload=False,
                                         duration_weeks=duration)


# --- main ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Powerlifting Coach", page_icon="🏋️",
                       layout="wide")
    st.title("🏋️ Powerlifting Coach")
    st.caption("An auto-regulating training agent. Log sessions, ask "
               "questions, get programming.")

    if "chat" not in st.session_state:
        st.session_state.chat = Chat(STATE_PATH)
    chat: Chat = st.session_state.chat

    # Restore conversation from the persisted chat history on first load.
    if "history" not in st.session_state:
        persisted = chat.agent.state.get("chat_history", [])
        st.session_state.history = [(h["role"], h["content"]) for h in persisted]

    _sidebar(chat)

    # Two top-level tabs: Chat (current block + conversation) and Season Plan
    # (past / current / future blocks). Keeps the chat area uncluttered while
    # giving past + future blocks their own room to breathe.
    tab_chat, tab_season = st.tabs(["💬 Chat & current block",
                                    "🗓️ Season plan"])

    with tab_season:
        _season_panel(chat)

    with tab_chat:
        _season_summary_strip(chat)
        _program_panel(chat)

        for role, content in st.session_state.history:
            with st.chat_message(role):
                st.markdown(content, unsafe_allow_html=False)

        # Latest LLM tool trace, rendered AFTER history so it survives reruns.
        last_trace = getattr(chat, "last_trace", None) or []
        if st.session_state.get("show_trace") and last_trace:
            with st.expander(f"🔧 Last LLM tool trace ({len(last_trace)} calls)",
                             expanded=True):
                for i, (name, args, result) in enumerate(last_trace, 1):
                    has_error = isinstance(result, dict) and "error" in result
                    badge = "❌" if has_error else "✓"
                    st.markdown(f"**{i}. `{name}` {badge}**")
                    st.json({"input": args, "result": result})

    lifter_set = bool(chat.agent.state["lifter"].get("name"))
    block_set = bool(chat.agent.state["block"].get("type"))
    if not lifter_set:
        placeholder = "set up your numbers in the sidebar first…"
    elif not block_set:
        placeholder = "tell me what you're training for and what's bugging you…"
    else:
        placeholder = "log a session or ask me anything…"
    # chat_input lives at the page level (outside tabs) so it's always
    # visible — Streamlit anchors it to the bottom of the page.
    prompt = st.chat_input(placeholder)

    if prompt:
        st.session_state.history.append(("user", prompt))
        chat.agent.state["chat_history"].append({"role": "user", "content": prompt})
        with tab_chat:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.status("thinking…", expanded=True) as status:
                    def _on_step(msg: str) -> None:
                        status.write(msg)
                    output = _capture(chat.handle, prompt, on_step=_on_step)
                    status.update(label="done", state="complete", expanded=False)
                rendered = _render_response(output)
                st.markdown(rendered)
        st.session_state.history.append(("assistant", rendered))
        chat.agent.state["chat_history"].append(
            {"role": "assistant", "content": rendered})
        chat.agent._save()
        st.rerun()


if __name__ == "__main__":
    main()
