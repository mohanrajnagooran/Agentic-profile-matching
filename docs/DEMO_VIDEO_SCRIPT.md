# Demo video script (≈5–6 minutes)

Record your screen with OBS Studio or the Windows Snipping Tool (screen recording); press Win + Alt + R to record with the Xbox Game Bar.
Run `streamlit run app.py` before you start recording. Copy and paste the messages below so you don't make typos on camera.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:30 | README, then `docs/state_machine.png` | "This is a LangGraph agent that matches 100 resumes to a job description. The main pipeline is Parse JD → Extract Requirements → Search Resumes → Rank → Report, then a human-feedback loop that routes each message to a handler." |
| 0:30–1:00 | `matching_agent.py`: the `AgentState` class and `build_graph()` | "The state keeps the conversation history, the current requirements and every version of them, scores for all candidates, the shortlist with reasoning, and a reasoning log. `human_feedback` uses `interrupt()` to wait for the user and `Command(goto=...)` to route." |
| 1:00–1:40 | Streamlit: choose the sample JD and press **Start matching** | "It pulled out the must-haves (React 3+ years, TypeScript, Node.js, REST APIs) and the nice-to-haves, used RAG search, scored all 100 resumes and built a top 10. Each row says why the candidate is there." Open the **Agent reasoning log** in the sidebar. |
| 1:40–2:20 | Type `Compare the top 3 matches side by side` | "This is the compare_candidates tool: must-have coverage with years of evidence, nice-to-haves, impact and leadership, and a verdict." |
| 2:20–3:00 | Type `Why did John rank higher than Jane?` | "Every score is a weighted sum of named parts, so the agent can show a point-by-point difference. It also shows where Jane is stronger and what would flip the result." |
| 3:00–3:50 | Type `Make AWS a must-have`, then ask `Why did John rank higher than Jane?` again | "I changed the requirements mid-conversation. It re-ranked and explained each move: candidates with AWS went up and John dropped because he has no AWS. When I ask the same question again, the answer reflects the new criteria." |
| 3:50–4:40 | Type `Run multi-round screening` | "Round 1 takes the top 10 out of 100. Round 2 looks deeper: years each skill is actually used in roles, skills listed but never used, seniority, measured impact, leadership and tenure. Round 3 gives Hire, Borderline or No Hire with a confidence level, plus suggestions for borderline candidates." |
| 4:40–5:20 | Type `Tell me about Jane Doe`, then `Interview questions for Jane Doe` | "This is a detailed match report with strengths, gaps and improvement suggestions. The interview questions dig into her real gaps, like TypeScript." |
| 5:20–5:50 | Terminal: `pytest -q`, then open `docs/TEST_SCENARIOS.md` | "There are six documented conversation flows, and they run as automated tests along with the unit tests. Everything passes." |
| 5:50–6:00 | — | "To sum up: it explains every ranking, re-ranks as the criteria change, and keeps a human in the loop." |

Tips: zoom the browser to about 110% so the text is readable. Close tabs you don't need. If you go over time, cut the interview-questions part first.
