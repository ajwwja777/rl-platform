function buttonsFor(phase) {
  return {
    start: phase === "ready",
    success: ["rollout", "hil", "paused", "terminal_pending"].includes(phase),
    failure: ["rollout", "hil", "paused", "terminal_pending"].includes(phase),
    abort: ["rollout", "hil", "paused", "terminal_pending"].includes(phase),
    pause: ["rollout", "hil"].includes(phase),
    resume: phase === "paused",
    next: phase === "waiting_scene",
    stop: !["disarmed", "stopped"].includes(phase),
  };
}

function statusText(state) {
  const mode = state.shadow_mode ? "SHADOW（零发布）" : "LIVE";
  return `${mode} · ${String(state.phase || "unknown").toUpperCase()} · chunk ${state.chunk_count || 0}`;
}

function guidanceText(state) {
  const phase = state.phase;
  if (phase === "ready") return "模型已就绪且保持暂停：点击“开始 Session”后才开始本轮推理与录制。";
  if (phase === "rollout") return "本轮正在推理：完成后请选择成功、失败或放弃；示教按钮可随时接管。";
  if (phase === "hil") return "HIL 接管中：策略已暂停；释放最后一个示教按钮后会 fresh replan。";
  if (phase === "paused") return "本轮已人工暂停：可继续，也可选择成功、失败或放弃。";
  if (phase === "terminal_pending") return "本轮已暂停并等待终局：现在请选择成功、失败或放弃。";
  if (phase === "replay_committing") return state.home_after_terminal
    ? "正在固化本轮数据；完成后将自动执行前双臂归位，请勿进入机械臂工作区。"
    : "正在固化本轮数据，请稍候。";
  if (phase === "waiting_scene") return "本轮已保存：请复位场景，然后点击“场景已复位，开始下一轮”。";
  if (phase === "fault") return "系统已 fail-closed：不要继续动作，请查看下方故障信息。";
  if (phase === "stopped") return "Session 已结束。";
  return "正在准备 RLT Session。";
}

if (typeof module !== "undefined") module.exports = { buttonsFor, guidanceText, statusText };

if (typeof document !== "undefined") {
  const $ = (id) => document.getElementById(id);
  const actions = {
    start: "/api/session/start",
    success: "/api/episode/success",
    failure: "/api/episode/failure",
    abort: "/api/episode/abort",
    pause: "/api/session/pause",
    resume: "/api/session/resume",
    next: "/api/episode/next",
    stop: "/api/session/stop",
  };
  let current = null;
  let posting = false;

  function render(state) {
    current = state;
    $("headline").textContent = statusText(state);
    $("guidance").textContent = guidanceText(state);
    $("identity").textContent = `session ${state.session_id || "—"} / episode ${state.episode_id}`;
    const task5Index = state.task5_episode_index == null ? "—" : state.task5_episode_index;
    const actorVersion = state.actor_version == null ? "—" : state.actor_version;
    const learnerVersion = state.learner_version == null ? "—" : state.learner_version;
    const mask = state.expert_mask || [false, false];
    $("task5").textContent = `Task5 index ${task5Index} / UUID ${state.task5_episode_uuid || "pending"}`;
    $("hil").textContent = `HIL left=${Boolean(mask[0])} right=${Boolean(mask[1])}`;
    $("versions").textContent = `actor ${actorVersion} / learner ${learnerVersion}`;
    $("fault").textContent = state.fault_reason ? `FAULT: ${state.fault_reason}` : "";
    const enabled = buttonsFor(state.phase);
    for (const name of Object.keys(actions)) $(name).disabled = posting || !enabled[name];
  }

  async function refresh() {
    try {
      const response = await fetch("/api/session", { cache: "no-store" });
      if (!response.ok) throw new Error(`status HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      $("fault").textContent = `页面状态不可用：${error.message}`;
    }
  }

  async function act(name) {
    if (!current || posting) return;
    posting = true;
    render(current);
    try {
      const response = await fetch(actions[name], {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ episode_id: current.episode_id, generation: current.generation }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
      render(payload);
    } catch (error) {
      $("fault").textContent = `操作未执行：${error.message}`;
    } finally {
      posting = false;
      await refresh();
    }
  }

  for (const name of Object.keys(actions)) $(name).addEventListener("click", () => act(name));
  refresh();
  setInterval(refresh, 500);
}
