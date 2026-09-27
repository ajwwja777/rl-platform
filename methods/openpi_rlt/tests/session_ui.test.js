const test = require("node:test");
const assert = require("node:assert/strict");
const { buttonsFor, guidanceText, statusText } = require("../web/rlt_session/app.js");

test("buttons are enabled only for valid session phases", () => {
  assert.equal(buttonsFor("ready").start, true);
  assert.equal(buttonsFor("rollout").success, true);
  assert.equal(buttonsFor("hil").success, true);
  assert.equal(buttonsFor("fault").next, false);
  assert.equal(buttonsFor("waiting_scene").next, true);
});

test("status text makes zero-publisher shadow explicit", () => {
  assert.match(statusText({ phase: "rollout", shadow_mode: true, chunk_count: 3 }), /SHADOW/);
  assert.match(statusText({ phase: "rollout", shadow_mode: true, chunk_count: 3 }), /3/);
});

test("operator guidance makes each gated phase actionable", () => {
  assert.match(guidanceText({ phase: "ready" }), /开始 Session/);
  assert.match(guidanceText({ phase: "rollout" }), /成功.*失败.*放弃/);
  assert.match(guidanceText({ phase: "terminal_pending" }), /已暂停.*成功.*失败.*放弃/);
  assert.match(guidanceText({ phase: "waiting_scene" }), /场景.*下一轮/);
});
