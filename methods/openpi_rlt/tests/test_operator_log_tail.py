from __future__ import annotations


def test_operator_log_filter_hides_polling_and_warmup_noise_but_keeps_actions() -> None:
    from methods.openpi_rlt.scripts.operator_log_tail import visible_line

    assert not visible_line('[rlt-session] 127.0.0.1 "GET /api/session HTTP/1.1" 200 -')
    assert not visible_line("Waiting for warmup replay_size=0 required=600")
    assert visible_line("[rlt-live] chunk=12 latency=0.094s actor_version=-1")
    assert visible_line('[rlt-session] 127.0.0.1 "POST /api/episode/success HTTP/1.1" 200 -')
