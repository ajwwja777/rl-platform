import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('training_plot',Path(__file__).resolve().parents[1]/'scripts/plot_training_metrics.py')
plot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_missing_and_unupdated_values_are_not_fabricated(tmp_path):
    p=tmp_path/'metrics.jsonl'
    p.write_text('{"global_step":1,"actor_q":0,"did_actor_update":0,"q1_mean":0.5}\n'
                 '{"global_step":2,"actor_q":0.7,"did_actor_update":1,"q1_mean":0.6}\n'
                 '{"global_step":3',encoding='utf-8')
    rows,partial=plot.load_rows(p)
    assert partial==1
    assert plot.series(rows,'actor_q',True)==[(2,0.7)]
    page,missing=plot.render(rows,label='<unsafe>',role='training',source_sha='abc',incomplete=partial)
    assert 'q1_rewarded_chunk_mean' in missing
    assert '&lt;unsafe&gt;' in page
    assert 'Not recorded / no eligible rows' in page


def test_malformed_completed_rows_and_mixed_resume_steps_are_rejected(tmp_path):
    p=tmp_path/'bad.jsonl';p.write_text('bad\n{"global_step":2}\n')
    with pytest.raises(ValueError,match='Malformed completed'):plot.load_rows(p)
    p.write_text('{"global_step":2}\n{"global_step":1}\n')
    with pytest.raises(ValueError,match='Non-increasing'):plot.load_rows(p)


def test_identity_and_rows_come_from_one_snapshot_even_if_log_grows(tmp_path):
    p=tmp_path/'live.jsonl';original=b'{"global_step":1}\n'
    p.write_bytes(original+b'{"global_step":2}\n')
    rows,_=plot.load_rows(p,data=original)
    assert [r['global_step'] for r in rows]==[1]


def test_partially_missing_metric_is_not_connected_across_gap():
    rows=[{'global_step':1,'q1_mean':.1}, {'global_step':2},
          {'global_step':3,'q1_mean':.3}]
    page,_=plot.render(rows,label='gap',role='training',source_sha='abc')
    panel=page.split('<h3>Q1 mean (recorded actions)</h3>')[1].split('</article>')[0]
    assert '<polyline' not in panel
    assert panel.count('<circle')==2
    assert 'missing eligible rows=1' in panel
