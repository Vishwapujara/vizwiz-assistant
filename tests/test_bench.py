from vizwiz.bench.sweep import delta, parse_prometheus, read_locust_csv, stage_means_ms

BEFORE = """# TYPE vizwiz_requests_total counter
vizwiz_requests_total{status="answered"} 10.0
vizwiz_stage_seconds_sum{stage="answer"} 2.0
vizwiz_stage_seconds_count{stage="answer"} 10.0
"""
AFTER = """# TYPE vizwiz_requests_total counter
vizwiz_requests_total{status="answered"} 30.0
vizwiz_requests_total{status="unanswerable"} 5.0
vizwiz_stage_seconds_sum{stage="answer"} 6.0
vizwiz_stage_seconds_count{stage="answer"} 30.0
"""


def test_prometheus_deltas_and_stage_means():
    b, a = parse_prometheus(BEFORE), parse_prometheus(AFTER)
    assert delta(a, b, "vizwiz_requests_total", "status") == {"answered": 20.0, "unanswerable": 5.0}
    assert stage_means_ms(a, b) == {"answer": 200.0}  # 4 s over 20 calls


def test_read_locust_csv(tmp_path):
    header = ("Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
              "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
              "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%")  # fmt: skip
    nums = "120,1,410,450.5,200,1900,800,2.0,0.01,410,500,520,560,700,900,1200,1500,1800,1900,1900"
    rows = [f"POST,assist,{nums}", f",Aggregated,{nums}"]
    (tmp_path / "run_stats.csv").write_text("\n".join([header, *rows]) + "\n")
    s = read_locust_csv(str(tmp_path / "run"))
    assert s == {"requests": 120, "failures": 1, "rps": 2.0, "mean_ms": 450.5,
                 "p50_ms": 410.0, "p90_ms": 700.0, "p95_ms": 900.0, "p99_ms": 1500.0}  # fmt: skip
