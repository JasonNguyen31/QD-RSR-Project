from src.common.api import ChatClient, ChatResult, EmptyResponse
from src.common.config import load_config
from src.tools import test_api as tool


class Fake(ChatClient):
    def __init__(self, results):
        super().__init__("u", "k", _sdk_client=object())
        self.results = results

    def chat(self, model, messages, temperature, top_p, max_tokens, provider_order=None):
        r = self.results[model]
        if isinstance(r, Exception):
            raise r
        self.tracker.add_ok(r)
        return r


def res(text, cost):
    return ChatResult(text, "stop", 5, 5, cost, 0.5, "m")


def test_check_teachers_counts_failures_and_reports_cost(capsys):
    cfg = load_config()
    ids = [t["model_id"] for t in cfg["teachers"]]
    fake = Fake({ids[0]: res("\\boxed{391}", 0.0001), ids[1]: EmptyResponse("rỗng"), ids[2]: res("391", None)})
    assert tool.check_teachers(cfg, fake) == 1
    out = capsys.readouterr().out
    assert "[LỖI] qwen72b" in out and "KHÔNG" in out
