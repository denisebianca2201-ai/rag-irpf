import httpx
import pytest
from typer.testing import CliRunner

from ask_the_lion.cli import app
from ask_the_lion.config import Settings
from ask_the_lion.download import DownloadError, download_pdf


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_saves_pdf(tmp_path):
    client = _client(lambda r: httpx.Response(200, content=b"%PDF-1.7 data"))
    dest = download_pdf("https://x.test/a.pdf", tmp_path / "sub" / "a.pdf", client=client)
    assert dest.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.rglob("*.part"))


def test_download_rejects_non_pdf(tmp_path):
    client = _client(lambda r: httpx.Response(200, content=b"<html>blocked</html>"))
    with pytest.raises(DownloadError):
        download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf", client=client)
    assert not (tmp_path / "a.pdf").exists()


def test_download_http_error(tmp_path):
    client = _client(lambda r: httpx.Response(403))
    with pytest.raises(DownloadError):
        download_pdf("https://x.test/a.pdf", tmp_path / "a.pdf", client=client)


def test_download_requires_url(tmp_path):
    with pytest.raises(DownloadError):
        download_pdf("", tmp_path / "a.pdf")


def test_settings_defaults_and_env(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")
    s = Settings(_env_file=None)
    assert s.top_k == 9 and s.max_question_chars == 1000


def test_cli_download_without_url_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PDF_URL", raising=False)
    result = CliRunner().invoke(app, ["download"])
    assert result.exit_code == 1


def test_cli_index_without_pdf_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["index"])
    assert result.exit_code == 1


def test_cli_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert all(cmd in result.output for cmd in ["download", "index", "ask", "chat"])
