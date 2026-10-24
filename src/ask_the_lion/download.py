from pathlib import Path

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; ask-the-lion/0.1)"


class DownloadError(RuntimeError):
    pass


def download_pdf(
    url: str,
    dest: Path,
    client: httpx.Client | None = None,
) -> Path:
    """Download the PDF to `dest`, validating that the payload really is a PDF."""
    if not url:
        raise DownloadError("PDF_URL is not set; configure it in .env")

    owns_client = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=60, headers={"User-Agent": USER_AGENT}
    )
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not download {url}: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    if not response.content.startswith(b"%PDF"):
        raise DownloadError(f"{url} did not return a PDF")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest
