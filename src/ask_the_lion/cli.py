import typer

from ask_the_lion.config import Settings

app = typer.Typer(help="Ask the Lion: RAG over the official IRPF Q&A document.")


@app.command()
def download() -> None:
    """Download the official Q&A PDF."""
    from ask_the_lion.download import DownloadError, download_pdf

    settings = Settings()
    try:
        path = download_pdf(settings.pdf_url, settings.pdf_path)
    except DownloadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Saved {path}")


@app.command()
def index() -> None:
    """Split the PDF into questions and build the vector index."""
    from ask_the_lion.embeddings import build_embeddings
    from ask_the_lion.ingest import ingest_pdf

    settings = Settings()
    if not settings.pdf_path.exists():
        typer.echo(f"Error: {settings.pdf_path} not found; run `download` first", err=True)
        raise typer.Exit(1)
    count = ingest_pdf(settings.pdf_path, build_embeddings(settings), settings)
    typer.echo(f"Indexed {count} chunks")


def _print_result(result) -> None:
    typer.echo(result.answer)
    for source in result.sources:
        typer.echo(f"  - Q{source.question_number} (page {source.page})")


@app.command()
def ask(question: str) -> None:
    """Ask a single question."""
    from ask_the_lion.rag import build_pipeline

    _print_result(build_pipeline(Settings()).ask(question))


@app.command()
def chat() -> None:
    """Interactive chat; empty line or Ctrl-D to quit."""
    from ask_the_lion.rag import build_pipeline

    pipeline = build_pipeline(Settings())
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        _print_result(pipeline.ask(question))
