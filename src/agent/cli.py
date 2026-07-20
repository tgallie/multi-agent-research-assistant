"""Command-line entrypoint for research runs."""

import json

import typer

from agent.factory import build_graph

app = typer.Typer(no_args_is_help=True)


@app.command()
def query(question: str, trace: bool = typer.Option(False, "--trace")) -> None:
    """Run the current research pipeline for one question."""

    result = build_graph().run(question)
    payload = result.model_dump(mode="json") if trace else result.output.model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
