#!/usr/bin/env python3
"""CLI de avaliacao do mcp-brasil.

Recebe uma pergunta em linguagem natural, sobe o servidor mcp-brasil via
stdio e roda um tool-use loop com Claude (Anthropic) ate produzir a
resposta final.

Tres escopos de tools:

- legislation (default): forca `MCP_BRASIL_TOOL_SEARCH=none` no servidor,
  pega as ~317 tools nativas e filtra client-side para apenas DOU,
  Camara, Senado, jurisprudencia, TCU + meta (~68 tools).
- smart: usa o default `MCP_BRASIL_TOOL_SEARCH=bm25` do mcp-brasil, em
  que o servidor expoe so 7 meta-tools; o proprio Claude descobre e
  invoca features dinamicamente via `search_tools` + `call_tool`.
- all: forca `MCP_BRASIL_TOOL_SEARCH=none` e expoe todas as ~317 tools
  diretamente (caro, lento — modo debug).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from typing import Any

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_TOOL_ITERS = 15

# Escopo legislation: prefixos de tools cobertas.
LEGISLATION_FEATURE_PREFIXES = (
    "diario_oficial_",
    "camara_",
    "senado_",
    "jurisprudencia_",
    "tcu_",
)

# Meta-tools do mcp-brasil (sempre incluidas em qualquer escopo filtrado).
META_TOOLS = (
    "listar_features",
    "recomendar_tools",
    "planejar_consulta",
    "executar_lote",
    "listar_datasets_disponiveis",
)

SYSTEM_PROMPT = (
    "Voce e um assistente que responde perguntas sobre dados publicos "
    "brasileiros usando as ferramentas disponiveis do servidor mcp-brasil. "
    "Use as ferramentas necessarias para coletar evidencia antes de "
    "responder. Cite a fonte (orgao/sistema) quando aplicavel. Responda "
    "sempre em portugues do Brasil, de forma objetiva."
)


def build_server_params(scope: str) -> StdioServerParameters:
    """Retorna parametros para subir o mcp-brasil via stdio.

    Em escopos `legislation` e `all`, forca `MCP_BRASIL_TOOL_SEARCH=none`
    para que as ~312 tools nativas apareçam no `list_tools` do MCP e
    possam ser filtradas client-side. Em `smart`, mantem o default `bm25`
    do mcp-brasil — apenas 7 meta-tools sao expostas e o proprio servidor
    faz proxy via `call_tool`/`search_tools`.

    Prefere `uvx` (forma oficial do README do mcp-brasil); cai para
    `python -m mcp_brasil.server` se `uvx` nao estiver no PATH.
    """
    env = os.environ.copy()
    if scope == "smart":
        env.setdefault("MCP_BRASIL_TOOL_SEARCH", "bm25")
    else:
        env["MCP_BRASIL_TOOL_SEARCH"] = "none"

    if shutil.which("uvx"):
        return StdioServerParameters(
            command="uvx",
            args=["--from", "mcp-brasil", "python", "-m", "mcp_brasil.server"],
            env=env,
        )

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_brasil.server"],
        env=env,
    )


def is_legislation_tool(name: str) -> bool:
    if name in META_TOOLS:
        return True
    return any(name.startswith(p) for p in LEGISLATION_FEATURE_PREFIXES)


def select_initial_tools(all_tools: list[Any], scope: str) -> list[Any]:
    """Filtra a lista de tools retornada pelo MCP conforme o escopo.

    - all: passa tudo (com `MCP_BRASIL_TOOL_SEARCH=none`, sao ~312 tools).
    - legislation: filtra por prefixo de feature legislativa/regulatoria.
    - smart: passa tudo (com `MCP_BRASIL_TOOL_SEARCH=bm25` o servidor ja
      expoe so 7 meta-tools; o proxy via `call_tool`/`search_tools` cobre
      as 533 dinamicamente).
    """
    if scope in {"all", "smart"}:
        return list(all_tools)
    if scope == "legislation":
        return [t for t in all_tools if is_legislation_tool(t.name)]
    raise ValueError(f"Escopo desconhecido: {scope}")


def mcp_tool_to_anthropic(tool: Any) -> dict[str, Any]:
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": schema,
    }


def stringify_tool_result(call_result: Any) -> str:
    """Converte o `CallToolResult` do MCP em string para enviar ao Claude.

    Concatena todos os blocos de texto. Para outros tipos (image/embedded
    resource), serializa em JSON best-effort.
    """
    parts: list[str] = []
    for block in call_result.content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        try:
            parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
        except Exception:
            parts.append(str(block))
    return "\n".join(parts) if parts else ""


async def run_one(
    session: ClientSession,
    tools_payload: list[dict[str, Any]],
    question: str,
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
    max_iters: int,
    verbose: bool,
) -> str:
    """Executa o tool-use loop para uma unica pergunta. Retorna o texto."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for iteration in range(max_iters):
        if verbose:
            print(
                f"[turn {iteration + 1}] enviando {len(tools_payload)} tools "
                f"para o modelo",
                file=sys.stderr,
            )

        response = await client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools_payload,
            messages=messages,
        )

        assistant_blocks = [
            b.model_dump(exclude_none=True) for b in response.content
        ]
        messages.append({"role": "assistant", "content": assistant_blocks})

        if response.stop_reason != "tool_use":
            texts = [
                b.text for b in response.content if getattr(b, "type", "") == "text"
            ]
            return "\n".join(t for t in texts if t).strip()

        tool_results_blocks: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input or {}

            if verbose:
                print(
                    f"[tool_call] {tool_name} args="
                    f"{json.dumps(tool_args, ensure_ascii=False)[:300]}",
                    file=sys.stderr,
                )

            try:
                call_result = await session.call_tool(tool_name, tool_args)
                content_text = stringify_tool_result(call_result)
                is_error = bool(getattr(call_result, "isError", False))
            except Exception as exc:
                content_text = f"Erro ao executar tool {tool_name}: {exc}"
                is_error = True

            if not content_text:
                content_text = "(tool retornou conteudo vazio)"

            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_text,
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results_blocks})

    return (
        "[limite de iteracoes atingido sem resposta final; rode com "
        "--max-tool-iters maior ou --verbose para investigar]"
    )


async def list_tools_only(scope: str) -> None:
    server_params = build_server_params(scope)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.list_tools()
            selected = select_initial_tools(res.tools, scope)
            for t in selected:
                print(t.name)
            print(
                f"\nScope={scope}: {len(selected)} tool(s) enviadas ao "
                f"modelo (de {len(res.tools)} expostas pelo servidor).",
                file=sys.stderr,
            )


async def amain(args: argparse.Namespace) -> int:
    if args.list_tools:
        await list_tools_only(args.scope)
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERRO: ANTHROPIC_API_KEY nao definida. Configure no .env ou "
            "exporte no shell.",
            file=sys.stderr,
        )
        return 2

    model = args.model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    client = anthropic.AsyncAnthropic(api_key=api_key)

    server_params = build_server_params(args.scope)

    if args.verbose:
        print(
            f"[startup] subindo mcp-brasil ({server_params.command} "
            f"{' '.join(server_params.args)})",
            file=sys.stderr,
        )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            all_tools = tools_result.tools

            initial = select_initial_tools(all_tools, args.scope)
            tools_payload = [mcp_tool_to_anthropic(t) for t in initial]

            if args.verbose:
                print(
                    f"[startup] {len(all_tools)} tools no servidor; "
                    f"escopo={args.scope} -> {len(tools_payload)} tools "
                    f"para o modelo",
                    file=sys.stderr,
                )

            async def ask(question: str) -> str:
                return await run_one(
                    session,
                    tools_payload,
                    question,
                    client=client,
                    model=model,
                    max_iters=args.max_tool_iters,
                    verbose=args.verbose,
                )

            if args.question:
                try:
                    answer = await ask(args.question)
                except anthropic.AuthenticationError as exc:
                    print(
                        f"ERRO de autenticacao na API Anthropic: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                except anthropic.APIError as exc:
                    print(
                        f"ERRO da API Anthropic: {exc}", file=sys.stderr
                    )
                    return 1
                print(answer)
                return 0

            print(
                f"mcp-brasil-eval REPL [scope={args.scope} model={model}] "
                "(Ctrl-D ou 'exit' para sair)",
                file=sys.stderr,
            )
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print(file=sys.stderr)
                    return 0
                if not line:
                    continue
                if line.lower() in {"exit", "quit", ":q"}:
                    return 0
                try:
                    answer = await ask(line)
                except anthropic.AuthenticationError as exc:
                    print(
                        f"[erro] autenticacao Anthropic: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                except Exception as exc:  # noqa: BLE001
                    print(f"[erro] {exc}", file=sys.stderr)
                    continue
                print(answer)
                print()


def main() -> None:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

    parser = argparse.ArgumentParser(
        description=(
            "CLI de avaliacao do mcp-brasil: recebe pergunta em linguagem "
            "natural e responde via Claude + tools do mcp-brasil."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Pergunta one-shot. Sem este argumento, abre REPL.",
    )
    parser.add_argument(
        "--scope",
        choices=("legislation", "smart", "all"),
        default="legislation",
        help=(
            "Escopo de tools expostas ao Claude. "
            "legislation=DOU/Camara/Senado/jurisprudencia/TCU + meta "
            "(default, ~68 tools); smart=7 meta-tools com proxy nativo do "
            "mcp-brasil via call_tool/search_tools; "
            "all=todas as ~317 tools nativas (lento/caro)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Modelo Anthropic. Default: ANTHROPIC_MODEL env ou "
            f"{DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--max-tool-iters",
        type=int,
        default=DEFAULT_MAX_TOOL_ITERS,
        help=f"Max iteracoes do tool-use loop (default {DEFAULT_MAX_TOOL_ITERS}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime tool calls e diagnosticos no stderr.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Lista todas as tools do mcp-brasil e sai (debug).",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
