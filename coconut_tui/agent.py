"""Orchestrates a single user turn: dispatches slash-commands to Tools, or
plain text to the LLMProvider, publishing AgentEvents to the bus as it goes.

Deliberately does not import textual (or any UI toolkit) — it can be driven
and tested headlessly with plain asyncio/pytest. It also deliberately does
not surface any model-internal reasoning: only observable actions (what ran,
what it said) become events, per the "no private reasoning in the UI" design
constraint.
"""

from coconut_tui.bus import EventBus
from coconut_tui.events import (
    AgentError,
    AgentFinished,
    AgentInfo,
    AgentStarted,
    AssistantTextDelta,
    FileDiff,
    ToolCompleted,
    ToolOutput,
    ToolStarted,
)
from coconut_tui.providers.base import LLMProvider
from coconut_tui.tools.base import Tool

HELP_TEXT = (
    "Coconut es un modelo base (sin fine-tuning de instrucciones): escribe el "
    "principio de una frase para que la continue, no le hagas preguntas directas "
    "-- funciona mejor con prosa tipo Wikipedia que con conversacion.\n\n"
    "Comandos:\n"
    "  /read <ruta>    lee un fichero del repo\n"
    "  /test [ruta]    ejecuta pytest\n"
    "  /diff <ruta>    muestra el git diff de un fichero\n"
    "  /temp <valor>   cambia la temperature\n"
    "  /tokens <n>     cambia cuantos tokens generar por turno\n"
    "  /help           muestra esta ayuda"
)


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        bus: EventBus,
        tools: dict[str, Tool] | None = None,
    ):
        self.provider = provider
        self.bus = bus
        self.tools = tools or {}
        self.temperature = 0.8
        self.max_new_tokens = 200
        self.top_k: int | None = 50

    async def handle_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._handle_command(text[1:])
        else:
            await self._handle_prompt(text)

    async def _handle_command(self, command_text: str) -> None:
        parts = command_text.split(maxsplit=1)
        name = parts[0]
        argument = parts[1] if len(parts) > 1 else ""

        if name == "help":
            await self.bus.publish(AgentInfo(message=HELP_TEXT))
            return
        if name == "temp":
            await self._set_temperature(argument)
            return
        if name == "tokens":
            await self._set_max_tokens(argument)
            return

        tool = self.tools.get(name)
        if tool is None:
            await self.bus.publish(AgentError(message=f"comando desconocido: /{name}"))
            return
        await self._run_tool(name, tool, argument)

    async def _set_temperature(self, argument: str) -> None:
        try:
            self.temperature = float(argument)
            await self.bus.publish(AgentInfo(message=f"temperature = {self.temperature}"))
        except ValueError:
            await self.bus.publish(AgentError(message="uso: /temp <numero>"))

    async def _set_max_tokens(self, argument: str) -> None:
        try:
            self.max_new_tokens = int(argument)
            await self.bus.publish(AgentInfo(message=f"max_new_tokens = {self.max_new_tokens}"))
        except ValueError:
            await self.bus.publish(AgentError(message="uso: /tokens <entero>"))

    async def _run_tool(self, name: str, tool: Tool, argument: str) -> None:
        description = f"{tool.description}: {argument}" if argument else tool.description
        await self.bus.publish(ToolStarted(tool=name, description=description))
        try:
            output = await tool.run(argument)
        except Exception as exc:
            await self.bus.publish(ToolOutput(tool=name, output=str(exc)))
            await self.bus.publish(ToolCompleted(tool=name, success=False))
            return

        await self.bus.publish(ToolOutput(tool=name, output=output))
        if name == "diff":
            await self.bus.publish(FileDiff(path=argument, diff=output))
        await self.bus.publish(ToolCompleted(tool=name, success=True))

    async def _handle_prompt(self, prompt: str) -> None:
        await self.bus.publish(AgentStarted(prompt=prompt))
        chunks: list[str] = []
        try:
            async for chunk in self.provider.stream(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            ):
                chunks.append(chunk)
                await self.bus.publish(AssistantTextDelta(text=chunk))
        except Exception as exc:
            await self.bus.publish(AgentError(message=str(exc)))
            return
        await self.bus.publish(AgentFinished(full_text="".join(chunks)))
