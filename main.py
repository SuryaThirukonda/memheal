
import datetime as dt
import json
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from memheal.store import MemoryStore
load_dotenv()

client = OpenAI()                  # reads OPENAI_API_KEY from env
MODEL = "gpt-5.4-mini"             # economical model; check your account's model list
MAX_TOOL_ROUNDS = 5                # cap on tool calls per user message

# One session id per process run. Written to every memory this run creates, so
# you can always ask "which conversation taught me this?".
SESSION = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
store = MemoryStore()              # ./memheal.db, created on first run

INSTRUCTIONS = (
    "You are a helpful assistant. Use tools when appropriate.\n"
    "You have a persistent memory that survives restarts. Call remember() when the "
    "user tells you something worth keeping across conversations. Call recall() "
    "before answering questions about the user or about earlier sessions — do not "
    "guess. If a memory is contradicted, call forget() with its id."
)


# Tools ------------------------------------------------

def print_to_terminal(message: str) -> str:
    print(f"[TOOL OUTPUT] {message}")
    return "printed successfully"

def math(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def remember(text: str, source_type: str) -> str:
    mem_id = store.add(text, source_type=source_type, session=SESSION)
    print(f"[MEMORY +] {mem_id}  ({source_type})  {text}")
    return f"stored as {mem_id}"


def recall(query: str) -> str:
    # No session filter: this searches every past run, which is the whole point.
    rows = store.search(query)
    if not rows:
        return "no matching memories"
    return "\n".join(
        f"[{r['id']}] ({r['source_type']}, session {r['session']}) {r['text']}"
        for r in rows
    )


def forget(memory_id: str) -> str:
    # Not a DELETE — the row stays, flagged, so it can still explain itself.
    if store.set_status(memory_id, "retracted"):
        print(f"[MEMORY -] {memory_id} retracted")
        return f"retracted {memory_id}"
    return f"no such memory: {memory_id}"


TOOLS = {"print_to_terminal": print_to_terminal,
        "math": math,
        "remember": remember,
        "recall": recall,
        "forget": forget,
         }

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "print_to_terminal",
        "description": "Print a message to the user's terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to print."}
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        
        "type": "function",
        "name": "math",
        "description": "Perform a basic math operation on two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number."},
                "b": {"type": "number", "description": "Second number."},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform.",
                },
            },
            "required": ["a", "b", "operation"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "remember",
        "description": (
            "Store a fact in persistent memory that survives restarts. Use for things "
            "the user states about themselves, decisions made, or conclusions worth "
            "keeping. Do not store small talk or one-off arithmetic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact, written as a standalone sentence that "
                                   "will still make sense with no conversation around it.",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["user", "tool", "inference"],
                    "description": "'user' if the user stated it, 'tool' if it came "
                                   "from a tool result, 'inference' if you concluded it.",
                },
            },
            "required": ["text", "source_type"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "recall",
        "description": (
            "Search persistent memory across all past sessions. Call this before "
            "answering anything about the user or earlier conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Substring to match against memory text. "
                                   "Pass an empty string to list all memories.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "forget",
        "description": (
            "Retract a memory that is wrong or outdated. The memory is flagged, not "
            "deleted, and stops appearing in recall(). Requires an id you saw in a "
            "recall() or remember() result — never invent one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Memory id, e.g. 'm_4cc49b53'.",
                }
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# ---------------------------------------------------------------------------
# 2. Turn handler

def handle_turn(history: list, action_log: list) -> None:
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            tools=TOOL_SCHEMAS,
            input=history,
        )

        # Keep everything the model produced (messages + function calls)
        history.extend(response.output)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            print(f"[ASSISTANT] {response.output_text}")
            return

        for call in calls:
            args = json.loads(call.arguments)     # arguments arrive as a JSON string
            func = TOOLS.get(call.name)
            result = func(**args) if func else f"unknown tool: {call.name}"

            action_log.append({"tool": call.name, "args": args})
            history.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": str(result),
            })

    print("[ASSISTANT] (stopped: too many tool calls)")


# ---------------------------------------------------------------------------
# 3. Chat loop
# ---------------------------------------------------------------------------

def main() -> None:
    history, action_log = [], []
    print(f"[SESSION] {SESSION}   memories on disk: {store.counts()}")
    try:
        while True:
            user_text = input("\nYou: ").strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            if not user_text:
                continue
            history.append({"role": "user", "content": user_text})
            handle_turn(history, action_log)
    except (KeyboardInterrupt, EOFError):
        print()                       # Ctrl+C / Ctrl+D land here instead of a traceback
    finally:
        # `history` dies with the process; anything remember()ed is already on disk.
        print("\n[ACTION LOG]")
        print(json.dumps(action_log, indent=2))
        print(f"[MEMORY] {store.counts()} in {store.path}")
        store.close()


if __name__ == "__main__":
    main()