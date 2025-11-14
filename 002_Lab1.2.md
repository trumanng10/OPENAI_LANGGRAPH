
## Lab 2: Building a Chatbot with Conversational Memory using LangGraph

---

## Learning Objectives

By the end of this lab, participants will be able to:

1. Install and import the required dependencies for LangGraph and LangChain OpenAI in a notebook environment.
2. Define a richer `ConversationState` structure that includes full conversation history and additional context.
3. Implement a **user input node** that simulates multi-turn user inputs based on the conversation round.
4. Implement a **chatbot node with memory** that:

   * Builds a conversation history string from prior messages
   * Sends this history plus current input to the LLM
   * Updates the message history and context in the state
5. Build a LangGraph with:

   * Nodes for user input and response generation
   * A conditional edge function (`should_continue`) to loop the conversation and stop after a fixed number of rounds.
6. Use `graph.stream()` to observe step-by-step execution of the graph and inspect how memory grows over time.
7. Use `graph.invoke()` to run multiple conversation rounds in a simpler way and analyze the final conversation history.

---

## Step-by-Step Lab Guide (with Explanation)

> Participants should **copy the code as-is** into a notebook, only adjusting the API key part if they want to actually call the OpenAI model.

---

### Step 1 – Install and Import Dependencies

```python
# Step 1: Install and import dependencies
#!pip install langgraph langchain-openai

import os
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

**What to do**

1. Open a new Jupyter/Colab notebook for **Lab 2**.
2. Paste this block into the first cell.
3. If `langgraph` and `langchain-openai` are not installed, **uncomment** the `pip` line and run once.
4. Uncomment the `os.environ["OPENAI_API_KEY"] = "your-api-key-here"` line and insert your own key if you want live responses.

**Explanation**

* `StateGraph, END` are used to build and terminate the graph.
* `TypedDict` and `List` help define a more structured state type.
* `ChatOpenAI` is the LLM interface.
* The API key (if used) allows the code to call the OpenAI model.

---

### Step 2 – Define the State Structure with Conversation History

```python
# Step 2: Define proper state structure with conversation history
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]  # Stores entire conversation history
    user_input: str
    response: str
    conversation_context: str  # Additional context tracking
```

**What to do**

1. Paste this into the next cell and run it.

**Explanation**

* `ConversationState` now explicitly models the structure of the state:

  * `messages`: a list of dictionaries, each with `role` and `content`, e.g. `{"role": "user", "content": "Hello"}`.
  * `user_input`: the current user message for this round.
  * `response`: the latest assistant reply.
  * `conversation_context`: a simple string summarizing which “round” the conversation is in.
* This makes it easier to reason about **multi-turn** interactions and memory.

---

### Step 3 – Create Node Functions with Memory

```python
# Step 3: Create node functions with memory
def user_input_node(state: ConversationState):
    # In real scenario, this would get actual user input
    # For demo, we'll simulate a conversation
    conversation_round = len(state["messages"]) // 2
    user_inputs = [
        "Hello!",
        "Tell me about AI",
        "How does machine learning work?",
        "What are neural networks?",
        "Thanks, that was helpful!"
    ]
    current_input = user_inputs[min(conversation_round, len(user_inputs)-1)]
    return {"user_input": current_input}

def chatbot_with_memory_node(state: ConversationState):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Build conversation history for context
    conversation_history = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in state["messages"]
    ])

    # Create prompt with full conversation history
    prompt = f"""
    Conversation History:
    {conversation_history}

    Current User Input: {state['user_input']}

    Based on the conversation history above, provide a helpful response.
    Maintain context from previous messages and continue the conversation naturally.
    """

    response = llm.invoke(prompt)

    # Update messages with new exchange
    updated_messages = state["messages"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "messages": updated_messages,
        "conversation_context": f"Conversation round {len(updated_messages)//2}"
    }
```

**What to do**

1. Paste this into a new cell and run it.

**Explanation – `user_input_node`**

* `conversation_round = len(state["messages"]) // 2`

  * Each “round” consists of 2 messages (user + assistant).
  * This line calculates which round you’re in based on how many messages have already been exchanged.
* `user_inputs[...]`

  * Predefined list of 5 user utterances to simulate a natural conversation.
* `current_input`

  * Selects the correct line from `user_inputs` depending on the round, and caps at the last item.
* Returns a dict containing `"user_input"`, which becomes part of the next state.

**Explanation – `chatbot_with_memory_node`**

* Builds `conversation_history` by joining all past messages into a readable transcript.
* Prepares a `prompt` that includes:

  * The full conversation history as context
  * The current user input
  * Instructions to maintain context and respond naturally
* `response = llm.invoke(prompt)` sends this to the LLM.
* `updated_messages` appends the new user and assistant messages to the history.
* Returns:

  * `response`: the latest reply text
  * `messages`: the full updated history
  * `conversation_context`: a short descriptor like `"Conversation round 3"`.

This demonstrates how **prompt-based memory** can be implemented by feeding previous messages into each new LLM call.

---

### Step 4 – Build the Graph

```python
# Step 4: Build the graph
builder = StateGraph(ConversationState)
builder.add_node("get_input", user_input_node)
builder.add_node("generate_response", chatbot_with_memory_node)
```

**What to do**

1. Paste into a new cell and run.

**Explanation**

* `StateGraph(ConversationState)` tells LangGraph what the state type looks like.
* `"get_input"` node will handle synthetic user input.
* `"generate_response"` node will handle LLM calls and memory updates.

At this point, the nodes exist, but the flow between them is not defined yet.

---

### Step 5 – Define Edges and Conversation Loop

```python
# Step 5: Define edges with conversation loop
def should_continue(state: ConversationState):
    # Continue conversation for 5 rounds, then end
    if len(state["messages"]) >= 10:  # 5 rounds (user + assistant = 2 messages per round)
        return "end"
    return "continue"

builder.set_entry_point("get_input")
builder.add_edge("get_input", "generate_response")
builder.add_conditional_edges(
    "generate_response",
    should_continue,
    {
        "continue": "get_input",
        "end": END
    }
)
```

**What to do**

1. Paste into a new cell and run.

**Explanation**

* `should_continue(state)`:

  * Checks the length of `state["messages"]`.
  * Each round adds 2 messages (user + assistant).
  * When there are 10 messages, that means **5 rounds** have occurred → return `"end"`.
  * Otherwise, return `"continue"`.
* `set_entry_point("get_input")` – execution starts with the user input node.
* `add_edge("get_input", "generate_response")` – after user input, go to the chatbot node.
* `add_conditional_edges("generate_response", should_continue, {...})`:

  * After each chatbot response, evaluate `should_continue`.
  * If `"continue"` → loop back to `"get_input"`.
  * If `"end"` → go to `END` and stop execution.

This creates a **conversation loop** inside the graph, controlled by the state.

---

### Step 6 – Compile and Run with Memory

```python
# Step 6: Compile and run with proper memory
graph = builder.compile()

# Initialize with empty conversation
initial_state = {
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ],
    "user_input": "",
    "response": "",
    "conversation_context": "start"
}

print("=== Chatbot with Conversational Memory ===\n")

# FIXED: Proper way to handle stream output
current_round = 0
for step in graph.stream(initial_state):
    for node_name, node_output in step.items():
        if node_name == "get_input":
            current_user_input = node_output.get("user_input", "")
            print(f"Round {current_round + 1}:")
            print(f"User: {current_user_input}")

        elif node_name == "generate_response":
            current_response = node_output.get("response", "")
            message_count = len(node_output.get("messages", []))
            print(f"Assistant: {current_response}")
            print(f"Memory: {message_count} messages in history")
            print("-" * 50)
            current_round += 1
```

**What to do**

1. Paste this block into a new cell.
2. Run it.

**What you should see**

* The header:
  `=== Chatbot with Conversational Memory ===`
* For each round (up to 5):

  * A **User** line with the simulated input.
  * An **Assistant** line with the model’s response.
  * A **Memory** line showing how many messages are in the conversation history.
  * A separator line of `-` characters.

**Explanation**

* `graph = builder.compile()` turns your builder definition into an executable graph.
* `initial_state` starts with:

  * One system message
  * Empty strings for `user_input` and `response`
  * Context set to `"start"`.
* `graph.stream(initial_state)` yields each step of execution:

  * You see when `"get_input"` runs and what `user_input` is chosen.
  * You see when `"generate_response"` runs and how the memory size increases.
* This is very useful for **debugging and teaching** the flow of a multi-step, multi-round graph.

---

### Step 7 – Alternative: Using `invoke()` for Simpler Debugging

```python
# Alternative approach using invoke for simpler debugging
print("\n=== Testing with invoke() method ===\n")
test_state = {
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ],
    "user_input": "",
    "response": "",
    "conversation_context": "start"
}

# Run multiple conversation rounds
for i in range(3):
    result = graph.invoke(test_state)
    print(f"Round {i+1}:")
    print(f"User: {result['user_input']}")
    print(f"Assistant: {result['response']}")
    print(f"Memory: {len(result['messages'])} messages in history")
    print("-" * 50)

    # Update state for next round
    test_state = result
```

**What to do**

1. Paste after the previous block in the same cell or a new cell, and run.

**What you should see**

* A section header: `=== Testing with invoke() method ===`
* Three rounds printed:

  * Each shows `User: ...`, `Assistant: ...`, and current `Memory: ... messages in history`.

**Explanation**

* Instead of streaming each node step, `graph.invoke(test_state)`:

  * Runs the graph from start to end once.
  * Returns the final state after that run.
* By updating `test_state = result` after each iteration, the next `invoke()` call continues from where the previous one left off.
* This is a quick way to **simulate continuing a conversation** across multiple invocations.

---

### Step 8 – Inspect Final Conversation History

```python
# Show final conversation history
print("\n=== Final Conversation History ===")
for i, msg in enumerate(test_state["messages"]):
    print(f"{i+1}. {msg['role']}: {msg['content']}")
```

**What to do**

1. Paste this at the end and run.

**What you should see**

* A numbered list of all messages in `test_state["messages"]`:

  * System message
  * Each user input
  * Each assistant response

**Explanation**

* This gives a **full transcript** of the conversation that the model has seen and produced.
* It’s useful for:

  * Verifying that context is being preserved.
  * Debugging any unexpected behavior.
  * Showing students how a chat log is stored in `messages`.

