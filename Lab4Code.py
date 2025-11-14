# Step 1: Install and import dependencies
#!pip install langgraph langchain-openai

import os
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, List
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Step 2: Enhanced Support State
class SupportState(TypedDict):
    user_query: str
    intent: str
    department: str
    response: str
    escalated: bool
    conversation_history: List[Dict[str, str]]
    needs_human: bool
    resolution_status: str
    should_continue: bool  # New field to control conversation flow

# Step 3: Interactive Input Function
def get_user_input():
    """Get real-time input from user"""
    print("\n" + "="*50)
    print("💬 CUSTOMER SUPPORT CHAT")
    print("="*50)
    print("Type your issue below (or type 'exit' to quit, 'help' for options):")

    user_input = input("You: ").strip()

    if user_input.lower() == 'exit':
        return None, True  # Signal to end
    elif user_input.lower() == 'help':
        print("\n📋 Common issues you can ask about:")
        print("  • Billing: 'I was charged twice', 'Where is my refund?'")
        print("  • Technical: 'App keeps crashing', 'Can't login', 'My laptop won't start'")
        print("  • Sales: 'I want to upgrade', 'Pricing plans'")
        print("  • General: 'Contact information', 'Business hours'")
        return "help", False
    elif user_input.lower() in ['thanks', 'thank you', 'that\'s all', 'no more questions']:
        return user_input, True  # Signal to end conversation

    return user_input, False

# Step 4: Enhanced Intent Classification
def classify_intent(state: SupportState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Build conversation context
    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in state["conversation_history"][-4:]
        ])

    classification = llm.invoke(f"""
    Analyze this customer support query and classify it into the most appropriate category.

    {conversation_context}

    Current Query: {state['user_query']}

    Categories:
    - billing: Payment issues, refunds, charges, invoices, billing errors
    - technical: App problems, login issues, bugs, errors, technical difficulties, hardware issues, software problems
    - sales: Upgrades, pricing, plans, purchases, subscriptions, product information
    - general: Everything else - contact info, hours, general questions, account issues

    Return ONLY the category name (billing, technical, sales, or general).
    """)

    intent = classification.content.strip().lower()

    # Validate intent
    valid_intents = ["billing", "technical", "sales", "general"]
    if intent not in valid_intents:
        intent = "general"

    return {"intent": intent}

# Step 5: Department Routing
def route_to_department(state: SupportState):
    intent = state["intent"]

    # Check for escalation keywords
    escalation_keywords = ["manager", "supervisor", "escalate", "complaint", "angry", "frustrated", "human", "representative"]
    user_query_lower = state["user_query"].lower()

    if any(keyword in user_query_lower for keyword in escalation_keywords):
        return "human_agent"

    departments = {
        "billing": "billing_support",
        "technical": "tech_support",
        "sales": "sales_support",
        "general": "general_support"
    }

    return departments.get(intent, "general_support")

# Step 6: Department Handlers with Better Responses
def billing_support(state: SupportState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in state["conversation_history"][-4:]
        ])

    prompt = f"""
    You are a friendly billing support agent.

    {conversation_context}

    Customer's current issue: "{state['user_query']}"

    Provide a helpful, empathetic response focused on:
    - Payment issues and disputes
    - Refund requests and processing
    - Billing questions and invoices
    - Subscription charges and cancellations

    Be professional but warm. Offer specific help and ask clarifying questions if needed.
    Keep response conversational and under 3 sentences.
    """

    response = llm.invoke(prompt)

    updated_history = state.get("conversation_history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "department": "billing",
        "escalated": False,
        "conversation_history": updated_history,
        "resolution_status": "pending",
        "should_continue": True  # Always continue after response
    }

def tech_support(state: SupportState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in state["conversation_history"][-4:]
        ])

    prompt = f"""
    You are a technical support specialist.

    {conversation_context}

    Customer's technical issue: "{state['user_query']}"

    Provide helpful technical assistance for:
    - Software problems and errors
    - Hardware issues (computers, laptops, devices)
    - Login and account access problems
    - App crashes and performance issues

    Be patient and clear. Suggest troubleshooting steps and ask for specific details.
    Keep response conversational and under 3 sentences.
    """

    response = llm.invoke(prompt)

    updated_history = state.get("conversation_history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "department": "technical",
        "escalated": False,
        "conversation_history": updated_history,
        "resolution_status": "pending",
        "should_continue": True
    }

def sales_support(state: SupportState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in state["conversation_history"][-4:]
        ])

    prompt = f"""
    You are a sales support agent.

    {conversation_context}

    Customer's inquiry: "{state['user_query']}"

    Help with:
    - Product information and features
    - Pricing plans and options
    - Upgrades and purchases
    - Subscription details and benefits

    Be enthusiastic and helpful. Provide clear information about options.
    Keep response conversational and under 3 sentences.
    """

    response = llm.invoke(prompt)

    updated_history = state.get("conversation_history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "department": "sales",
        "escalated": False,
        "conversation_history": updated_history,
        "resolution_status": "pending",
        "should_continue": True
    }

def general_support(state: SupportState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in state["conversation_history"][-4:]
        ])

    prompt = f"""
    You are a general customer support agent.

    {conversation_context}

    Customer's question: "{state['user_query']}"

    Provide helpful assistance for:
    - General questions and information
    - Contact details and business hours
    - Account management issues
    - Company policies and procedures

    Be warm and professional. Redirect to specific departments if needed.
    Keep response conversational and under 3 sentences.
    """

    response = llm.invoke(prompt)

    updated_history = state.get("conversation_history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "department": "general",
        "escalated": False,
        "conversation_history": updated_history,
        "resolution_status": "pending",
        "should_continue": True
    }

def human_agent(state: SupportState):
    """Handle escalated cases"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    prompt = f"""
    The customer's issue requires human agent escalation.

    Customer's concern: "{state['user_query']}"
    Detected Intent: {state['intent']}

    Provide a polite response that:
    1. Acknowledges their concern is important
    2. Explains they're being connected to a human specialist
    3. Assures quick resolution
    4. Provides estimated wait time (2-3 minutes)

    Be empathetic and professional. Keep it to 2 sentences.
    """

    response = llm.invoke(prompt)

    updated_history = state.get("conversation_history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "response": response.content,
        "department": "human_agent",
        "escalated": True,
        "conversation_history": updated_history,
        "resolution_status": "escalated",
        "needs_human": True,
        "should_continue": False  # End conversation after escalation
    }

# Step 7: Build SIMPLE Graph without Recursion
def should_continue(state: SupportState):
    """Determine if conversation should continue"""
    # Check for end indicators in user query
    end_indicators = ["thanks", "thank you", "that's all", "no more", "exit", "bye", "done"]
    user_query_lower = state.get("user_query", "").lower()

    if any(indicator in user_query_lower for indicator in end_indicators):
        return "end_conversation"

    # Use the should_continue flag from department nodes
    if state.get("should_continue", True):
        return "continue"
    else:
        return "end_conversation"

def end_conversation(state: SupportState):
    """Handle conversation ending"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    prompt = """
    The customer conversation is ending. Provide a friendly closing message.
    Thank them for contacting support and offer help in the future.
    Keep it warm and professional in 1-2 sentences.
    """

    closing_message = llm.invoke(prompt)

    return {
        "response": closing_message.content,
        "resolution_status": "resolved",
        "should_continue": False
    }

# Build the graph WITHOUT recursive follow-up
builder = StateGraph(SupportState)

# Add nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("billing_support", billing_support)
builder.add_node("tech_support", tech_support)
builder.add_node("sales_support", sales_support)
builder.add_node("general_support", general_support)
builder.add_node("human_agent", human_agent)
builder.add_node("end_conversation", end_conversation)

# Set up the workflow - SIMPLE version
builder.set_entry_point("classify_intent")

# Route from classification to appropriate department
builder.add_conditional_edges(
    "classify_intent",
    route_to_department,
    {
        "billing_support": "billing_support",
        "tech_support": "tech_support",
        "sales_support": "sales_support",
        "general_support": "general_support",
        "human_agent": "human_agent"
    }
)

# After each department, check if we should continue
builder.add_conditional_edges(
    "billing_support", should_continue, {"continue": END, "end_conversation": "end_conversation"}
)
builder.add_conditional_edges(
    "tech_support", should_continue, {"continue": END, "end_conversation": "end_conversation"}
)
builder.add_conditional_edges(
    "sales_support", should_continue, {"continue": END, "end_conversation": "end_conversation"}
)
builder.add_conditional_edges(
    "general_support", should_continue, {"continue": END, "end_conversation": "end_conversation"}
)
builder.add_conditional_edges(
    "human_agent", should_continue, {"continue": END, "end_conversation": "end_conversation"}
)

# End conversation stops the graph
builder.add_edge("end_conversation", END)

# Compile the graph with recursion limit
graph = builder.compile()

# Step 8: Fixed Interactive Chat Interface
def run_interactive_support():
    """Main function to run the interactive support system"""
    print("🚀 AI CUSTOMER SUPPORT SYSTEM INITIALIZED")
    print("✨ Type your issues and chat naturally with the support agent!")
    print("💡 Type 'help' for suggestions, 'exit' to quit\n")

    # Initial state
    state = {
        "user_query": "",
        "intent": "",
        "department": "",
        "response": "",
        "escalated": False,
        "conversation_history": [],
        "needs_human": False,
        "resolution_status": "pending",
        "should_continue": True
    }

    conversation_count = 0

    while True:
        # Get user input
        user_input, should_end = get_user_input()

        if user_input is None or should_end:
            if state.get("conversation_history"):
                print(f"\n🤖 Support: Thank you for contacting support! Have a great day!")
            print("\n👋 Thank you for using our support system. Goodbye!")
            break

        # Update state with new query
        state["user_query"] = user_input

        try:
            # Process through LangGraph for ONE conversation turn
            result = graph.invoke(state)

            # Display response
            department_display = result['department'].replace('_', ' ').title()
            print(f"\n🤖 {department_display}: {result['response']}")

            # Update state for next iteration
            state = result
            conversation_count += 1

            # Check if we should end based on the response
            if not result.get("should_continue", True) or result.get("resolution_status") == "resolved":
                print(f"\n🎉 Conversation completed! Thank you for using our support system.")
                break

            # Safety limit
            if conversation_count >= 10:
                print(f"\n⚠️  Maximum conversation length reached. Thank you for chatting!")
                break

        except Exception as e:
            print(f"\n❌ Error processing your request: {str(e)}")
            print("Please try rephrasing your question.")
            # Reset to avoid state corruption
            state["user_query"] = ""

# Step 9: Simple Test Function
def test_support_system():
    """Test the support system with sample queries"""
    test_queries = [
        "My laptop cannot start Windows",
        "I was charged twice this month",
        "My app keeps crashing when I try to login",
        "I want to upgrade to the premium plan",
        "I need to speak to a manager immediately!"
    ]

    print("🧪 TESTING SUPPORT SYSTEM\n")

    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: '{query}'")
        state = {
            "user_query": query,
            "intent": "",
            "department": "",
            "response": "",
            "escalated": False,
            "conversation_history": [],
            "needs_human": False,
            "resolution_status": "pending",
            "should_continue": True
        }

        try:
            result = graph.invoke(state)
            print(f"   Department: {result['department']}")
            print(f"   Response: {result['response']}")
            print(f"   Escalated: {result['escalated']}\n")
        except Exception as e:
            print(f"   Error: {str(e)}\n")

# Step 10: Run the system
print("🚀 Enhanced Interactive Support System Ready!")
print("\nChoose an option:")
print("1. run_interactive_support() - Chat with the support agent")
print("2. test_support_system() - Run automated tests")

# Uncomment to run tests
# test_support_system()

# Uncomment to run interactive system
# run_interactive_support()
