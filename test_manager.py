from context_manager import ContextManager

# Demo summariser function (replace with an actual LLM API call in production)
def mock_summariser(prompt):
    return "User and assistant discussed Python programming and token limits previously."

# Initialize the manager
manager = ContextManager(window_limit=8000)

# Create a long conversation history
long_history = [
    {"role": "system", "content": "You are an expert Python AI Assistant."},
    {"role": "user", "content": "Hello! Can you help me learn Python?"},
    {"role": "assistant", "content": "Yes, absolutely! What do you want to start with?"},
    {"role": "user", "content": "Let's talk about functions."},
    {"role": "assistant", "content": "Functions are reusable blocks of code... [details]"},
    {"role": "user", "content": "Now explain token limits in LLMs."},
    {"role": "assistant", "content": "Token limits restrict how much text an LLM can process at once."},
    {"role": "user", "content": "How do we handle very long text using sliding windows?"},
    {"role": "assistant", "content": "Sliding windows keep recent messages within budget."},
    {"role": "user", "content": "Can we compress old messages?"},
    {"role": "assistant", "content": "Yes, rolling summaries help compress old turns into a compact note."},
    {"role": "user", "content": "Awesome! Give me a quick code example for this."}
]

# Demo retrieved data (RAG)
sample_rag_doc = "Python 3.11 introduced performance boosts and finer exception handling. " * 100

print("--- Starting Context Manager Test ---\n")

# Build the final context
final_context, total_tokens = manager.build_final_context(
    history=long_history, 
    retrieved_doc=sample_rag_doc, 
    summariser_func=mock_summariser
)

print(f"Total messages in final context: {len(final_context)}")
print(f"Total token cost (including overhead): {total_tokens} tokens\n")

print("--- Final Context Structure ---")
for i, msg in enumerate(final_context):
    print(f"[{i}] Role: {msg['role']} | Tokens: {manager.ntok(msg['content'])}")
    print(f"    Content preview: {msg['content'][:80]}...\n")

# Budget check assertion
assert total_tokens <= manager.window_limit, "⚠️ Warning: Token budget exceeded the window limit!"
print("✅ Test passed successfully! Context managed within budget limits.")