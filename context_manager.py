import tiktoken

class ContextManager:
    def __init__(self, model_name="cl100k_base", window_limit=8000, budget_config=None):
        self.enc = tiktoken.get_encoding(model_name)
        self.window_limit = window_limit
        
        # বাজেট অ্যালোকেশন (ডিফল্ট বাজেট)
        self.budget = budget_config or {
            "system": 600,
            "rag": 3000,
            "history": 3000,
            "reply": 1400
        }
        assert sum(self.budget.values()) <= self.window_limit, "Budget exceeds window limit!"

    def ntok(self, text: str) -> int:
        """টোকেন গণনা করার হেল্পার ফাংশন"""
        return len(self.enc.encode(text))

    def clip_text(self, text: str, cap: int) -> str:
        """নির্দিষ্ট বাজেটের বেশি হলে টেক্সট কেটে ছোট করা (Hard cap)"""
        ids = self.enc.encode(text)
        return self.enc.decode(ids[:cap])

    def fit_sliding_window(self, messages, budget_limit):
        """সর্বশেষ মেসেজগুলো থেকে বাজেট অনুযায়ী উইন্ডো ফিট করা"""
        if not messages:
            return [], 0
            
        system = messages[0] if messages[0]["role"] == "system" else None
        rest = messages[1:] if system else messages
        
        used = self.ntok(system["content"]) + 4 if system else 0
        kept = []
        
        for m in reversed(rest):
            cost = self.ntok(m["content"]) + 4
            if used + cost > budget_limit:
                break
            kept.append(m)
            used += cost
            
        window = [system] if system else []
        window.extend(list(reversed(kept)))
        return window, used

    def compress_old_turns(self, old_turns, summariser_func, cap=400):
        """পুরনো কথোপকথন বা টার্নগুলোকে রোলিং সামারিতে রূপান্তর করা"""
        if not old_turns:
            return None
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old_turns)
        summary = summariser_func(f"Summarise for context, <{cap} tokens:\n{text}")
        return {"role": "system", "content": f"[earlier summary] {summary}"}

    def build_final_context(self, history, retrieved_doc="", summariser_func=None):
        """সবকিছু একত্রিত করে ফাইনাল কনটেক্সট তৈরি করা"""
        # ১. RAG টেক্সট ক্লিপ করা
        rag_text = self.clip_text(retrieved_doc, self.budget["rag"]) if retrieved_doc else ""
        
        # ২. সিস্টেম প্রম্পট ঠিক রাখা
        system_prompt = history[0] if history and history[0]["role"] == "system" else {"role": "system", "content": "You are a helpful assistant."}
        
        # ৩. কনভার্সেশন হিস্ট্রি হ্যান্ডলিং (স্লাইডিং উইন্ডো + কম্প্রেশন)
        conversations = history[1:] if history and history[0]["role"] == "system" else history
        
        # যদি হিস্ট্রি বড় হয়, তবে পুরনো অংশ সামারি করব এবং নতুন অংশগুলো অক্ষুণ্ণ রাখব
        if len(conversations) > 6 and summariser_func:
            recent_turns = conversations[-6:]
            archive = conversations[:-6]
            summary_msg = self.compress_old_turns(archive, summariser_func, cap=400)
            
            # উইন্ডো তৈরি
            active_history, _ = self.fit_sliding_window(recent_turns, self.budget["history"])
            final_messages = [system_prompt]
            if summary_msg:
                final_messages.append(summary_msg)
            final_messages.extend(active_history)
        else:
            final_messages, _ = self.fit_sliding_window(history, self.budget["history"])

        # ৪. যদি RAG বা ডুকুমেন্ট থাকে তা সিস্টেম প্রম্পট বা কনটেক্সটে যুক্ত করা
        if rag_text:
            final_messages.insert(1, {"role": "system", "content": f"[Retrieved Context]: {rag_text}"})

        # টোকেন খরচ চেক করা
        total_tokens = sum(self.ntok(m["content"]) + 4 for m in final_messages) + 3
        return final_messages, total_tokens