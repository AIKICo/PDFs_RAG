from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class MemoryBasedQA:
    def __init__(self, model_name: str = "gemma3"):
        self.model_name = model_name

        self.memory = ConversationSummaryMemory(
            llm=OllamaLLM(model=model_name, base_url="http://localhost:11434"),
            memory_key="history",
            return_messages=True
        )

        self.prompt = PromptTemplate.from_template("""
        شما یک دستیار هوش مصنوعی هستید که تنها بر اساس اطلاعات زمینه پاسخ می‌دهد.  
        لطفاً هیچ دانش خارجی یا فرضیات خود را اضافه نکنید.

        🔹 **تاریخچه مکالمه:**  
        ---------------------  
        {history}  
        ---------------------  

        🔹 **پرسش جدید:**  
        {question}  

        🔹 **پاسخ دقیق و مستند:**  
        """)

    def ask(self, question: str) -> str:
        context = self.memory.load_memory_variables({}).get("history", "")

        llm = OllamaLLM(model=self.model_name, base_url="http://localhost:11434", temperature=0.1)
        response = llm.invoke(self.prompt.format(history=context, question=question))

        self.memory.save_context({"input": question}, {"output": response})

        return response

    def clear_memory(self):
        """پاک کردن حافظه مکالمه‌ای"""
        self.memory.clear()
