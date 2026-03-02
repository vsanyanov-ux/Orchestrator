import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Если вы используете Open AI, раскомментируйте строку ниже и замените ChatGoogleGenerativeAI:
#from langchain_openai import ChatOpenAI


# Загружаем переменные окружения (OPENAI_API_KEY, LANGCHAIN_API_KEY и т.д.)
load_dotenv()

# ==========================================
# PRO TIP: Раскомментируйте или добавьте в .env файл 
# для визуализации "мыслей" и шагов в LangSmith:
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # Берем ключ из .env файла
os.environ["LANGCHAIN_PROJECT"] = "Two-Step-Thinker-Project"


def run_two_step_thinker(topic: str):
    """
    Демонстрация двухшаговой цепочки:
    Шаг 1: Краткое объяснение концепции.
    Шаг 2: Генерация провокационных вопросов на базе ответа из Шага 1.
    """
    print(f"--- Запускаем «Двухшагового Мыслителя» для темы: '{topic}' ---\n")
    
    # Инициализируем LLM (можно использовать разные модели или разную температуру)
    # Для фактов (объяснение) ставим температуру пониже, для креатива (вопросы) - повыше.
    llm_explainer = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    llm_questioner = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # ---------------------------------------------------------
    # ШАГ 1: Объяснитель (The Explainer)
    # ---------------------------------------------------------
    prompt1 = PromptTemplate.from_template(
        "Кратко объясни следующую концепцию простыми словами (не более 3-4 предложений):\n{topic}"
    )
    
    # Первая цепочка: промпт -> LLM -> парсер в строку
    chain1 = prompt1 | llm_explainer | StrOutputParser()
    
    # ---------------------------------------------------------
    # ШАГ 2: Вопрошающий (The Questioner)
    # ---------------------------------------------------------
    prompt2 = PromptTemplate.from_template(
        "В качестве контекста дано следующее краткое объяснение концепции:\n\n"
        "{explanation}\n\n"
        "Основываясь ТОЛЬКО на этом контексте, сгенерируй 3 интересных, "
        "глубоких и провокационных вопроса для дальнейшего изучения этой темы. "
        "Вопросы должны бросать вызов обыденному пониманию."
    )
    
    # Вторая цепочка
    chain2 = prompt2 | llm_questioner | StrOutputParser()
    
    # ---------------------------------------------------------
    # ОРКЕСТРАЦИЯ ПОТОКА (С помощью LCEL)
    # ---------------------------------------------------------
    # Мы хотим передать 'topic' в первую цепочку, получить 'explanation',
    # а затем передать 'explanation' во вторую цепочку. 
    # 
    # Используя RunnablePassthrough.assign, мы можем "протаскивать" 
    # промежуточные результаты, чтобы потом вывести их все вместе:
    
    two_step_chain = (
        # 0. На входе ожидаем {"topic": "..."}
        {"topic": RunnablePassthrough()} 
        # 1. Записываем результат работы chain1 в ключ "explanation"
        | RunnablePassthrough.assign(explanation=chain1)
        # 2. Вызываем chain2 (ему уже доступен "explanation" из предыдущего шага) и записываем в "questions"
        | RunnablePassthrough.assign(questions=chain2)
    )
    
    print("🤖 LLM думает (это займет пару секунд)...\n")
    
    # Запускаем нашу объединенную цепочку!
    result = two_step_chain.invoke(topic)
    
    # Выводим результаты
    print("🔹 ШАГ 1: Краткое объяснение")
    print("-" * 50)
    print(result["explanation"])
    print("\n")
    
    print("🔹 ШАГ 2: Провокационные вопросы")
    print("-" * 50)
    print(result["questions"])
    print("\n")
    
    print("✅ Готово! Если LangSmith включен (LANGCHAIN_TRACING_V2=true), "
          "перейдите в панель управления, чтобы увидеть графическое дерево выполнения и профилирование!")

if __name__ == "__main__":
    # Тестируем концепцию
    run_two_step_thinker("Теория относительности")
