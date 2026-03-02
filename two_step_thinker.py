import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_gigachat.chat_models import GigaChat

# Загружаем переменные окружения
load_dotenv()

# ==========================================
# Настройки трассировки LangSmith
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_PROJECT"] = "Two-Step-Thinker-GigaChat"


def run_two_step_thinker(topic: str):
    """
    Демонстрация двухшаговой цепочки на GigaChat:
    Шаг 1: Краткое объяснение концепции.
    Шаг 2: Генерация провокационных вопросов на базе ответа из Шага 1.
    """
    print(f"--- Запускаем «Двухшагового Мыслителя» (GigaChat) для темы: '{topic}' ---\n")
    
    # ПРОВЕРКА КЛЮЧЕЙ
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        print("❌ ОШИБКА: Переменная GIGACHAT_CREDENTIALS не найдена в .env или не загружена!")
        return

    # Печатаем первые 5 символов для проверки (безопасно)
    print(f"ℹ️ Ключ загружен (начинается на: {credentials[:5]}...)")
    
    # Инициализируем GigaChat
    # Мы явно передаем credentials и scope
    llm_explainer = GigaChat(
        credentials=credentials,
        scope="GIGACHAT_API_PERS", # Для физлиц по умолчанию
        model="GigaChat", 
        temperature=0.3, 
        verify_ssl_certs=False
    )
    llm_questioner = GigaChat(
        credentials=credentials,
        scope="GIGACHAT_API_PERS",
        model="GigaChat", 
        temperature=0.7, 
        verify_ssl_certs=False
    )
    
    # ---------------------------------------------------------
    # ШАГ 1: Объяснитель (The Explainer)
    # ---------------------------------------------------------
    prompt1 = PromptTemplate.from_template(
        "Кратко объясни следующую концепцию простыми словами (не более 3-4 предложений):\n{topic}"
    )
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
    chain2 = prompt2 | llm_questioner | StrOutputParser()
    
    # ---------------------------------------------------------
    # ОРКЕСТРАЦИЯ ПОТОКА (С помощью LCEL)
    # ---------------------------------------------------------
    two_step_chain = (
        {"topic": RunnablePassthrough()} 
        | RunnablePassthrough.assign(explanation=chain1)
        | RunnablePassthrough.assign(questions=chain2)
    )
    
    print("🤖 GigaChat думает...\n")
    
    try:
        # Запускаем нашу объединенную цепочку!
        result = two_step_chain.invoke(topic)
        
        # Выводим результаты
        print("🔹 ШАГ 1: Краткое объяснение от GigaChat")
        print("-" * 50)
        print(result["explanation"])
        print("\n")
        
        print("🔹 ШАГ 2: Провокационные вопросы")
        print("-" * 50)
        print(result["questions"])
        print("\n")
        
    except Exception as e:
        print(f"❌ Ошибка вызова GigaChat: {e}")
        print("\nПОДСКАЗКА: Если ключ верный, попробуйте создать новый в кабинете SberDevices.")

if __name__ == "__main__":
    print("--- ДОБРО ПОЖАЛОВАТЬ В «ДВУХШАГОВОГО МЫСЛИТЕЛЯ» ---")
    print("(используется GigaChat от Сбера)\n")
    
    while True:
        user_topic = input("Введите тему для изучения (или 'exit' для выхода): ").strip()
        
        if user_topic.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
            
        if not user_topic:
            continue
            
        run_two_step_thinker(user_topic)
        print("\n" + "="*60 + "\n")
