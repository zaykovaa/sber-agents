"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from langchain_core.tools import tool
import rag

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)

@tool
def calculate_loan_payment(principal: float, annual_rate: float, months: int) -> str:
    """
    Рассчитывает ежемесячный платеж по аннуитетному кредиту.
    
    Args:
        principal: Сумма кредита (в рублях)
        annual_rate: Годовая процентная ставка (в процентах, например 15.5 для 15.5%)
        months: Срок кредита в месяцах
    
    Returns:
        JSON строка с расчетом: ежемесячный платеж, общая сумма выплат, переплата
    """
    try:
        # Преобразуем годовую ставку в месячную (в долях, не процентах)
        monthly_rate = (annual_rate / 100) / 12
        
        if monthly_rate == 0:
            # Если ставка 0%, просто делим на количество месяцев
            monthly_payment = principal / months
        else:
            # Формула аннуитетного платежа: A = P * (r * (1 + r)^n) / ((1 + r)^n - 1)
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        
        total_payment = monthly_payment * months
        overpayment = total_payment - principal
        
        result = {
            "monthly_payment": round(monthly_payment, 2),
            "total_payment": round(total_payment, 2),
            "overpayment": round(overpayment, 2),
            "principal": round(principal, 2),
            "annual_rate": annual_rate,
            "months": months
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_loan_payment: {e}", exc_info=True)
        return json.dumps({"error": f"Ошибка расчета: {str(e)}"}, ensure_ascii=False)

@tool
def calculate_deposit_interest(principal: float, annual_rate: float, days: int, capitalization: bool = True) -> str:
    """
    Рассчитывает доход по вкладу с учетом процентов.
    
    Args:
        principal: Сумма вклада (в рублях)
        annual_rate: Годовая процентная ставка (в процентах, например 7.5 для 7.5%)
        days: Срок вклада в днях
        capitalization: Если True - капитализация процентов (проценты добавляются к сумме), 
                        если False - проценты не капитализируются
    
    Returns:
        JSON строка с расчетом: доход, итоговая сумма
    """
    try:
        # Преобразуем годовую ставку в дневную (в долях)
        daily_rate = (annual_rate / 100) / 365
        
        if capitalization:
            # Формула сложных процентов: A = P * (1 + r)^n
            final_amount = principal * (1 + daily_rate) ** days
        else:
            # Простые проценты: A = P * (1 + r * n)
            final_amount = principal * (1 + daily_rate * days)
        
        income = final_amount - principal
        
        result = {
            "principal": round(principal, 2),
            "income": round(income, 2),
            "final_amount": round(final_amount, 2),
            "annual_rate": annual_rate,
            "days": days,
            "capitalization": capitalization
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_deposit_interest: {e}", exc_info=True)
        return json.dumps({"error": f"Ошибка расчета: {str(e)}"}, ensure_ascii=False)

@tool
def calculate_percentage(amount: float, percentage: float) -> str:
    """
    Рассчитывает процент от суммы (универсальный калькулятор процентов).
    
    Args:
        amount: Исходная сумма (в рублях)
        percentage: Процент (например 15 для 15%)
    
    Returns:
        JSON строка с результатом расчета
    """
    try:
        result_amount = amount * (percentage / 100)
        
        result = {
            "original_amount": round(amount, 2),
            "percentage": percentage,
            "result": round(result_amount, 2),
            "description": f"{percentage}% от {amount} руб."
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_percentage: {e}", exc_info=True)
        return json.dumps({"error": f"Ошибка расчета: {str(e)}"}, ensure_ascii=False)

