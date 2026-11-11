import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader


async def main():
    # 初始化模型和评分器
    model = OpenAIChatModel(model="qwen3.7-max")
    grader = RelevanceGrader(model=model)

    # 数据准备
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. It uses algorithms to build models that can make predictions or decisions.",
    }

    # 运行评估
    result = await grader.aevaluate(**data)
    print(result)


asyncio.run(main())
