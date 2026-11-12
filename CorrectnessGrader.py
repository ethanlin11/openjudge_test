import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common import CorrectnessGrader


async def main():
    # 初始化模型和正确性评分器
    model = OpenAIChatModel(model="qwen3.7-max")
    grader = CorrectnessGrader(model=model)

    # 数据准备-完美适配真相的用例
    dataset_true = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "reference_response": "Paris",
    }

    # 数据准备-完全忽略或与事实真相相矛盾的用例
    dataset_false = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Lyon.",
        "reference_response": "Paris",
    }

    # 运行评估
    print("评估用例1：")
    result = await grader.aevaluate(**dataset_true)
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

    print("评估用例2：")
    result = await grader.aevaluate(**dataset_false)
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")


asyncio.run(main())

note = """
    CorrectnessGrader
    评估AI响应的答案是否符合提供的事实真相答案。检查事实一致性、信息覆盖度以及对齐情况。
    使用时机：
        基于知识问答的评估
        考试/测试回答评分
        与标准答案的比较
        教育内容评估
    参数：
        query                   str     required    问题
        response                str     required    模型回答
        reference_response      str     required    正确/参考答案
        context                 str     optional    额外的上下文
"""
