import asyncio
from openjudge.graders.agent import TrajectoryComprehensiveGrader
from openjudge.models import OpenAIChatModel


async def main():
    # 初始化模型和轨迹综合性评估评分器
    model = OpenAIChatModel(model="qwen3.7-max")
    grader = TrajectoryComprehensiveGrader(model=model)

    # 数据准备
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Find Python files modified today"},
        {
            "role": "assistant",
            "content": "I'll search for Python files.",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "search_files",
                        "arguments": '{"pattern": "*.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Found: main.py, utils.py"},
        {
            "role": "assistant",
            "content": "I'll check their modification dates.",
            "tool_calls": [
                {
                    "id": "2",
                    "function": {
                        "name": "get_file_info",
                        "arguments": '{"files": ["main.py", "utils.py"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "content": "main.py: today, utils.py: yesterday",
        },
        {"role": "assistant", "content": "Found 1 file modified today: main.py"},
    ]

    # 运行评估
    result = await grader.aevaluate(messages=messages)
    print("评估结果：")
    print(f"Overall Score: {result.score}")
    print(f"Is Resolved: {result.metadata['is_resolved']}")
    print(f"Avg Contribution: {result.metadata['avg_contribution']}")
    print(f"Avg Relevance: {result.metadata['avg_relevance']}")
    print(f"Avg Accuracy: {result.metadata['avg_accuracy']}")
    print(f"Avg Efficiency: {result.metadata['avg_efficiency']}")
    # 每步细节
    for step in result.metadata["step_evaluations"]:
        print(f"Step {step['step_index']}: {step['step_reason']}")

    print(f"Reason: {result.reason}")


asyncio.run(main())

note = """
    TrajectoryComprehensiveGrader
    对完整agent轨迹进行全面评估
    使用它用来：
        端对端agent评估
        综合轨迹评估
        agent基准评估
        生产质量监控
    评估标准：在完整轨迹中对步骤贡献、相关性、准确性和效率的综合考量
    参数：
        message                     List[Dict[str,Any]]     required    完整消息历史（包含system、user、assistant及tool）
        resolution_threshold        float                   optional    视为成功决策的阈值（默认0.8）
    得分说明：每个维度使用1-5范围评分，归一化到0-1
        5->1.0：Excellent
        4->0.75：Good
        3->0.5：Acceptable
        2->0.25：Poor
        1->0.0：Very poor
    总体得分是所有步骤和维度的平均值。
"""
