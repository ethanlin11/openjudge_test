import asyncio
from openjudge.graders.agent import ToolSelectionGrader
from openjudge.models import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner, GraderConfig


async def main():
    # 初始化模型和工具选择评分器
    model = OpenAIChatModel(model="qwen3.7-max")
    grader = ToolSelectionGrader(model=model)

    # 定义映射器：从单一agent轨迹（数据集单一用例）提取评分器输入；返回的是ToolSelectionGrader评估所需的参数
    def extract_tool_inputs(data: dict) -> dict:
        messages = data["messages"]
        query = next((m["content"] for m in messages if m["role"] == "user"), "")
        tool_calls = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    )
        return {
            "query": query,
            "tool_definitions": data["available_tools"],
            "tool_calls": tool_calls,
        }

    # 实例化GradingRunner，指定映射器mapper
    runner = GradingRunner(
        grader_configs={
            "tool_selection": GraderConfig(grader=grader, mapper=extract_tool_inputs)
        },
        max_concurrency=16,  # 最大并发：16
        show_progress=True,  # 过程展示
    )

    # 数据集准备(agent轨迹)
    dataset = [
        {  # Bad case: should use calculator, not search_web
            "messages": [
                {"role": "user", "content": "What's 15% tip on a $45 bill?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": '{"query": "15% tip on $45"}',
                            }
                        }
                    ],
                },
            ],
            "available_tools": [
                {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                },
                {"name": "search_web", "description": "Search the web for information"},
            ],
        },
        {  # Good case: correctly uses get_weather
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            }
                        }
                    ],
                },
            ],
            "available_tools": [
                {"name": "get_weather", "description": "Get weather information"},
                {"name": "search_web", "description": "Search the web for information"},
            ],
        },
    ]

    # 执行批量评估
    results = await runner.arun(dataset)

    # 结果输出
    for i, result in enumerate(results["tool_selection"]):
        print(f"Trace {i}: Score={result.score}")


asyncio.run(main())
