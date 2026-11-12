import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.agent import ToolSelectionGrader


async def main():
    # 初始化模型和工具选择评分器
    model = OpenAIChatModel(model="qwen3.7-max")
    grader = ToolSelectionGrader(model=model)

    # 数据准备
    data = {
        "query": "Find all Python files modified in the last week",
        "tool_definitions": [
            {"name": "search_files", "description": "Search for files by pattern"},
            {"name": "git_log", "description": "Get git commit history"},
            {"name": "read_file", "description": "Read file contents"},
        ],
        "tool_calls": [
            {"name": "search_files", "arguments": '{"pattern": "*.py"}'},
            {"name": "git_log", "arguments": {"days": 7}},
        ],
    }

    # 运行评估
    print("评估结果：")
    result = await grader.aevaluate(**data)
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")


asyncio.run(main())

note = """
    ToolSelectionGrader
    评估解决用户查询时工具选择的质量
    使用它用来：
        评估工具选择的适当性
        评估agent决策质量
        比较不同agent架构
    评估标准：工具相关性、选择完整性、效率和对工具能力的理解
    参数：
        query                   str or List[Dict]       required    用户问题或对话历史
        tool_definitions        List[Dict[str,Any]]     required    可用工具定义
        tool_calls              List[Dict[str,Any]]     required    agent实际选择的工具
    得分说明：
        5：最佳工具选择，最直接有效
        4：合理选择，能完成任务但不是最佳选择
        3：可接受，相关但不直接匹配
        2：差劲，明显不匹配任务
        1：不正确，没有选择工具或者完全不相关
"""
