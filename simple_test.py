from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage
from core.nodes import student_learning_tutor_node, project_coach_node, competition_advisor_node
from core.state import AgentState

async def test_student_learning_mode():
    """测试用例1：学生学习模式"""
    print("=== 测试用例1：学生学习模式 ===")
    
    # 构建初始状态
    state = AgentState(
        messages=[
            HumanMessage(content="我是大一新生，想参加创新创业大赛，但是不知道怎么做，能给我一些建议吗？")
        ],
        user_role="student",
        active_agent="",
        next_agent="",
        context_data={
            "portal_role": "student",
            "current_stage": "核心价值探测",
            "student_profile": {
                "grade": "大一",
                "major": "计算机科学",
                "interests": ["人工智能", "创业"]
            }
        },
        next_step=""
    )
    
    # 直接调用学习辅导节点
    result = await student_learning_tutor_node(state)
    
    # 输出结果
    print("学生输入:", state["messages"][0].content)
    print("AI回复:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
    print("\n")

async def test_competition_coach_mode():
    """测试用例2：竞赛教练模式"""
    print("=== 测试用例2：竞赛教练模式 ===")
    
    # 构建初始状态
    state = AgentState(
        messages=[
            HumanMessage(content="我想做一个智能农业项目，目标是帮助农民提高产量。现在需要撰写商业计划书，能和我一起协作完成吗？")
        ],
        user_role="student",
        active_agent="",
        next_agent="",
        context_data={
            "portal_role": "student",
            "current_stage": "逻辑压力测试",
            "project_full_text": "智能农业项目计划书\n\n项目概述：利用AI技术帮助农民监测土壤状况，优化灌溉和施肥，提高农作物产量。\n\n目标用户：中小规模农场主\n\n价值主张：通过数据分析降低种植成本，提高产量和品质。\n\n商业模式：订阅制服务，按亩收费。\n\n市场分析：农业智能化是未来趋势，市场潜力巨大。\n\n财务预测：预计三年后实现盈利。",
            "project_name": "智能农业助手"
        },
        next_step=""
    )
    
    # 直接调用项目教练节点
    result = await project_coach_node(state)
    
    # 输出结果
    print("学生输入:", state["messages"][0].content)
    print("AI回复:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
    print("\n")

async def test_defense_mode():
    """测试用例3：答辩模式"""
    print("=== 测试用例3：答辩模式 ===")
    
    # 构建初始状态
    state = AgentState(
        messages=[
            HumanMessage(content="我需要模拟答辩，希望你扮演不同类型的专家来提问，包括激进型VC、技术流专家、保守型银行家等。我的项目是智能农业助手，帮助农民提高产量。")
        ],
        user_role="student",
        active_agent="",
        next_agent="",
        context_data={
            "portal_role": "student",
            "current_stage": "落地可行性校验",
            "project_full_text": "智能农业项目计划书\n\n项目概述：利用AI技术帮助农民监测土壤状况，优化灌溉和施肥，提高农作物产量。\n\n目标用户：中小规模农场主\n\n价值主张：通过数据分析降低种植成本，提高产量和品质。\n\n商业模式：订阅制服务，按亩收费。\n\n市场分析：农业智能化是未来趋势，市场潜力巨大。\n\n财务预测：预计三年后实现盈利。",
            "project_name": "智能农业助手",
            "competition_type": "互联网+",
            "competition_mode": "答辩模拟"
        },
        next_step=""
    )
    
    # 直接调用竞赛顾问节点
    result = await competition_advisor_node(state)
    
    # 输出结果
    print("学生输入:", state["messages"][0].content)
    print("AI回复:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
    print("\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_student_learning_mode())
    asyncio.run(test_competition_coach_mode())
    asyncio.run(test_defense_mode())
