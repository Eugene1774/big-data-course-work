from typing import Dict, List, Any

# 赛事评估模型配置
CONTEST_MODELS = {
    "世纪杯 (学术科技类)": {
        "dimensions": ["创新性", "技术/科学性", "商业/实用性", "团队/社会效益"],
        "mapping": {
            "创新性": {"痛点发现": 0.7, "逻辑表达": 0.3},
            "技术/科学性": {"方案策划": 0.8, "逻辑表达": 0.2},
            "商业/实用性": {"商业建模": 0.7, "资源杠杆": 0.3},
            "团队/社会效益": {"资源杠杆": 0.5, "逻辑表达": 0.5}
        }
    },
    "“人工智能+”/机器人大赛": {
        "dimensions": ["创新性", "技术支撑", "应用前景", "团队稳定性"],
        "mapping": {
            "创新性": {"痛点发现": 0.6, "方案策划": 0.4},
            "技术支撑": {"方案策划": 0.7, "逻辑表达": 0.3},
            "应用前景": {"商业建模": 0.6, "资源杠杆": 0.4},
            "团队稳定性": {"资源杠杆": 0.6, "逻辑表达": 0.4}
        }
    },
    "中国国际大学生创新大赛": {
        "dimensions": ["创新性", "教育维度", "商业维度", "社会价值"],
        "mapping": {
            "创新性": {"痛点发现": 0.5, "方案策划": 0.5},
            "教育维度": {"逻辑表达": 0.8, "痛点发现": 0.2},
            "商业维度": {"商业建模": 0.8, "资源杠杆": 0.2},
            "社会价值": {"资源杠杆": 0.7, "逻辑表达": 0.3}
        }
    }
}

def calculate_contest_scores(five_forces_scores: List[float], contest_name: str) -> Dict[str, Any]:
    """根据选择的赛事计算四维得分"""
    ff_labels = ["痛点发现", "方案策划", "商业建模", "资源杠杆", "逻辑表达"]
    ff_map = dict(zip(ff_labels, five_forces_scores))
    
    model = CONTEST_MODELS.get(contest_name, CONTEST_MODELS["世纪杯 (学术科技类)"])
    contest_scores = []
    
    for dim in model["dimensions"]:
        weights = model["mapping"][dim]
        score = sum(ff_map.get(k, 0) * v for k, v in weights.items())
        contest_scores.append(round(score, 2))
        
    return {
        "dimensions": model["dimensions"],
        "scores": contest_scores,
        "overall": round(sum(contest_scores) / len(contest_scores), 2)
    }