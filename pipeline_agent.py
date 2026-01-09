"""
Persona Interview System using LangGraph v1 create_agent
도구 에이전트와 타겟 LLM 간의 대화 시스템
"""

import json
import random
from typing import Dict, List, Any, TypedDict, Annotated
from collections import defaultdict
import anthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic


# ============================================================================
# 유틸리티 함수들
# ============================================================================

def load_json(filepath: str) -> Dict:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """JSON 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# 도구 함수들 (Tool Agent가 사용)
# ============================================================================

def load_persona_definition(dummy: str = "") -> str:
    """페르소나 정의 JSON 파일을 로드합니다."""
    try:
        data = load_json('persona_definition.json')
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error loading persona definition: {str(e)}"


def load_bridging_relationships(dummy: str = "") -> str:
    """브리징 관계 JSON 파일을 로드합니다."""
    try:
        data = load_json('bridging_relationships.json')
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error loading bridging relationships: {str(e)}"


def extract_bridging_from_conversation(conversation: str) -> str:
    """
    대화에서 브리징 관계를 추출합니다.
    실제로는 LLM이 대화를 분석하여 관계를 찾아야 하지만,
    여기서는 간단히 기존 브리징 관계를 참조합니다.
    """
    try:
        relationships = load_json('bridging_relationships.json')
        return json.dumps(relationships, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error extracting bridging: {str(e)}"


def create_importance_graph(bridging_data: str) -> str:
    """
    브리징 관계를 기반으로 중요도 그래프를 생성합니다.
    각 노드(속성)의 중요도는 연결된 엣지 수로 결정됩니다.
    """
    try:
        data = json.loads(bridging_data)
        relationships = data.get('bridging_relationships', {}).get('rules', [])
        
        # 노드별 연결 횟수 카운트
        node_connections = defaultdict(int)
        edge_list = []
        
        for rule in relationships:
            from_node = rule['from']
            to_nodes = rule['to']
            strength = rule.get('strength', 'medium')
            
            # 각 연결에 대해 카운트
            for to_node in to_nodes:
                node_connections[from_node] += 1
                node_connections[to_node] += 1
                edge_list.append({
                    'from': from_node,
                    'to': to_node,
                    'strength': strength
                })
        
        # 중요도 계산 (연결 횟수 기반)
        max_connections = max(node_connections.values()) if node_connections else 1
        importance_scores = {
            node: count / max_connections 
            for node, count in node_connections.items()
        }
        
        # 중요도 순으로 정렬
        sorted_nodes = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        graph_result = {
            'nodes': [
                {'name': node, 'importance': score, 'connections': node_connections[node]}
                for node, score in sorted_nodes
            ],
            'edges': edge_list,
            'summary': {
                'total_nodes': len(node_connections),
                'total_edges': len(edge_list),
                'most_important': sorted_nodes[0][0] if sorted_nodes else None,
                'most_important_score': sorted_nodes[0][1] if sorted_nodes else 0
            }
        }
        
        return json.dumps(graph_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Error creating graph: {str(e)}"


def generate_random_persona(dummy: str = "") -> str:
    """
    랜덤 페르소나를 생성합니다.
    각 카테고리에서 하나의 항목만 선택합니다.
    """
    try:
        persona_def = load_json('persona_definition.json')
        structure = persona_def['structure']
        
        persona = {}
        
        # Social Role: 랜덤으로 하나의 카테고리에서 하나의 역할 선택
        social_categories = structure['social_role']['categories']
        category_key = random.choice(list(social_categories.keys()))
        persona['social_role'] = random.choice(social_categories[category_key]['examples'])
        
        # Personality: 각 특성마다 0 또는 1 선택
        personality_traits = {}
        for trait_obj in structure['personality']['categories']:
            trait_name = list(trait_obj.keys())[0]
            personality_traits[trait_name] = random.choice(['0', '1'])
        persona['personality'] = personality_traits
        
        # Background: 각 카테고리에서 하나씩 선택
        background = {}
        for bg_key, bg_data in structure['background']['categories'].items():
            background[bg_key] = random.choice(bg_data['examples'])
        persona['background'] = background
        
        # Interests: 각 카테고리에서 하나씩 선택
        interests = {}
        for int_key, int_data in structure['interests']['categories'].items():
            interests[int_key] = random.choice(int_data['examples'])
        persona['interests'] = interests
        
        result = {
            'persona': persona,
            'description': 'Randomly generated persona with one item per category'
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Error generating persona: {str(e)}"


# ============================================================================
# 타겟 LLM 페르소나 생성
# ============================================================================

def create_target_persona() -> Dict[str, Any]:
    """타겟 LLM에 부여할 페르소나 생성 (각 유형별로 한 항목만)"""
    persona_result = json.loads(generate_random_persona())
    return persona_result['persona']


def format_persona_for_prompt(persona: Dict[str, Any]) -> str:
    """페르소나를 프롬프트용 텍스트로 변환"""
    parts = []
    parts.append(f"직업/역할: {persona['social_role']}")
    
    parts.append("\n성격 특성 (Big Five):")
    for trait, value in persona['personality'].items():
        level = "높음" if value == "1" else "낮음"
        parts.append(f"  - {trait}: {level}")
    
    parts.append("\n배경 정보:")
    for key, value in persona['background'].items():
        parts.append(f"  - {key}: {value}")
    
    parts.append("\n관심사 및 선호:")
    for key, value in persona['interests'].items():
        parts.append(f"  - {key}: {value}")
    
    return "\n".join(parts)


# ============================================================================
# 에이전트 생성
# ============================================================================

def create_tool_agent(model: ChatAnthropic, memory: MemorySaver):
    """도구 에이전트 생성 - 페르소나 분석 및 그래프 생성 담당"""
    
    tools = [
        {
            "name": "load_persona_definition",
            "description": "페르소나 정의 JSON 파일을 로드합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "사용하지 않는 더미 파라미터"}
                },
                "required": []
            }
        },
        {
            "name": "load_bridging_relationships",
            "description": "브리징 관계 JSON 파일을 로드합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "사용하지 않는 더미 파라미터"}
                },
                "required": []
            }
        },
        {
            "name": "extract_bridging_from_conversation",
            "description": "대화에서 브리징 관계를 추출합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "conversation": {"type": "string", "description": "분석할 대화 내용"}
                },
                "required": ["conversation"]
            }
        },
        {
            "name": "create_importance_graph",
            "description": "브리징 관계를 기반으로 중요도 그래프를 생성합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "bridging_data": {"type": "string", "description": "브리징 관계 JSON 데이터"}
                },
                "required": ["bridging_data"]
            }
        },
        {
            "name": "generate_random_persona",
            "description": "랜덤 페르소나를 생성합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dummy": {"type": "string", "description": "사용하지 않는 더미 파라미터"}
                },
                "required": []
            }
        }
    ]
    
    # 도구 함수 매핑
    tool_functions = {
        "load_persona_definition": load_persona_definition,
        "load_bridging_relationships": load_bridging_relationships,
        "extract_bridging_from_conversation": extract_bridging_from_conversation,
        "create_importance_graph": create_importance_graph,
        "generate_random_persona": generate_random_persona
    }
    
    system_message = """당신은 페르소나 분석 전문가입니다.
주어진 도구들을 사용하여 다음 작업을 수행합니다:
1. 페르소나 정의 확인
2. 대화에서 브리징 관계 추출
3. 중요도 그래프 생성
4. 타겟 LLM과의 인터뷰를 통해 페르소나 파악

항상 체계적으로 접근하고, 각 단계를 명확히 설명하세요."""

    # LangGraph의 create_react_agent 사용
    agent = create_react_agent(
        model,
        tools=tools,
        checkpointer=memory,
        state_modifier=system_message
    )
    
    return agent, tool_functions


def create_target_llm(model: ChatAnthropic, persona: Dict[str, Any]):
    """타겟 LLM 생성 - 특정 페르소나를 가진 대화 상대"""
    
    persona_text = format_persona_for_prompt(persona)
    
    system_message = f"""당신은 다음과 같은 페르소나를 가진 사람입니다:

{persona_text}

이 페르소나에 완전히 몰입하여 대화하세요. 
질문에 답할 때는 이 페르소나의 특성, 배경, 가치관을 자연스럽게 반영하세요.
직접적으로 페르소나 정보를 나열하지 말고, 대화를 통해 자연스럽게 드러나도록 하세요."""

    return model, system_message


# ============================================================================
# 메인 실행 함수
# ============================================================================

def run_interview_system(api_key: str, num_questions: int = 5):
    """인터뷰 시스템 실행"""
    
    # 모델 초기화
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        temperature=0.7
    )
    
    # 메모리 초기화
    memory = MemorySaver()
    
    # 타겟 페르소나 생성
    target_persona = create_target_persona()
    print("=" * 80)
    print("생성된 타겟 페르소나:")
    print("=" * 80)
    print(json.dumps(target_persona, ensure_ascii=False, indent=2))
    print("\n")
    
    # 에이전트 생성
    tool_agent, tool_functions = create_tool_agent(model, memory)
    target_model, target_system_msg = create_target_llm(model, target_persona)
    
    # 대화 히스토리
    conversation_history = []
    
    # 초기 지시사항
    initial_instruction = f"""당신의 임무:
1. 먼저 페르소나 정의와 브리징 관계를 확인하세요.
2. 타겟 LLM과 {num_questions}개의 질문을 통해 인터뷰를 진행하세요.
3. 인터뷰 후 브리징 관계를 분석하고 중요도 그래프를 생성하세요.
4. 수집한 정보를 바탕으로 타겟의 페르소나를 추론하세요.

시작하세요!"""

    config = {"configurable": {"thread_id": "interview-session-1"}}
    
    print("=" * 80)
    print("인터뷰 시작")
    print("=" * 80)
    
    # 도구 에이전트 실행 (초기 설정)
    result = tool_agent.invoke(
        {"messages": [HumanMessage(content=initial_instruction)]},
        config=config
    )
    
    # 실제 도구 호출 처리
    for msg in result['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                
                if tool_name in tool_functions:
                    tool_result = tool_functions[tool_name](**tool_args)
                    print(f"\n[도구 호출: {tool_name}]")
                    if len(tool_result) > 500:
                        print(tool_result[:500] + "...")
                    else:
                        print(tool_result)
    
    # 인터뷰 진행
    for i in range(num_questions):
        print(f"\n{'=' * 80}")
        print(f"질문 {i+1}/{num_questions}")
        print('=' * 80)
        
        # 도구 에이전트가 질문 생성
        question_prompt = f"이제 타겟에게 {i+1}번째 질문을 하세요. 자연스럽고 페르소나를 파악할 수 있는 질문을 만드세요."
        
        result = tool_agent.invoke(
            {"messages": [HumanMessage(content=question_prompt)]},
            config=config
        )
        
        agent_question = result['messages'][-1].content
        print(f"\n[도구 에이전트 질문]: {agent_question}")
        
        # 타겟 LLM이 답변
        target_messages = [
            SystemMessage(content=target_system_msg),
            HumanMessage(content=agent_question)
        ]
        target_response = target_model.invoke(target_messages)
        target_answer = target_response.content
        print(f"\n[타겟 답변]: {target_answer}")
        
        # 대화 기록
        conversation_history.append({
            "question": agent_question,
            "answer": target_answer
        })
        
        # 도구 에이전트에게 답변 전달
        feedback_prompt = f"타겟의 답변: {target_answer}\n\n이 답변을 분석하고 다음 질문을 준비하세요."
        tool_agent.invoke(
            {"messages": [HumanMessage(content=feedback_prompt)]},
            config=config
        )
    
    # 최종 분석
    print(f"\n{'=' * 80}")
    print("최종 분석")
    print('=' * 80)
    
    conversation_text = json.dumps(conversation_history, ensure_ascii=False)
    
    final_analysis_prompt = f"""인터뷰가 완료되었습니다. 
다음 대화 내용을 분석하여:
1. 브리징 관계를 추출하세요
2. 중요도 그래프를 생성하세요
3. 타겟의 페르소나를 추론하세요

대화 내용:
{conversation_text}"""

    final_result = tool_agent.invoke(
        {"messages": [HumanMessage(content=final_analysis_prompt)]},
        config=config
    )
    
    # 최종 도구 호출 처리
    for msg in final_result['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                
                if tool_name in tool_functions:
                    tool_result = tool_functions[tool_name](**tool_args)
                    print(f"\n[도구 호출: {tool_name}]")
                    print(tool_result)
    
    print(f"\n{'=' * 80}")
    print("인터뷰 시스템 종료")
    print('=' * 80)
    
    # 결과 저장
    results = {
        "target_persona": target_persona,
        "conversation_history": conversation_history,
        "timestamp": "2025-10-27"
    }
    
    save_json(results, "interview_results.json")
    print("\n결과가 'interview_results.json'에 저장되었습니다.")
    
    return results


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    import os
    
    # API 키 설정 (환경변수 또는 직접 입력)
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
    
    if API_KEY == "your-api-key-here":
        print("경고: ANTHROPIC_API_KEY 환경변수를 설정하거나 코드에서 직접 API 키를 입력하세요.")
    else: