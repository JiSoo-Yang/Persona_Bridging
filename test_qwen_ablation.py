"""
Persona Interview System with Hugging Face Models
도구 에이전트(GPT)와 타겟 LLM(Hugging Face 모델) 간의 대화 시스템
- Agent(LLM): 대화 분석 및 브리징 추론
- Tools: 브리징 추출 프롬프트 생성, 결과 저장, 그래프 생성 및 시각화
- Qwen 임베딩 기반 유사도 평가
"""

# ============================================================================
# CRITICAL: macOS/thread & deps
# ============================================================================
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # ✅ torchvision 의존 차단
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용

import json
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import time
import re
from difflib import SequenceMatcher
import numpy as np

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    LogitsProcessorList, NoBadWordsLogitsProcessor,
)

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# 유틸리티
# ============================================================================

def load_json(filepath: str) -> Dict:
    """JSON 파일 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Dict, filepath: str):
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================================
# 도구 함수들 (Tool Agent가 사용)
# ============================================================================

# Resolve schema files relative to this script (works from any working directory)
def _resolve_schema(_name):
    _here = Path(__file__).resolve()
    for _base in (_here.parent, _here.parent.parent):
        _p = _base / "schema" / _name
        if _p.exists():
            return str(_p)
    return str(_here.parent / "schema" / _name)

PERSONA_SCHEMA_PATH = _resolve_schema("persona_schema.json")
BRIDGING_PATH = _resolve_schema("bridging_relationships.json")
TOOL_MODEL = 'gpt-4'

def load_persona_definition(dummy: str = "") -> str:
    """페르소나 정의 JSON 파일을 로드합니다."""
    try:
        data = load_json(PERSONA_SCHEMA_PATH)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error loading persona definition: {str(e)}"

def load_bridging_relationships(dummy: str = "") -> str:
    """브리징 관계 정의 JSON 파일을 로드합니다 (언어학적 정의)."""
    try:
        data = load_json(BRIDGING_PATH)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error loading bridging relationships: {str(e)}"

# 랜덤 페르소나 생성
def generate_random_persona(dummy: str = "") -> Dict:
    """
    랜덤 페르소나를 생성합니다. (각 대분류에서 1개씩만 선택)
    Returns:
        Dict with keys: 'persona', 'target_attributes', 'description'
    """
    try:
        persona_def = load_json(PERSONA_SCHEMA_PATH)
        structure = persona_def["structure"]

        persona = {}
        target_attributes = {}

        # 1. Social Role - 1개 선택
        social_categories = structure["social_role"]["categories"]
        category_key = random.choice(list(social_categories.keys()))
        selected_role = random.choice(social_categories[category_key]["examples"])
        persona["social_role"] = selected_role
        target_attributes["social_role"] = selected_role

        # 2. Personality - 5개 trait 중 1개만 선택
        personality_categories = structure["personality"]["categories"]
        selected_trait_obj = random.choice(personality_categories)
        selected_trait_name = list(selected_trait_obj.keys())[0]
        selected_trait_value = random.choice(["yes", "no"])
        persona["personality"] = {selected_trait_name: selected_trait_value}
        target_attributes["personality"] = {
            "trait": selected_trait_name,
            "value": selected_trait_value
        }

        # 3. Background - 여러 카테고리 중 1개만 선택
        background_categories = structure["background"]["categories"]
        selected_bg_key = random.choice(list(background_categories.keys()))
        selected_bg_value = random.choice(background_categories[selected_bg_key]["examples"])
        persona["background"] = {selected_bg_key: selected_bg_value}
        target_attributes["background"] = {
            "category": selected_bg_key,
            "value": selected_bg_value
        }

        # 4. Interests - 여러 카테고리 중 1개만 선택
        interests_categories = structure["interests"]["categories"]
        selected_int_key = random.choice(list(interests_categories.keys()))
        selected_int_value = random.choice(interests_categories[selected_int_key]["examples"])
        persona["interests"] = {selected_int_key: selected_int_value}
        target_attributes["interests"] = {
            "category": selected_int_key,
            "value": selected_int_value
        }

        result = {
            "persona": persona,
            "target_attributes": target_attributes,
            "description": "Randomly generated persona with one attribute per major category (4 total)"
        }
        return result

    except Exception as e:
        fallback = {
            "error": f"{type(e).__name__}: {str(e)}",
            "persona": {
                "social_role": "student",
                "personality": {"openness": "yes"},
                "background": {"education": "high school"},
                "interests": {"hobby": "reading"},
            },
            "target_attributes": {
                "social_role": "student",
                "personality": {"trait": "openness", "value": "yes"},
                "background": {"category": "education", "value": "high school"},
                "interests": {"category": "hobby", "value": "reading"}
            },
            "description": "Fallback persona due to loading error",
        }
        return fallback

@tool
def create_bridging_extraction_prompt_tool(conversation_json: str) -> str:
    """
    브리징 관계 추출을 위한 상세 프롬프트를 생성합니다.
    bridging_relationships.json의 정의를 프롬프트에 포함합니다.
    """
    try:
        bridging_def = load_json(BRIDGING_PATH)
        conversation = json.loads(conversation_json)
        utterances = []
        for turn in conversation:
            utterances.append(f"Q: {turn['question']}")
            utterances.append(f"A: {turn['answer']}")
        utterances_text = "\n".join([f"{i+1}. {utt}" for i, utt in enumerate(utterances)])

        prompt = f"""Extract **Bridging Inference Relations** from the conversation below.

=== CONVERSATION ===
{utterances_text}

=== BRIDGING INFERENCE DEFINITION ===
{json.dumps(bridging_def, ensure_ascii=False, indent=2)}

=== YOUR TASK ===
1. Identify implicit connections that require world knowledge or semantic frame understanding
2. Extract bridging relations following the relation types defined above
3. For each relation, provide:
   - anchor: A SINGLE WORD or SHORT PHRASE (1-3 words max) - the key concept mentioned first
   - anaphor: A SINGLE WORD or SHORT PHRASE (1-3 words max) - the implicitly connected concept
   - relation_type: one of (part-of, member-of, instrument, theme, cause-of, in, temporal)
   - explanation: why this requires inference
   - sentence_context: the relevant utterances

=== CRITICAL CONSTRAINTS ===
- anchor and anaphor MUST be SHORT (1-3 words maximum)
- They should be KEY CONCEPTS or NOUNS, not full phrases or clauses
- Extract the CORE CONCEPT only

=== GOOD EXAMPLES ===
✅ anchor: "data engineer", anaphor: "pipeline"
✅ anchor: "training", anaphor: "public facilities"
✅ anchor: "murder", anaphor: "knife"
✅ anchor: "room", anaphor: "ceiling"

=== BAD EXAMPLES (TOO LONG) ===
❌ anchor: "data engineer", anaphor: "reviewing the latest data pipeline updates and ensuring all data sources are synchronized"

=== OUTPUT FORMAT ===
Return ONLY valid JSON in this exact format:
{{
  "bridging_relations": [
    {{
      "anchor": "short concept (1-3 words)",
      "anaphor": "short concept (1-3 words)",
      "relation_type": "part-of|member-of|instrument|theme|cause-of|in|temporal",
      "explanation": "why inference is required",
      "sentence_context": "relevant sentences"
    }}
  ]
}}

If no bridging relations found, return: {{"bridging_relations": []}}
"""
        return prompt
    except Exception as e:
        return f"Error creating prompt: {str(e)}"

@tool
def save_bridging_results_tool(bridging_json: str, filename: str = "bridging_results.json") -> str:
    """추출된 브리징 관계를 JSON 파일로 저장합니다."""
    try:
        data = json.loads(bridging_json)
        output_path = f"outputs/{filename}"
        save_json(data, output_path)
        return f"Successfully saved bridging results to {output_path}"
    except Exception as e:
        return f"Error saving bridging results: {str(e)}"

@tool
def create_and_save_graph_tool(bridging_json: str, output_prefix: str = "graph") -> str:
    """
    브리징 관계를 기반으로 중요도 그래프를 생성하고 JSON과 PNG로 저장합니다.
    """
    try:
        data = json.loads(bridging_json)
        relations = data.get("bridging_relations", [])
        if not relations:
            return "No bridging relations found to create graph."

        node_connections = defaultdict(int)
        edge_list = []
        for rel in relations:
            anchor = rel.get("anchor", "")
            anaphor = rel.get("anaphor", "")
            relation_type = rel.get("relation_type", "unknown")
            if anchor and anaphor:
                node_connections[anchor] += 1
                node_connections[anaphor] += 1
                edge_list.append({"from": anchor, "to": anaphor, "relation_type": relation_type})

        max_connections = max(node_connections.values()) if node_connections else 1
        importance_scores = {node: cnt / max_connections for node, cnt in node_connections.items()}
        sorted_nodes = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        graph_data = {
            "nodes": [
                {"name": node, "importance": score, "connections": node_connections[node]}
                for node, score in sorted_nodes
            ],
            "edges": edge_list,
            "summary": {
                "total_nodes": len(node_connections),
                "total_edges": len(edge_list),
                "most_important_node": sorted_nodes[0][0] if sorted_nodes else None,
                "most_important_score": sorted_nodes[0][1] if sorted_nodes else 0,
            }
        }

        json_path = f"outputs/{output_prefix}_structure.json"
        save_json(graph_data, json_path)

        # PNG 시각화 (가능한 경우)
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            G = nx.DiGraph()
            for node_info in graph_data["nodes"]:
                G.add_node(node_info["name"], importance=node_info["importance"])
            for edge in graph_data["edges"]:
                G.add_edge(edge["from"], edge["to"], relation=edge["relation_type"])

            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=2, iterations=50)

            node_sizes = [3000 * importance_scores.get(n, 0.1) + 500 for n in G.nodes()]
            node_colors = [importance_scores.get(n, 0) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                   cmap=plt.cm.YlOrRd, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6, width=2)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            edge_labels = {(e["from"], e["to"]): e["relation_type"] for e in graph_data["edges"]}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title("Bridging Relations Importance Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            png_path = f"outputs/{output_prefix}_visualization.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            visualization_msg = f"\n- Visualization saved: {png_path}"
        except ImportError:
            visualization_msg = "\n- Visualization skipped (matplotlib/networkx not available)"
        except Exception as viz_error:
            visualization_msg = f"\n- Visualization failed: {str(viz_error)}"

        result_msg = f"""Graph created successfully!
- Structure JSON saved: {json_path}
{visualization_msg}

Summary:
- Total nodes: {graph_data['summary']['total_nodes']}
- Total edges: {graph_data['summary']['total_edges']}
- Most important node: {graph_data['summary']['most_important_node']} (score: {graph_data['summary']['most_important_score']:.2f})

Graph structure saved for analysis."""
        return result_msg

    except Exception as e:
        return f"Error creating graph: {str(e)}"

# ============================================================================
# 페르소나 누설(redaction) 유틸
# ============================================================================

def _collect_persona_terms(persona: Dict, ground_truth: Dict) -> List[str]:
    """페르소나/GT의 값 텍스트를 수집하여 정규식 매칭용 용어 목록 생성."""
    terms = set()

    def _add(x):
        if not x: return
        if isinstance(x, str):
            s = x.strip()
            if s: terms.add(s)
        elif isinstance(x, dict):
            for v in x.values():
                _add(v)

    _add(persona.get("social_role"))
    _add(persona.get("personality"))
    _add(persona.get("background"))
    _add(persona.get("interests"))

    _add(ground_truth.get("social_role"))
    _add(ground_truth.get("personality"))
    _add(ground_truth.get("background"))
    _add(ground_truth.get("interests"))

    return [t for t in terms if isinstance(t, str) and len(t) >= 3]


def redact_persona_leaks(text: str, persona_terms: List[str]) -> str:
    """
    타겟 LLM 답변에서 페르소나 단서를 룰-베이스로 제거/가림.
    """
    s = text

    # 1) 프리픽스형 자기소개 제거: "As a/an/the <role/desc>, "
    s = re.sub(r'(?i)^\s*as\s+(an?|the)\s+[^,.;:!?]+[,.;:!?-–—]\s*', '', s)

    # 2) 문장 내 자기소개 제거: "I am a/an|the <role/desc> ..." → 해당 구절만 축약
    s = re.sub(r"(?i)\bI\s*(?:am|'m|'m)\s+(?:an?|the)\s+[^,.;:!?]+", 'I', s)

    # 3) 직무 표현: "I work/serve as <role>"
    s = re.sub(r'(?i)\bI\s*(?:work|serve)\s+as\s+[^,.;:!?]+', 'I', s)

    # 4) 페르소나 값 자체 가리기
    for term in persona_terms:
        pat = r'(?i)\b' + re.escape(term) + r's?\b'
        s = re.sub(pat, ' ', s)

    # 5) 일반 표현 축약
    s = re.sub(r'(?i)\bmy\s+(background|role|job|position|personality|interests?)\b', 'my ', s)

    # 6) 과도 삭제 방지
    cleaned = s.strip()
    if len(cleaned) < 5:
        cleaned = "I prefer to focus on the topic rather than my personal background."

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# ============================================================================
# Qwen 임베딩 추출 & 코사인 유사도 계산 (개선)
# ============================================================================

class QwenEmbeddingExtractor:
    """Qwen 모델을 사용한 임베딩 추출기"""
    
    def __init__(self, model_name: str):
        print(f"\n[Embedding] Loading {model_name} for embedding extraction...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.to("cpu")
        self.model.eval()
        print("[Embedding] Model loaded successfully!")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        텍스트의 임베딩 벡터 추출 (마지막 히든 스테이트의 평균)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 마지막 히든 스테이트 가져오기
            hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            # 평균 풀링
            embedding = hidden_states.mean(dim=1).squeeze(0)  # (hidden_dim,)
        
        return embedding.cpu().numpy()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(np.dot(vec1_norm, vec2_norm))

def normalize_text(text: str) -> str:
    """
    텍스트 정규화: 대소문자, 단복수, 관사, 구두점 등 표면적 차이 제거
    """
    # 1. 소문자 변환
    text = text.lower()
    
    # 2. 관사 제거 (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    
    # 3. 소유격 제거 ('s)
    text = re.sub(r"'s\b", '', text)
    
    # 4. 구두점 제거
    text = re.sub(r'[^\w\s]', '', text)
    
    # 5. 복수형 단순화 (단, 3글자 이하는 제외)
    words = text.split()
    normalized_words = []
    for word in words:
        # 3글자 이하는 그대로 유지 (yes, no 등)
        if len(word) <= 3:
            normalized_words.append(word)
        # 일반적인 복수형 패턴
        elif word.endswith('ies') and len(word) > 4:
            # "companies" -> "company"
            normalized_words.append(word[:-3] + 'y')
        elif word.endswith('es') and len(word) > 4:
            # "boxes" -> "box"
            normalized_words.append(word[:-2])
        elif word.endswith('s') and len(word) > 3:
            # "agents" -> "agent"
            normalized_words.append(word[:-1])
        else:
            normalized_words.append(word)
    
    # 6. 여러 공백을 하나로
    text = ' '.join(normalized_words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def compute_embedding_similarity(
    ground_truth: Dict, 
    predicted: Dict, 
    embedding_extractor: QwenEmbeddingExtractor
) -> Dict:
    """
    Qwen 임베딩을 사용한 유사도 계산 (정규화된 텍스트 사용)
    """
    print("\n[Similarity] Computing embedding-based similarity...")
    
    # 각 카테고리별 텍스트 추출
    def extract_text(data: Dict, category: str) -> str:
        if category == "social_role":
            return str(data.get("social_role", ""))
        elif category == "personality":
            p = data.get("personality", {})
            # trait = p.get("trait", "")
            value = p.get("value", "")
            return value
        elif category == "background":
            bg = data.get("background", {})
            cat = bg.get("category", "")
            val = bg.get("value", "")
            return val
        elif category == "interests":
            it = data.get("interests", {})
            cat = it.get("category", "")
            val = it.get("value", "")
            return val
        return ""
    
    categories = ["social_role", "personality", "background", "interests"]
    similarities = {}
    
    for category in categories:
        gt_text_raw = extract_text(ground_truth, category)
        pred_text_raw = extract_text(predicted, category)
        
        # 정규화
        gt_text = normalize_text(gt_text_raw)
        pred_text = normalize_text(pred_text_raw)
        
        print(f"  [{category}]")
        print(f"    GT (raw):   {gt_text_raw}")
        print(f"    Pred (raw): {pred_text_raw}")
        print(f"    GT (norm):  {gt_text}")
        print(f"    Pred (norm): {pred_text}")
        
        if gt_text and pred_text:
            # 완전 일치 시 1.0
            if gt_text == pred_text:
                sim = 1.0
                print(f"    ✅ Exact match after normalization!")
            else:
                gt_emb = embedding_extractor.get_embedding(gt_text)
                pred_emb = embedding_extractor.get_embedding(pred_text)
                sim = cosine_similarity(gt_emb, pred_emb)
            similarities[category] = sim
            print(f"    Similarity: {sim:.4f}")
        else:
            similarities[category] = 0.0
            print(f"    Similarity: 0.0000 (missing data)")
    
    # 평균 계산
    overall = np.mean(list(similarities.values()))
    
    return {
        "scores": {
            "social_role": similarities.get("social_role", 0.0),
            "personality": similarities.get("personality", 0.0),
            "background": similarities.get("background", 0.0),
            "interests": similarities.get("interests", 0.0),
            "overall": overall
        },
        "ground_truth": ground_truth,
        "predicted": predicted,
        "method": "qwen_embedding_cosine_similarity_normalized"
    }

# ============================================================================
# 최종 페르소나 JSON 추출
# ============================================================================

def extract_predicted_persona_from_text(text: str) -> Optional[Dict]:
    """
    ToolAgent의 자유텍스트에서 최종 예측 페르소나 JSON을 견고하게 추출.
    중첩된 {} 와 문자열 내부의 중괄호까지 올바르게 처리한다.
    """
    if not text:
        return None

    print("\n[DEBUG] Extracting persona from text (first 500 chars):")
    print(text[:500])
    print("...\n")

    candidates: List[str] = []

    # 1) ```json ... ``` fenced code blocks
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    print(f"[DEBUG] Found {len(code_blocks)} code blocks")
    candidates.extend(code_blocks)

    # 2) Balanced-brace scan: extract every top-level {...} object.
    #    Correctly handles nested objects and braces inside strings,
    #    unlike a non-greedy regex (which stops at the first closing brace).
    def _balanced_objects(s: str) -> List[str]:
        objs: List[str] = []
        depth = 0
        start = None
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(s[start:i + 1])
                        start = None
        return objs

    candidates.extend(_balanced_objects(text))
    print(f"[DEBUG] Found {len(candidates)} candidate JSON object(s)")

    # 3) Try candidates last-first (the final prediction usually comes last).
    seen = set()
    for cand in reversed(candidates):
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "social_role" in obj:
            print("[DEBUG] ✅ Successfully extracted persona JSON!")
            return obj

    print("[DEBUG] ❌ Could not extract persona JSON from any method")
    return None


class HuggingFaceModelWrapper:
    """
    Qwen 전용 래퍼
    - CPU float32
    - <think>/<analysis>/<reasoning> 태그만 금지
    - 첫 시도 실패 시 필터 OFF로 1회 재시도
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        mode: str = "safe"  # "speed" or "safe"
    ):
        self.model_name = model_name
        self.device = "cpu"  # 강제 CPU
        self.max_new_tokens = max_new_tokens
        self.mode = mode

        print(f"\n[Qwen3] Loading {model_name} on {self.device} (mode={self.mode})")

        # ✅ trust_remote_code 제거 (내장 구현 사용)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        _ = AutoConfig.from_pretrained(model_name)

        # CPU float32 안전 설정 + eager attn
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",   # 맥/CPU에서 안전한 경로
        )
        self.model.to("cpu")
        self.model.eval()

        # pad token 보정
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 태그만 금지 ---
        self._bad_phrases = [
            "<think>", "</think>", "<analysis>", "</analysis>", "<reasoning>", "</reasoning>"
        ]
        self._bad_words_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in self._bad_phrases]
        self._logits_processors = LogitsProcessorList()
        if any(self._bad_words_ids):
            self._logits_processors.append(
                NoBadWordsLogitsProcessor(self._bad_words_ids, eos_token_id=self.tokenizer.eos_token_id)
            )

    def invoke(self, messages, do_sample: bool = False, temperature: float = 0.0):
        # LangChain 메시지 → dict(role, content) 정규화
        def _to_dict(m):
            role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
            if role in ("human", "user"): role = "user"
            elif role in ("ai", "assistant"): role = "assistant"
            elif role == "system": role = "system"
            content = getattr(m, "content", None) or str(m)
            return {"role": role, "content": content}

        chat = [_to_dict(m) for m in messages]

        # Qwen 전용 chat template
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to("cpu")

        def _generate_once(use_filters: bool):
            return self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=8,  # ✅ 빈 응답 방지
                do_sample=do_sample,
                temperature=(temperature if do_sample else 0.0),
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
                logits_processor=(self._logits_processors if use_filters else None),
            )

        with torch.inference_mode():
            # 1차: 금지태그 필터 ON
            out = _generate_once(use_filters=True)
            gen_only = out[0][inputs.input_ids.shape[1]:]
            text1 = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

            # 1차 결과가 너무 짧으면(혹은 비면) 2차: 필터 OFF
            if len(text1) < 2:
                out = _generate_once(use_filters=False)
                gen_only = out[0][inputs.input_ids.shape[1]:]
                text1 = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        # 시작부에 붙은 태그/메타 표현만 '가볍게' 제거
        cleaned = re.sub(r'^\s*<(think|analysis|reasoning)>\s*', '', text1, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*(Let me think|I (will|need to) think|Here is my reasoning)[:\s-]*', '', cleaned, flags=re.IGNORECASE)

        class Response:
            def __init__(self, content): self.content = content
        return Response(cleaned.strip())

# ============================================================================
# 타겟 LLM 생성
# ============================================================================

def create_target_persona() -> tuple:
    """타겟 페르소나를 랜덤 생성하고 페르소나, ground_truth, prediction_template 반환"""
    result = generate_random_persona()
    persona = result.get("persona", {})
    ground_truth = result.get("target_attributes", {})

    # Prediction Template 생성 (값 없이 카테고리만)
    prediction_template = {
        "social_role": "??",
        "personality": {
            "trait": ground_truth.get("personality", {}).get("trait", "unknown"),
            "value": "??"
        },
        "background": {
            "category": ground_truth.get("background", {}).get("category", "unknown"),
            "value": "??"
        },
        "interests": {
            "category": ground_truth.get("interests", {}).get("category", "unknown"),
            "value": "??"
        }
    }
    return persona, ground_truth, prediction_template

def create_target_llm(model_name: str, persona: Dict, device: str = "cpu") -> tuple:
    """타겟 LLM 생성 (페르소나 주입)"""
    model = HuggingFaceModelWrapper(model_name, device)

    system_msg = f"""You are roleplaying as a person with the following characteristics:

Social Role: {persona.get('social_role', 'unknown')}
Personality Traits: {json.dumps(persona.get('personality', {}), indent=2)}
Background: {json.dumps(persona.get('background', {}), indent=2)}
Interests: {json.dumps(persona.get('interests', {}), indent=2)}

STRICT OUTPUT RULES:
- Answer in exactly ONE complete sentence.
- Do NOT include any XML/angle-bracket tags like <think>, <analysis>, or similar.
- Do NOT reveal internal thoughts, step-by-step reasoning, or meta comments (e.g., "Let me think").
- Do NOT start with "As a ..." or explicitly state your job title.
- If unsure, just provide your best single-sentence reply without showing any reasoning."""
    return model, system_msg

# ============================================================================
# 에이전트 생성
# ============================================================================

def create_tool_agent(openai_api_key: str, memory: MemorySaver, prediction_template: Dict):
    """Tool Agent (GPT) 생성 - 예측해야 할 속성 카테고리만 전달 (값 없음)"""
    llm = ChatOpenAI(model=TOOL_MODEL, api_key=openai_api_key, temperature=0.3)

    tools = [
        create_bridging_extraction_prompt_tool,
        save_bridging_results_tool,
        create_and_save_graph_tool,
    ]

    prediction_targets = f"""
TARGET ATTRIBUTES TO PREDICT (4 total) - You must predict the VALUES:

1. Social Role: ?? (predict the specific role)
2. Personality Trait - {prediction_template.get('personality', {}).get('trait', 'unknown')}: ?? (yes/no)
3. Background - {prediction_template.get('background', {}).get('category', 'unknown')}: ?? (specific value)
4. Interests - {prediction_template.get('interests', {}).get('category', 'unknown')}: ?? (specific value)

NOTE: You know the categories but NOT the values. Infer through interviewing + bridging analysis.
"""

    system_message = f"""You are an expert interview agent specializing in bridging inference analysis and persona prediction.

{prediction_targets}

Workflow:
1) Interview the target LLM (ask 1-by-1 questions)
2) Use create_bridging_extraction_prompt_tool to get a detailed prompt
3) Extract bridging relations using that prompt
4) Save with save_bridging_results_tool
5) Build & save graph with create_and_save_graph_tool
6) Predict all 4 attribute VALUES with brief reasoning

CRITICAL: Your FINAL response MUST include a JSON object in this EXACT format:
```json
{{
  "social_role": "predicted role here",
  "personality": {{
    "trait": "{prediction_template.get('personality', {}).get('trait', 'unknown')}",
    "value": "yes or no"
  }},
  "background": {{
    "category": "{prediction_template.get('background', {}).get('category', 'unknown')}",
    "value": "predicted value here"
  }},
  "interests": {{
    "category": "{prediction_template.get('interests', {}).get('category', 'unknown')}",
    "value": "predicted value here"
  }}
}}
```

Do NOT add any text after the JSON. The JSON must be the last thing in your response.
"""

    agent = create_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_message,
    )
    return agent

# ============================================================================
# Baseline Agent (No Bridging / No Graph) - 담화만으로 페르소나 추론
# ============================================================================

def create_baseline_agent(openai_api_key: str, prediction_template: Dict):
    """
    브리징/그래프/툴 호출 없이, 오직 담화(conversation_history)만 보고
    최종 페르소나 JSON을 출력하는 Baseline Agent.
    - 파서 안정성을 위해 반드시 ```json 코드펜스``` 안에 '순수 JSON'만 출력하도록 강제.
    - 템퍼러처 0으로 결정적 출력.
    """
    llm = ChatOpenAI(model=TOOL_MODEL, api_key=openai_api_key, temperature=0.0)

    # 시스템 메시지: 오직 JSON 코드펜스, 추가 텍스트 금지
    system_message = (
        "You are a concise interviewer who ONLY looks at the conversation history to infer a persona.\n"
        "You must NOT call any tools. You must NOT invent information beyond what can be reasonably inferred from the conversation alone.\n"
        "Return your answer ONLY as a single JSON object wrapped in a triple backtick fence with the language tag `json`.\n"
        "Do not include any extra text before or after the fence. Do not add comments. Use double quotes only. No trailing commas."
    )

    # LangChain의 create_agent 대신, LLM 단독 호출을 간단히 래핑
    class _BaselineAgent:
        def __init__(self, _llm, _sysmsg):
            self.llm = _llm
            self.system_message = _sysmsg

        def invoke(self, conversation_json: str) -> str:
            # 템플릿의 trait/category는 반드시 그대로 유지하도록 명시
            prompt = f"""Here is the interview conversation as JSON (array of {{question, answer}}):
{conversation_json}

Using ONLY this conversation (no external knowledge), infer the persona VALUES for the 4 categories.

Return ONLY the JSON below wrapped in a triple backtick fence (```json ... ```), with no extra text:

```json
{{
  "social_role": "predicted role here",
  "personality": {{
    "trait": "{prediction_template.get('personality', {}).get('trait', 'unknown')}",
    "value": "yes or no"
  }},
  "background": {{
    "category": "{prediction_template.get('background', {}).get('category', 'unknown')}",
    "value": "predicted value here"
  }},
  "interests": {{
    "category": "{prediction_template.get('interests', {}).get('category', 'unknown')}",
    "value": "predicted value here"
  }}
}}
```"""
            msgs = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=prompt),
            ]
            out = self.llm.invoke(msgs)
            return getattr(out, "content", str(out))  # 코드펜스 포함 그대로 반환

    return _BaselineAgent(llm, system_message)


def clean_agent_question(text: str) -> str:
    """LLM 출력에서 불필요한 접두어 제거"""
    patterns = [
        r"(?i)^here'?s\s+the\s+\w+\s+question.*?:\s*",
        r"(?i)^now\s+(let'?s|ask).*?:\s*",
        r"(?i)^question\s*\d*\s*[:\-]\s*",
        r"(?i)^q\d+\s*[:\-]\s*",
        r"(?i)^target\s+llm.*?:\s*",
    ]
    cleaned = text.strip()
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned).strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    return cleaned

# ============================================================================
# 메인 실행 함수
# ============================================================================

def run_baseline_ablation(
    openai_api_key: str,
    target_model_name: str,
    prediction_template: Dict,
    ground_truth: Dict,
    conversation_history: List[Dict],
    embedding_extractor: QwenEmbeddingExtractor
) -> Tuple[Dict, Dict]:
    """
    Baseline 에이전트로 페르소나 예측 → Qwen 임베딩 기반 유사도 계산.
    반환:
      - baseline_pred (dict): baseline이 출력한 페르소나 JSON
      - baseline_similarity (dict): 임베딩 코사인 유사도 리포트
    """
    baseline_agent = create_baseline_agent(openai_api_key, prediction_template)
    conversation_json = json.dumps(conversation_history, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("🧪 Baseline (No Bridging/Graph) - Persona Prediction")
    print("="*80)

    baseline_text = baseline_agent.invoke(conversation_json)
    print("\n[Baseline Raw Output]")
    print(baseline_text[:1000])

    baseline_pred = extract_predicted_persona_from_text(baseline_text) or {}
    if not baseline_pred:
        print("\n❌ Baseline: Could not parse persona JSON.")
        baseline_similarity = {
            "error": "Could not parse baseline predicted persona JSON.",
            "ground_truth": ground_truth,
            "predicted": baseline_pred
        }
        return baseline_pred, baseline_similarity

    # 동일한 임베딩 추출기(모델)로 유사도 계산
    baseline_similarity = compute_embedding_similarity(
        ground_truth, baseline_pred, embedding_extractor
    )

    # 저장
    save_json(baseline_pred, "outputs/baseline_predicted_persona.json")
    save_json(baseline_similarity, "outputs/baseline_similarity.json")

    # 출력
    print("\n" + "="*80)
    print("📊 GT vs Baseline - Qwen Embedding Similarity (Cosine)")
    print("="*80)
    s = baseline_similarity["scores"]
    print(f"\n1️⃣  Social Role:     {s['social_role']:.4f}")
    print(f"2️⃣  Personality:     {s['personality']:.4f}")
    print(f"3️⃣  Background:      {s['background']:.4f}")
    print(f"4️⃣  Interests:       {s['interests']:.4f}")
    print(f"\n{'─'*80}")
    print(f"📈 OVERALL AVERAGE:  {s['overall']:.4f}")
    print(f"{'─'*80}\n")

    return baseline_pred, baseline_similarity

def run_interview_system(
    openai_api_key: str,
    target_model_name: str,
    num_questions: int = 5,
    device: str = "cpu",
):
    print(f"\n{'='*80}")
    print("Persona Interview System with Qwen Embedding Similarity")
    print(f"Tool Agent: OpenAI {TOOL_MODEL}")
    print(f"Target LLM: {target_model_name}")
    print(f"{'='*80}\n")

    os.makedirs("outputs", exist_ok=True)

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "interview-session-1"}, "recursion_limit": 50}

    target_persona, ground_truth, prediction_template = create_target_persona()

    print("=" * 80)
    print("Generated Target Persona (Ground Truth - Hidden from Agent):")
    print("=" * 80)
    print(json.dumps(target_persona, ensure_ascii=False, indent=2))
    print("\n" + "=" * 80)
    print("Ground Truth Attributes (Hidden from Agent):")
    print("=" * 80)
    print(json.dumps(ground_truth, ensure_ascii=False, indent=2))
    print("\n" + "=" * 80)
    print("Prediction Template (Given to Agent - Values Hidden):")
    print("=" * 80)
    print(json.dumps(prediction_template, ensure_ascii=False, indent=2))
    print("\n")

    tool_agent = create_tool_agent(openai_api_key, memory, prediction_template)
    target_model, target_system_msg = create_target_llm(target_model_name, target_persona, device)

    conversation_history = []
    target_conversation = [SystemMessage(content=target_system_msg)]

    initial_instruction = f"""Your mission:
Interview the target LLM with {num_questions} questions to understand their persona.
Ask questions one at a time, analyzing each response before asking the next."""

    print("=" * 80)
    print("Interview Started")
    print("=" * 80)

    for i in range(num_questions):
        print(f"\n{'='*80}\nQuestion {i+1}/{num_questions}\n{'='*80}")

        if i == 0:
            question_prompt = initial_instruction + "\n\nCreate your first question."
        else:
            question_prompt = f"Based on the previous answer, create your next question (question {i+1}/{num_questions})."

        result = tool_agent.invoke({"messages": [HumanMessage(content=question_prompt)]}, config=config)
        raw_output = result["messages"][-1].content
        agent_question = clean_agent_question(raw_output)
        print(f"[🕵 Tool Agent Question]: {agent_question}")

        # === Target LLM 호출 ===
        target_conversation.append(HumanMessage(content=agent_question))
        target_response = target_model.invoke(target_conversation)
        target_answer = target_response.content

        # ✅ 페르소나 누설 마스킹
        persona_terms = _collect_persona_terms(target_persona, ground_truth)
        target_answer_redacted = redact_persona_leaks(target_answer, persona_terms)

        # 콘솔에서 원본/마스킹본 확인(원하면 원본 출력 제거 가능)
        print(f"[🎯 Target Answer (raw)]: {target_answer}")
        print(f"[🛡 Redacted]: {target_answer_redacted}")

        # Tool Agent/히스토리에는 redacted만 사용
        assistant_msg = type('Message', (), {'type': 'assistant', 'content': target_answer_redacted})()
        target_conversation.append(assistant_msg)

        conversation_history.append({"question": agent_question, "answer": target_answer_redacted})

        if i < num_questions - 1:
            feedback_prompt = (
                f"The target answered: '{target_answer_redacted}'\n\n"
                f"Briefly analyze this response (1-2 sentences) and prepare for the next question."
            )
            tool_agent.invoke({"messages": [HumanMessage(content=feedback_prompt)]}, config=config)

    print(f"\n{'='*80}\nFinal Analysis - Bridging Inference Extraction\n{'='*80}")
    conversation_json = json.dumps(conversation_history, ensure_ascii=False, indent=2)

    final_analysis_prompt = f"""The interview is complete. Now perform bridging inference analysis and persona prediction:

PREDICTION TARGETS (predict the VALUES for these categories):
{json.dumps(prediction_template, ensure_ascii=False, indent=2)}

STEP 1: Call create_bridging_extraction_prompt_tool with the conversation JSON
STEP 2: Use the returned prompt to analyze the conversation and extract bridging relations
STEP 3: Call save_bridging_results_tool to save your extracted relations
STEP 4: Call create_and_save_graph_tool to generate and visualize the importance graph
STEP 5: Based on the graph structure and conversation patterns, predict ALL 4 target attribute VALUES.

IMPORTANT: Return your FINAL PREDICTION as a single JSON object with keys:
- social_role (string)
- personality {{ "trait": string, "value": "yes" | "no" }}
- background {{ "category": string, "value": string }}
- interests {{ "category": string, "value": string }}

Conversation JSON:
{conversation_json}

Begin analysis now."""
    final_result = tool_agent.invoke({"messages": [HumanMessage(content=final_analysis_prompt)]}, config=config)
    final_text = final_result["messages"][-1].content if final_result and "messages" in final_result else ""

    print("\n" + "="*80)
    print("Agent Final Analysis (raw)")
    print("="*80)
    print(final_text)

    # ===== 최종 페르소나 JSON 추출 =====
    predicted_persona = extract_predicted_persona_from_text(final_text) or {}

    # ===== Qwen 임베딩 모델 1회 로드 (FULL/BASELINE 공용) =====
    embedding_extractor = QwenEmbeddingExtractor(target_model_name)

    # ===== FULL (우리 모델) 유사도 =====
    if predicted_persona:
        similarity_report = compute_embedding_similarity(
            ground_truth, predicted_persona, embedding_extractor
        )
    else:
        similarity_report = {
            "error": "Could not parse predicted persona JSON from ToolAgent output.",
            "ground_truth": ground_truth,
            "predicted": predicted_persona
        }

    # 저장
    save_json(similarity_report, "outputs/persona_similarity.json")

    # ===== 📊 카테고리별 유사도 출력 =====
    print("\n" + "="*80)
    print("📊 GT vs Our Model - Qwen Embedding Similarity (Cosine)")
    print("="*80)
    
    if "scores" in similarity_report:
        scores = similarity_report["scores"]
        print(f"\n1️⃣  Social Role:     {scores['social_role']:.4f}")
        print(f"2️⃣  Personality:     {scores['personality']:.4f}")
        print(f"3️⃣  Background:      {scores['background']:.4f}")
        print(f"4️⃣  Interests:       {scores['interests']:.4f}")
        print(f"\n{'─'*80}")
        print(f"📈 OVERALL AVERAGE:  {scores['overall']:.4f}")
        print(f"{'─'*80}\n")
    else:
        print("\n❌ Error: Could not compute similarity scores")
        if "error" in similarity_report:
            print(f"   Reason: {similarity_report['error']}")

    print(f"\n{'='*80}\nInterview System Completed\n{'='*80}")

    # ===== 🧪 Baseline Ablation: 담화만 보고 예측 =====
    baseline_pred, baseline_similarity = run_baseline_ablation(
        openai_api_key=openai_api_key,
        target_model_name=target_model_name,
        prediction_template=prediction_template,
        ground_truth=ground_truth,
        conversation_history=conversation_history,
        embedding_extractor=embedding_extractor
    )

    # 비교 요약 저장
    comparison = {
        "gt": ground_truth,
        "full_pred": predicted_persona,
        "baseline_pred": baseline_pred,
        "full_scores": similarity_report.get("scores", {}),
        "baseline_scores": baseline_similarity.get("scores", {}),
    }
    save_json(comparison, "outputs/ablation_full_vs_baseline.json")

    print("\n" + "="*80)
    print("🔬 FULL vs BASELINE (Overall)")
    print("="*80)
    full_overall = similarity_report.get("scores", {}).get("overall", None)
    base_overall = baseline_similarity.get("scores", {}).get("overall", None)
    print(f"GT ↔ FULL overall:     {full_overall if full_overall is not None else 'N/A'}")
    print(f"GT ↔ BASELINE overall: {base_overall if base_overall is not None else 'N/A'}")

    results = {
        "target_model": target_model_name,
        "target_persona": target_persona,
        "ground_truth": ground_truth,
        "prediction_template": prediction_template,
        "conversation_history": conversation_history,
        "agent_final_text": final_text,
        "predicted_persona": predicted_persona,
        "similarity_report_full": similarity_report,          # 변경: 키명 분리
        "baseline_predicted_persona": baseline_pred,           # 추가
        "similarity_report_baseline": baseline_similarity,     # 추가
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_filename = f"outputs/interview_results_{target_model_name.replace('/', '_')}.json"
    save_json(results, output_filename)
    print(f"\nInterview results saved to '{output_filename}'")
    print("\nGenerated files:")
    print("  - outputs/bridging_results.json (bridging relations)")
    print("  - outputs/graph_structure.json (graph data)")
    print("  - outputs/graph_visualization.png (graph visualization)")
    print("  - outputs/persona_similarity.json (GT vs Predicted similarity with Qwen embeddings)")
    print("  - outputs/baseline_predicted_persona.json (baseline prediction)")
    print("  - outputs/baseline_similarity.json (GT vs Baseline similarity)")
    print("  - outputs/ablation_full_vs_baseline.json (side-by-side comparison)")

    return results


# ============================================================================
# 실행 스크립트
# ============================================================================

if __name__ == "__main__":
    def load_env_file(env_path=".env"):
        """Load environment variables from .env file"""
        env_file = Path(env_path)
        if env_file.exists():
            print(f"✅ Loading environment variables from {env_path}")
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if not os.environ.get(key):
                            os.environ[key] = value
            print("✅ Environment variables loaded successfully\n")
        else:
            print(f"⚠️  No .env file found at {env_path}")
            print("You can create one with:\n  OPENAI_API_KEY=your-key-here\n  HF_TOKEN=your-token-here\n")

    start = time.time()
    load_env_file()

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found!")
        raise SystemExit(1)

    print("✅ OPENAI_API_KEY found")
    print("✅ HF_TOKEN found" if HF_TOKEN else "ℹ️  HF_TOKEN not set\n")

    TARGET_MODEL = "Qwen/Qwen3-1.7B"

    # ✅ 여러 번 인터뷰 실행
    NUM_RUNS = 1  # 원하는 횟수
    for i in range(NUM_RUNS):
        print(f"\n{'='*80}")
        print(f"🔥 RUN {i+1}/{NUM_RUNS}")
        print(f"{'='*80}\n")

        try:
            _ = run_interview_system(
                openai_api_key=OPENAI_API_KEY,
                target_model_name=TARGET_MODEL,
                num_questions=3,   # 각 세션에서 질문 개수
                device="cpu",
            )

        except Exception as e:
            print(f"\n❌ Run {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

    end = time.time()
    print(f"\n⏱️  Total execution time for {NUM_RUNS} runs: {end - start:.2f} seconds")
