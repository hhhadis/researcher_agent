import os
import json
import re
from typing import Dict, List, Optional
from zai import ZhipuAiClient


def _parse_json(s: str) -> Optional[Dict]:
    try:
        return json.loads(s)
    except Exception:
        # Fallback to regex search if simple parsing fails
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def classify_and_extract(title: str, abstract: str, sections: List[List[str]]) -> Dict:
    api_key = "77e73e22741f4b45854c777f4763236f.311YBhRcPuEQk3ZQ"

    if api_key:
        try:
            client = ZhipuAiClient(api_key=api_key)
            
            # Construct the text from abstract and sections
            paras = []
            if abstract:
                paras.append(abstract)
            for s in sections or []:
                if isinstance(s, list) and len(s) == 2:
                    paras.append(s[1])
                elif isinstance(s, str):
                    paras.append(s)
            full_text = "\n\n".join(paras)

            # Create the prompt for the model
            messages = [
                {
                    "role": "system",
                    "content": "你是一个科研文本分析助手。请阅读用户提供的学术论文文本，判断其是否与'扩散模型'(diffusion model)相关。如果相关，请提取能够支持你判断的3到6个关键段落。你的回答必须是一个JSON对象，包含两个字段: 'is_diffusion' (布尔值) 和 'paragraphs' (一个包含所提取段落字符串的列表)。如果论文不相关，'paragraphs'应为空列表。"
                },
                {
                    "role": "user",
                    "content": f"标题: {title}\n\n正文:\n{full_text}"
                }
            ]

            response = client.chat.completions.create(
                model="glm-4.5-flash",
                messages=messages,
                stream=False,
                max_tokens=4096,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            data = _parse_json(content) or {}
            
            is_diff = bool(data.get("is_diffusion"))
            paras = [p for p in (data.get("paragraphs") or []) if isinstance(p, str)]
            
            if is_diff and not paras and abstract:
                paras.append(abstract)

            return {"is_diffusion": is_diff, "paragraphs": paras}

        except Exception as e:
            print(f"An error occurred with the ZhipuAI API: {e}")
            # Fallback to local keyword matching on API error
            pass

    # Fallback: local keyword matching if no API key or if API fails
    text = " ".join([title or "", abstract or "", " ".join([x[1] if isinstance(x, list) and len(x)==2 else "" for x in sections or []])])
    kw = ["diffusion", "扩散模型", "denoising diffusion", "DDPM", "score matching", "stable diffusion", "扩散概率模型"]
    is_diff = any(k.lower() in text.lower() for k in kw)
    if not is_diff:
        return {"is_diffusion": False, "paragraphs": []}
    
    # If it is a diffusion paper, use all its paragraphs for embedding.
    paras = []
    if abstract:
        paras.append(abstract)
    for s in sections or []:
        t = s[1] if isinstance(s, list) and len(s) == 2 else str(s)
        paras.append(t)
    return {"is_diffusion": True, "paragraphs": paras}
