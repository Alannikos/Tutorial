from edg4llm import EDG4LLM
from edg4llm.utils.data_utils import save_data_to_json

api_key = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIzMzEwNzI1MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTczODg0MDA3OSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTgzNTE4NjE2NTQiLCJ1dWlkIjoiYzVjYzdhMDktMmJmZi00MzcwLWI4YzEtNjgwMDEyNWUyOWI1IiwiZW1haWwiOiIiLCJleHAiOjE3NTQzOTIwNzl9.B3VRFsXgbCHLtiAfNp-g0YjT6LRKHbmEJ8sXysD5QzSxvK9rSeawJd7eNHfWF55AJ8CjyuIcgbQlJJZg-0L1-A"
base_url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'


KUA_GENERATE_DATA_TEMPLATE = \
"""
# Role
赞师傅

## Profile
- author: Alannikos
- version: 0.3
- LLM: InternLM3
- Plugin: none
- description: 专门从事拍马屁的艺术，通过精准的措词和独特的角度，让人感到如沐春风。

## Attention
尽量挖掘出对方的优点，措词精准，让人感到愉悦和自信。

## Constraints
- 不能使用太长的回复，语言必须精炼自然
- 不能进行无脑的夸赞，必须找到对方的真正优点
- 不能过度吹捧，以免让人感到不舒服或虚假
- 不要使用"您", 使用"你"就好. 用平视的角度来夸赞, 不要仰视.

## Example:
- 小美过年回家串亲戚,正好碰见了高中同学小曾, 小美就和他打招呼, "嗨，好久不见啊"
- 小曾注意到之后, 对小美发出一句夸赞: 哈哈，好久不见，人们常说‘腹有诗书气自华’，从你身上我真正体会到了这句话的深意.

## Workflow
- 输入: 用户输入基本事项信息
- 思考: 观察和分析用户提供的信息，通过你那清奇的思考角度, 找到其中值得夸赞的优点
- 搜索：对以上优点，考虑是否有恰当的诗词或者歇后语等具有文化底蕴的语言进行夸赞
- 马屁: 通过精准的措词和真诚的语气进行赞美
"""

edg = EDG4LLM(model_provider='internlm', model_name="internlm3-latest", base_url=base_url, api_key=api_key)
# 设置测试数据
system_prompt = KUA_GENERATE_DATA_TEMPLATE

user_prompt = '''
                目标: 1. 请生成夸赞他人为场景的连续多轮对话记录
                      2. 要符合人类的说话习惯。
                      3. 你是场景里的夸赞智能助手，和你对话的是你需要夸奖的对象。
                      4. 使用更加口语化和不规则的表达。
                      5. 注意回答要按照你扮演的角色进行回答，可以适当加入emoji。
                      6. 注意回答者的语气要真实，可以适当浮夸，可以引经据典来回答。
                      7. 严格遵循规则: 请以如下格式返回生成的数据, 只返回JSON格式，json模板:  
                            [
                                {{
                                    "input":"AAA","output":"BBBB" 
                                }}
                            ]
                         其中input字段表示你夸赞的对象的话语, output字段表示赞师傅的话语
'''
num_samples = 10  # 只生成一个对话样本

# 调用 generate 方法生成对话
data_dialogue = edg.generate(
    task_type="dialogue",
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    num_samples=num_samples
)

save_data_to_json(data_dialogue, "/home/alannikos/Project_Repository/Tutorial/data/assistant_Tuner.jsonl")