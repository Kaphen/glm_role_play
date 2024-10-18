"""
一个简单的demo，调用CharacterGLM实现角色扮演，调用CogView生成图片，调用ChatGLM生成CogView所需的prompt。

依赖：
pyjwt
requests
streamlit
zhipuai
python-dotenv

运行方式：
```bash
streamlit run characterglm_api_demo_streamlit.py
```
"""

import requests
import time
import os
import random
import itertools
from typing import Literal, TypedDict, List, Union, Iterator, Optional

import jwt

import streamlit as st
from dotenv import load_dotenv


# 通过.env文件设置环境变量
# reference: https://github.com/theskumar/python-dotenv
load_dotenv()


## 数据类型 #####
class BaseMsg(TypedDict):
    pass


class TextMsg(BaseMsg):
    role: Literal["user", "assistant"]
    content: str


class ImageMsg(BaseMsg):
    role: Literal["image"]
    image: st.elements.image.ImageOrImageList
    caption: Optional[Union[str, List[str]]]


Msg = Union[TextMsg, ImageMsg]
TextMsgList = List[TextMsg]
MsgList = List[Msg]


class CharacterMeta(TypedDict):
    role_A_name: str   # 角色A名字
    role_A_info: str   # 角色A人设
    role_B_name: str  # 角色B名字
    role_B_info: str  # 角色B人设


def filter_text_msg(messages: MsgList) -> TextMsgList:
    return [m for m in messages if m["role"] != "image"]


## api ##
# 智谱开放平台API key，参考 https://open.bigmodel.cn/usercenter/apikeys
API_KEY: str = ""


class ApiKeyNotSet(ValueError):
    pass


def verify_api_key_not_empty():
    if not API_KEY:
        raise ApiKeyNotSet


def generate_token(apikey: str, exp_seconds: int):
    # reference: https://open.bigmodel.cn/dev/api#nosdk
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)
 
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
 
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def get_characterglm_response(messages: TextMsgList, meta: CharacterMeta):
    """ 通过http调用characterglm """
    # Reference: https://open.bigmodel.cn/dev/api#characterglm
    verify_api_key_not_empty()
    url = "https://open.bigmodel.cn/api/paas/v3/model-api/charglm-3/sse-invoke"
    resp = requests.post(
        url,
        headers={"Authorization": generate_token(API_KEY, 1800)},
        json=dict(
            model="charglm-3",
            meta=meta,
            prompt=messages,
            incremental=True)
    )
    resp.raise_for_status()
    
    # 解析响应（非官方实现）
    sep = b':'
    last_event = None
    for line in resp.iter_lines():
        if not line or line.startswith(sep):
            continue
        field, value = line.split(sep, maxsplit=1)
        if field == b'event':
            last_event = value
        elif field == b'data' and last_event == b'add':
            yield value.decode()


def get_characterglm_response_via_sdk(messages: TextMsgList, meta: CharacterMeta):
    """ 通过旧版sdk调用characterglm """
    # 与get_characterglm_response等价
    # Reference: https://open.bigmodel.cn/dev/api#characterglm
    # 需要安装旧版sdk，zhipuai==1.0.7
    import zhipuai
    verify_api_key_not_empty()
    zhipuai.api_key = API_KEY
    response = zhipuai.model_api.sse_invoke(
        model="charglm-3",
        meta= meta,
        prompt= messages,
        incremental=True
    )
    for event in response.events():
        if event.event == 'add':
            yield event.data


def get_chatglm_response_via_sdk(messages: TextMsgList):
    """ 通过sdk调用chatglm """
    # reference: https://open.bigmodel.cn/dev/api#glm-3-turbo  `GLM-3-Turbo`相关内容
    # 需要安装新版zhipuai
    from zhipuai import ZhipuAI
    verify_api_key_not_empty()
    client = ZhipuAI(api_key=API_KEY) # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-3-turbo",  # 填写需要调用的模型名称
        messages=messages,
        stream=True,
    )
    for chunk in response:
        yield chunk.choices[0].delta.content


def generate_role_appearance(role_profile: str):
    """ 用chatglm生成角色的外貌描写 """
    
    instruction = f"""
请从下列文本中，抽取人物的外貌描写。若文本中不包含外貌描写，请你推测人物的性别、年龄，并生成一段外貌描写。要求：
1. 只生成外貌描写，不要生成任何多余的内容。
2. 外貌描写不能包含敏感词，人物形象需得体。
3. 尽量用短语描写，而不是完整的句子。
4. 不要超过50字

文本：
{role_profile}
"""
    return get_chatglm_response_via_sdk(
        messages=[
            {
                "role": "user",
                "content": instruction.strip()
            }
        ]
    )


def generate_chat_scene_prompt(messages: TextMsgList, meta: CharacterMeta):
    """ 调用chatglm生成cogview的prompt，描写对话场景 """
    instruction = f"""
阅读下面的角色A人设与对话，生成一段文字描写场景。

{meta['role_A_name']}的人设：
{meta['role_A_info']}
    """.strip()

    if meta["role_B_info"]:
        instruction += f"""

{meta["role_B_name"]}的人设：
{meta["role_B_info"]}
""".rstrip()

    if messages:
        instruction += "\n\n对话：" + '\n'.join((meta['role_A_name'] if msg['role'] == "assistant" else meta['role_B_name']) + '：' + msg['content'].strip() for msg in messages)

    instruction += """
    
要求如下：
1. 只生成场景描写，不要生成任何多余的内容
2. 描写不能包含敏感词，人物形象需得体
3. 尽量用短语描写，而不是完整的句子
4. 不要超过50字
""".rstrip()
    print(instruction)
    
    return get_chatglm_response_via_sdk(
        messages=[
            {
                "role": "user",
                "content": instruction.strip()
            }
        ]
    )


def generate_cogview_image(prompt: str) -> str:
    """ 调用cogview生成图片，返回url """
    # reference: https://open.bigmodel.cn/dev/api#cogview
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=API_KEY) # 请填写您自己的APIKey
    
    response = client.images.generations(
        model="cogview-3", #填写需要调用的模型名称
        prompt=prompt
    )
    return response.data[0].url


def generate_fake_response(messages: TextMsgList, meta: CharacterMeta) -> Iterator[str]:
    # 用于debug
    bot_response = random.choice(['abcd', '你好', '世界', '42'])
    for c in bot_response:
        yield c
        time.sleep(0.5)


### UI ###
st.set_page_config(page_title="CharacterGLM API Demo", page_icon="🤖", layout="wide")
debug = os.getenv("DEBUG", "").lower() in ("1", "yes", "y", "true", "t", "on")

def update_api_key(key: Optional[str] = None):
    global API_KEY
    # 检查并初始化 API_KEY
    if 'API_KEY' not in st.session_state:
        st.session_state['API_KEY'] = API_KEY  # 或者设置为 null
    if debug:
        print(f'update_api_key. st.session_state["API_KEY"] = {st.session_state["API_KEY"]}, key = {key}')
    key = key or st.session_state["API_KEY"]
    if key:
        API_KEY = key

api_key = st.sidebar.text_input("API_KEY", value=os.getenv("API_KEY", ""), key="API_KEY", type="password", on_change=update_api_key)
update_api_key(api_key)


# 初始化
if "history" not in st.session_state:
    st.session_state["history"] = []
if "meta" not in st.session_state:
    st.session_state["meta"] = {
        "role_A_name": "",
        "role_A_info": "",
        "role_B_name": "",
        "role_B_info": "",
    }


def init_session():
    st.session_state["history"] = []


# 4个输入框，设置meta的4个字段
meta_labels = {
    "role_A_name": "角色A名",
    "role_A_info": "角色A人设",
    "role_B_name": "角色B名",
    "role_B_info": "角色B人设"
}

# 2x2 layout
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.text_input(label="角色A名", key="role_A_name", on_change=lambda : st.session_state["meta"].update(role_A_name=st.session_state["role_A_name"]), help="模型所扮演的角色的名字，不可以为空")
        st.text_area(label="角色A人设", key="role_A_info", on_change=lambda : st.session_state["meta"].update(role_A_info=st.session_state["role_A_info"]), help="角色的详细人设信息，不可以为空")

    with col2:
        st.text_input(label="角色B名", value="用户", key="role_B_name", on_change=lambda : st.session_state["meta"].update(role_B_name=st.session_state["role_B_name"]), help="用户的名字，不可以为空")
        st.text_area(label="角色B人设", value="", key="role_B_info", on_change=lambda : st.session_state["meta"].update(role_B_info=st.session_state["role_B_info"]), help="用户的详细人设信息，不可以为空")


def verify_meta() -> bool:
    # 检查`角色A名`和`角色A人设`是否空，若为空，则弹出提醒
    if st.session_state["meta"]["role_A_name"] == "" or st.session_state["meta"]["role_A_info"] == "":
        st.error("角色A名和角色A人设不能为空")
        return False
    else:
        return True


def draw_new_image():
    if not verify_meta():
        return
    text_messages = filter_text_msg(st.session_state["history"])
    if text_messages:
        image_prompt = "".join(
            generate_chat_scene_prompt(
                text_messages[-10: ],
                meta=st.session_state["meta"]
            )
        )
        
    else:
        image_prompt = "".join(generate_role_appearance(st.session_state["meta"]["role_A_info"]))

    if not image_prompt:
        st.error("调用chatglm生成Cogview prompt出错")
        return
    
    # TODO: 加上风格选项
    image_prompt = st.session_state["picture_style"] + '风格。' + image_prompt.strip()

    print(f"image_prompt = {image_prompt}")
    n_retry = 3
    st.markdown("正在生成图片，请稍等...")
    for i in range(n_retry):
        try:
            img_url = generate_cogview_image(image_prompt)
        except Exception as e:
            if i < n_retry - 1:
                st.error("遇到了一点小问题，重试中...")
            else:
                st.error("又失败啦，点击【生成图片】按钮可再次重试")
                return
        else:
            break
    img_msg = ImageMsg({"role": "image", "image": img_url, "caption": image_prompt})
    # 若history的末尾有图片消息，则替换它，（重新生成）
    # 否则，append（新增）
    while st.session_state["history"] and st.session_state["history"][-1]["role"] == "image":
        st.session_state["history"].pop()
    st.session_state["history"].append(img_msg)
    st.rerun()


button_labels = {
    "clear_meta": "清空人设",
    "clear_history": "清空对话历史",
    "select_style": "切换风格:",
    "gen_picture": "生成图片"
}
if debug:
    button_labels.update({
        "show_api_key": "查看API_KEY",
        "show_meta": "查看meta",
        "show_history": "查看历史"
    })

button2_labels = {
    "generate_chat_record": "生成对话",
    "download_chat_record": "下载对话内容到本地"
}
# 在同一行排列按钮
with st.container():
    n_button = len(button_labels)
    cols = st.columns(n_button)
    button_key_to_col = dict(zip(button_labels.keys(), cols))
    
    with button_key_to_col["clear_meta"]:
        clear_meta = st.button(button_labels["clear_meta"], key="clear_meta")
        if clear_meta:
            st.session_state["meta"] = {
                "role_A_name": "",
                "role_A_info": "",
                "role_B_name": "",
                "role_B_info": "",
            }
            st.rerun()

    with button_key_to_col["clear_history"]:
        clear_history = st.button(button_labels["clear_history"], key="clear_history")
        if clear_history:
            init_session()
            st.rerun()

    # 图片风格选项
    with button_key_to_col["select_style"]:
        option = st.selectbox(
            button_labels["select_style"],
            ("写实", "二次元", "水彩", "复古", "低多边形"),
            label_visibility="collapsed"
        )
        st.session_state["picture_style"] = option

    with button_key_to_col["gen_picture"]:
        gen_picture = st.button(button_labels["gen_picture"], key="gen_picture")

    if debug:
        with button_key_to_col["show_api_key"]:
            show_api_key = st.button(button_labels["show_api_key"], key="show_api_key")
            if show_api_key:
                print(f"API_KEY = {API_KEY}")
        
        with button_key_to_col["show_meta"]:
            show_meta = st.button(button_labels["show_meta"], key="show_meta")
            if show_meta:
                print(f"meta = {st.session_state['meta']}")
        
        with button_key_to_col["show_history"]:
            show_history = st.button(button_labels["show_history"], key="show_history")
            if show_history:
                print(f"history = {st.session_state['history']}")

# 生成对话数据按钮
with st.container():
    n_button2 = len(button2_labels)
    cols2 = st.columns(n_button2)
    button2_key_to_col = dict(zip(button2_labels.keys(), cols2))
    with button2_key_to_col["generate_chat_record"]:
        generate_chat_record = st.button(
            button2_labels["generate_chat_record"],
            key="generate_chat_record",
            icon=":material/mood:"
        )


# 展示对话历史
for msg in st.session_state["history"]:
    if msg["role"] == st.session_state["meta"]['role_A_name']:
        with st.chat_message(name=st.session_state["meta"]['role_A_name']):
            st.markdown(msg["content"])
    elif msg["role"] == st.session_state["meta"]['role_B_name']:
        with st.chat_message(name=st.session_state["meta"]['role_B_name']):
            st.markdown(msg["content"])
    elif msg["role"] == "image":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.image(msg["image"], caption=msg.get("caption", None))
    elif msg["role"] == "":
        continue
    else:
        raise Exception("Invalid role")


if gen_picture:
    draw_new_image()


def output_stream_response(response_stream: Iterator[str], placeholder):
    content = ""
    for content in itertools.accumulate(response_stream):
        placeholder.markdown(content)
    return content

meta_role_A_as_user = {
        'assistant_name': st.session_state["meta"]['role_B_name'],
        'assistant_info': st.session_state["meta"]['role_B_info'],
        'user_name': st.session_state["meta"]['role_A_name'],
        'user_info': st.session_state["meta"]['role_A_info']
    }

meta_role_B_as_user = {
        'assistant_name': st.session_state["meta"]['role_A_name'],
        'assistant_info': st.session_state["meta"]['role_A_info'],
        'user_name': st.session_state["meta"]['role_B_name'],
        'user_info': st.session_state["meta"]['role_B_info']
    }

'''
处理prompt并请求GLM大模型：
- 当轮到角色A回复时，在传给大模型的参数中，将角色A设为assistant, 将角色B设为user;
- 当轮到角色B回复时，在传给大模型的参数中，将角色B设为assistant, 将角色A设为user;
'''
def start_chat(role_name, placeholder):
    history_tmp = []
    history = filter_text_msg(st.session_state["history"])

    for m in history:
        new_message = m.copy()
        if new_message["role"] == role_name:
            new_message["role"] = 'user'
        else:
            new_message["role"] = 'assistant'
        history_tmp.append(new_message)

    if role_name == st.session_state["meta"]['role_A_name']:
        meta = meta_role_A_as_user
    else:
        meta = meta_role_B_as_user

    response_stream = get_characterglm_response(history_tmp, meta=meta)
    bot_response = output_stream_response(response_stream, placeholder)
    print(bot_response)
    return bot_response


# 交替生成对话数据并展示
if generate_chat_record:
    print(f"交替生成对话 = {st.session_state['meta']}")
    # 无历史对话时，角色A开场
    if len(filter_text_msg(st.session_state["history"])) < 1:
        start_message = "开始对话吧"
        st.session_state["history"].append({"role": st.session_state["meta"]['role_A_name'], "content": start_message})
    # 生成多轮对话
    for _ in range(10):
        # 根据角色A的最后一次发言,角色B回复
        with st.chat_message(name=st.session_state["meta"]['role_B_name']):
            role_B_message_placeholder = st.empty()
        role_B_response = start_chat(st.session_state["meta"]['role_A_name'], role_B_message_placeholder)
        if not role_B_response:
            st.markdown("生成出错")
            break
        else:
            st.session_state["history"].append(TextMsg({"role": st.session_state["meta"]['role_B_name'], "content": role_B_response}))

        # time.sleep(0.5)

        # 根据角色B的最后一次发言,角色A回复
        with st.chat_message(name=st.session_state["meta"]['role_A_name']):
            role_A_message_placeholder = st.empty()
        role_A_response = start_chat(st.session_state["meta"]['role_B_name'], role_A_message_placeholder)
        if not role_A_response:
            st.markdown("生成出错")
            break
        else:
            st.session_state["history"].append(TextMsg({"role": st.session_state["meta"]['role_A_name'], "content": role_A_response}))


# 保存对话数据按钮
if len(filter_text_msg(st.session_state["history"])) > 0:
    with st.container():
        with button2_key_to_col["download_chat_record"]:
            download_chat_record = st.download_button(
                label=button2_labels["download_chat_record"],
                key="download_chat_record",
                data=('\n'.join([f'{msg["role"]}: {msg["content"]}' for msg in filter_text_msg(st.session_state['history'])])),
                file_name='chat_history.txt',  # 下载文件的名称
                mime='text/plain', # MIME 类型
                icon=":material/book:",
                type="secondary"
            )