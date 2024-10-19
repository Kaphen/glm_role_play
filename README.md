# glm_role_play

支持设置两个角色的人设，自动生成两个角色的对话。还支持保存对话数据到本地文件、根据不同风格生成人像图片和场景图片

## 效果
### 1. 自动生成两个角色的对话
<div align="center">
<img src=resources/generate_chat.gif width="80%"/>
</div>

### 2. 保存对话数据到本地文件
<div align="center">
<img src=resources/save_chat_history.gif width="80%"/>
</div>

### 3. 根据不同风格生成人像图片和场景图片
<div align="center">
<img src=resources/picture_sytle.gif width="80%"/>
</div>

## 运行方式
### 步骤一：设置智普大模型API KEY
打开characterglm_api_demo_streamlit.py， 在第69行设置api key
```python
API_KEY: str = "xxxxx.xxxx"
```
### 步骤二：运行
```shell
pip install -r requirements.txt
streamlit run characterglm_api_demo_streamlit.py
```