from langchain_openai import ChatOpenAI

from Config.llm_config import llm_config
#全局缓存
_llm_client_cache = {}

def get_llm_client(model:str | None = None,json_model:bool = False):

    m = model or llm_config.llm_model

    key = (m,json_model,)
    #判断是否击中缓存，如果击中，直接返回缓存中的value
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    #没击中则添加模型实例
    client = ChatOpenAI(
        model=m,
        temperature=llm_config.llm_temperature,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )

    _llm_client_cache[key] = client
    return client

if __name__ == "__main__":
    client = get_llm_client()
    invoke = client.invoke("你好千问")
    print(invoke.content)