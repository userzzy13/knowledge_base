
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from torch.ao.quantization.fx.utils import return_arg_list

from Config.llm_config import llm_config
from Config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, \
    ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_processor.state import QueryGraphState
from tool.json_format_utils import format_json
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_request, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        #1. 校验参数
        session_id,original_query = self._step_1_validate_param(state)
        logger.info("校验通过")
        #2. 获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤二：获取到：{len(history)}条历史记录")
        #更新历史会话记录
        state["history"] = history

        #3. 用户初始消息保存,返回保存的消息ID
        message_id = save_chat_message(session_id,"user",original_query)
        logger.info(f"用户初始消息记录已保存,ID:{message_id}")


        #4. 提取信息
        extract_res = self._step_4_extract_info(original_query,history)#提取和生成了商品名和重写的问题
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query",original_query)
        #更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        #5,6. 如果提取到商品名则进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names) #将商品名向量化
            align_result = self._step_6_align_item_names(query_results) #将向量化的商品名进行对齐操作
        else:
            logger.info("未找到商品名，跳过检索")

        #7. 检查确认状态
        state = self._step_7_check_confirmation(state,align_result,history)

        #8. 写入最终历史
        self._step_8_write_history(state,session_id,rewritten_query,message_id)



        return state

    def _step_1_validate_param(self,state: QueryGraphState)->tuple[str,str]:

        session_id = state.get("session_id")
        if session_id is None:
            raise ValueError("核心参数session_id缺失")

        original_query = state["original_query"]
        if original_query is None:
            raise ValueError("核心参数original_query缺失")

        return session_id,original_query

    def _step_4_extract_info(self,query,history)->dict:
        """
        提取信息，重写用户提问的问题，提取用户提及的商品名称
        :param query:
        :param history:
        :return:
        """
        try:
            #1. 获取llm客户端
            chat_model = ChatOpenAI(
                model = llm_config.item_model,
                api_key = llm_config.api_key,
                base_url = llm_config.base_url,
                temperature=llm_config.llm_temperature,
                #json格式输出
                model_kwargs={
                    "response_format": {"type":"json_object"}
                }
            )

            #2. (多条)构造历史对话文本，对话为用户：内容
            history_text = ""
            for msg in history:
                role = msg["role"]
                text = msg["text"]
                history_text+= f"{role}:{text}\n"

            #3. 处理和拼接提示词
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query,
            )

            #4. 构造llm调用的消息队列
            message = [
                SystemMessage(ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(user_prompt)
            ]

            #5. 调用llm
            response = chat_model.invoke(message)
            content = response.content

            #6. 数据清洗
            if content.startswith("'''json"):
                content = content.replace("'''json","").replace("'''","")

            #7. 数据解析:json->dict
            result = json.loads(content)

            #8. 确保字段存在
            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            #9. 去空格
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]


            return result
        except Exception as e:
            logger.error(f"提取信息失败，大模型异常：{e}")
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self,item_names)->list[dict]:
        #1.初始化一个列表，用于存放处理好的向量数据
        results = []

        #2. 获取向量库的客户端
        client = get_milvus_client()

        #3. 校验是否连接成功
        if not client:
            logger.error("向量库连接失败")
            return results

        #4. 从环境变量中提取milvus中的集合名
        collection_name = milvus_config.item_name_collection

        #5. 对所有商品名生成向量
        #embeddings的格式为dict[str:list]
        embeddings = generate_embeddings(item_names)

        #6. 遍历每个商品名，进行检索
        for i in range(len(item_names)):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                #构造请求
                reqs = create_hybrid_search_request(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                #执行搜索
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8,0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                #存储匹配到的结果
                matches = []
                # 校验搜索结果是否有效（非空且包含数据，适配Milvus批量搜索格式）
                if search_res and len(search_res) > 0:
                    # 遍历当前商品名的Top5匹配结果（search_res[0]为该商品的独立搜索结果集）
                    for hit in search_res[0]:
                        # 提取匹配结果中的商品名和评分，做防KeyError处理（设置默认空字典）
                        # hit格式：{"id": 数据库ID, "distance": 相似度评分, "entity": {"item_name": "标准化商品名"}}
                        matches.append(
                            {
                                "item_name": hit.get("entity", {}).get("item_name"),  # 数据库标准化商品名
                                "score": hit.get("distance"),  # 0-1相似度评分
                            }
                        )

                # 将当前商品名的原始名称+匹配结果，封装后加入最终结果列表
                results.append({
                    "extracted_name": item_names[i],  # step4提取的原始商品名称
                    "matches": matches  # 该商品名的Top5匹配结果（含评分）
                })

            # 捕获单个商品名处理的异常（不中断其他商品名执行），仅记录错误日志
            except Exception as e:
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")

        # 返回所有商品名的向量化+搜索结果列表

        return results





    def _step_6_align_item_names(self, query_results:list[dict])->dict:

        #1. 初始化 确认商品名的列表
        confirmed_item_names = []

        #2. 初始化 候选商品名的列表
        option = []

        for res in query_results:
            #获取提取的原始商品名并格式化
            extracted_name = (res.get("extracted_name","") or "").strip()
            #获取搜索后得到的商品名
            matches = res.get("matches", []) or []
            if not matches:
                continue

            high = [m for m in matches if m.get("score")>=0.8]
            mid = [m for m in matches if 0.8 > m.get("score") >= 0.6]

            if len(high) > 0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                continue

            if len(mid) > 0:
                for m in mid[:3]:
                    option.append(m.get("item_name"))



        return {
            "confirmed_item_names": list(set(confirmed_item_names)),#确认的商品名
            "option": list(set(option))#待确认的商品名
        }

    #检查确认状态
    def _step_7_check_confirmation(self,state,alian_result,history):
        confirmed = alian_result.get("confirmed_item_names",[])
        options = alian_result.get("option",[])
        #分支A
        if confirmed:
            #收集历史记录中未确认商品的消息ID
            ids_to_update = []
            for msg in history:
                if not msg.get("item_name"):#如果历史记录没有商品名
                    mid = msg.get("_id")  #则获取消息id
                    if mid:
                        ids_to_update.append(str(mid)) #把id收集起来
            #如果有从历史记录中收集到的未确认商品名的消息id，则更新
            if ids_to_update:
                update_message_item_names(ids_to_update,confirmed)

            #更新完返回并改写state
            state["item_names"] = confirmed
            state["answer"] = ""

            return state


        #分支B
        if options:
            #拼接字符串
            options_str = ",".join(options)
            #构造向用户确认的提示词
            answer = f"您想问的是以下哪个产品：{options_str}？请明确型号"

            state["answer"] = answer
            state["item_names"] = []
            return state

        #分支C
        state["item_names"] = []
        state["answer"] = "抱歉，并没有找到符合条件的产品"
        return state

    #写入历史记录
    def _step_8_write_history(self,state,session_id,rewritten_query,message_id):

        if state.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state["answer"],
                rewritten_query="",
                item_names=state.get("item_names",""),
            )

        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )


        return state




if __name__ == "__main__":

    # 初始化图状态
    # "HAK 180 烫金机怎么用？"
    # "怎么用呢？"
    init_state = {
        "session_id": "test_001",
        "original_query": "你好，我是你爹"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(format_json(result))