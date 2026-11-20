from Config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_request, get_milvus_client, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        #提取用户的问题和确认的商品名
        rewritten_query = state.get("rewritten_query")
        item_name = state.get("item_names")

        try:
            #生成假设文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            #用(重写的问题+假设文档)去检索切片,(重写的问题+假设文档)当做参数传进
            res = self._step_2_search_embedding_hyde(rewritten_query,hyde_doc,item_name)


            #返回结果进行封装
            return {"hyde_embedding_chunks": res,
                    "hyde_doc":hyde_doc,}


        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败：{e}")
            return {"hyde_embedding_chunks": []}


    #生成假设性文档
    def _step_1_create_hyde_doc(self,rewritten_query):
        logger.info("开始生成假设性文档")
        try:
            #调用大模型生成假设性文档
            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt)
            return hyde_doc

        except Exception as e:
            logger.exception(f"生成假设性文档失败：{e}")
            return e




    #根据假设性文档进行检索
    def _step_2_search_embedding_hyde(self,rewritten_query,hyde_doc,item_names):

        try:
            #将改写的问题和生成的假设性文档进行拼接
            combined_text = rewritten_query+""+hyde_doc

            #生成向量
            embeddings = generate_embeddings([combined_text])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            #获取milvus集合名
            collection_name = milvus_config.chunks_collection

            # 处理 item_names 中的引号，防止注入或语法错误
            expr = None
            if item_names:
                # quoted = ", ".join(f'"{v}"' for v in item_names)
                # expr = f"item_name in [{quoted}]"
                # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
                expr = f'item_name in {item_names}'
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")

            # 构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_request(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                expr=expr,
                limit=10  # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            )

            # 执行混合向量检索
            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name"],
            )

            return res[0] if res else []

        except Exception as e:
            logger.error(f"假设性文档向量检索异常：{e}")
            raise e
