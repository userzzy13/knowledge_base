from Config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import create_hybrid_search_request, get_milvus_client, hybrid_search


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: QueryGraphState) -> QueryGraphState:
         """
         节点逻辑
         :param state: 工作流状态对象
         :return: 更新后的状态对象
         """
         try:
            #获取改写的问题和确认的商品名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            #生成向量
            embedding = generate_embeddings([query])
            dense_vector = embedding.get("dense_vector")[0]
            sparse_vector = embedding.get("sparse_vector")[0]

            #获取milvus的集合（表名）
            collection_name = milvus_config.chunks_collection

            #格式化，处理格式垃圾
            expr = None  #额外参数（过滤作用）
            if item_names:
                expr = f'item_name in {item_names}'
            else:
                logger.info("没有商品名用于过滤")


            #构造请求对象
            reqs = create_hybrid_search_request(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=10
            )

            #执行检索
            logger.info("开始检索。。。。。。。。。。。。。。。。。。。。。。。。。。。")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8,0.2),
                output_fields=["chunk_id","content","item_name"]
            )

            #构造并返回结果
            return {"embedding_chunks":res[0] if res else []}
         except Exception as e:
             logger.exception(f"向量搜索失败，{e}")
             return {}
