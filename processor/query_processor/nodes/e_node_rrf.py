from typing import List, Tuple, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        #获取各路的结果，除了网络搜索
        embedding_search_list = [
            doc.get('entity') for doc in (state.get('embedding_chunks') or []) if isinstance(doc,dict)
        ]

        hyde_embedding_search_list = [
            doc.get('entity') for doc in (state.get('hyde_embedding_chunks') or []) if isinstance(doc, dict)
        ]

        #为不同路的结果设置不同的权值
        rrf_input = [
            (embedding_search_list,1.0),
            (hyde_embedding_search_list,1.0)
        ]

        #用rrf计算chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_input)

        #获取文档
        rrf_chunks = [doc for doc,_ in rrf_merge_results]

        #更新state
        state['rrf_chunks'] = rrf_chunks

        return state



    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分
        :param rrf_inputs:  列表，每个元素是(各路的搜索结果列表, 权重)的元组
        :param k:           平滑参数(RFF常数)，通常取 60
        :param max_results: 合并完之后返回的文档数，None 表示全部
        :return:            合并以及排序后的文档列表，[(元素, RRF 得分), ...] 按得分降序
        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}  # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get('chunk_id')
                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)

                # 使用 setdefault 保留首次遇到的文档版本(只记录第一次)
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序
        unsorted_results = [(chunk_data[cid], score) for cid, score in chunk_scores.items()]
        sorted_results = sorted(
            unsorted_results,
            # 排序时看每个元素的第 2 个值（也就是分数）
            key=lambda x: x[1],
            reverse=True
        )
        # 等价写法
        # def get_score(item):
        #     return item[1]  # 返回分数
        # sorted_results = sorted(unsorted_results, key=get_score, reverse=True)

        # 动态截取前 max_results 条
        return sorted_results[:max_results] if max_results else sorted_results