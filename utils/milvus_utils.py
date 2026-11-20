import os
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

from Config.milvus_config import milvus_config
from tool.logger import logger

_milvus_client = None
def get_milvus_client():
    """
    获取全局单例的MilvusClient对象
    :return: MilvusClient实例
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(uri=milvus_config.milvus_url)

    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value



#准备搜索请求
def create_hybrid_search_request(dense_vector,sparse_vector,dense_params=None,sparse_params=None,expr=None,limit=5)->list:
    if dense_params is None:
        dense_params = {"metric_type":"COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type":"IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )
    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]



#执行搜索
def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5,0.5),
        norm_score=False,
        limit =5,
        output_fields=None,
        search_params=None
):
    """

    :param client: milvus实例
    :param collection_name: 集合名（表名）
    :param reqs: 搜索的请求列表，[dense_req,sparse_req],create_hybrid_search_request返回的值
    :param ranker_weights: 权重
    :param norm_score: 归一化
    :param limit: 限制拿多少条
    :param output_fields:需要返回的字段
    :param search_params: 搜索参数
    :return: 列表
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None