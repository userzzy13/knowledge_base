# processor/import_processor/nodes/node_bge_embedding.py
import json
import logging
from pathlib import Path

from typing import List, Dict

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def _backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:


        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path("D:/mineru_poll") / state.get("file_title") / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2  # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程

            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)


    def process(self, state: ImportGraphState):


        #步骤一：数据校验
        chunks = self._step_1_validate_input(state)

        #步骤二：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        self._backup(state, output_data)
        #步骤三：更新全局状态，将带向量的chunks回传
        state['chunks'] = output_data




        return state

    #步骤一
    def _step_1_validate_input(self, state: ImportGraphState):
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks


    #步骤二
    def _step_2_generate_embeddings(self, chunks:list[dict[str,str]]):
        # 初始化空列表，存储最终带向量的文本切片
        output_data = []
        #处理批次，防止一次处理过多数据
        batch_size = 5

        for i in range(0,len(chunks),batch_size):
            batch_texts = chunks[i:i+batch_size]#i的取值是0，5，10，15，分批处理文本
            input_texts = []
            for doc in batch_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                input_texts.append(f"{item_name}/n{content}"if item_name else content)
            #获取完批次的数据，将数据转成向量
            docs_embeddings = generate_embeddings(input_texts)
            #拿出docs_embeddings中的稠密和稀疏向量赋值给output_data
            for j,doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)
        self.logger.info(f"成功获取第 {i + 1}-{min(i + len(batch_texts), len(chunks))} 项的嵌入。")
        #返回带向量的chunks
        return output_data


if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\mineru_poll\H3C\chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state,
        "file_title": "H3C"
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))


