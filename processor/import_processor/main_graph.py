import json
import logging

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from state import ImportGraphState


class KBImportWorkFlow:
    def __init__(self,config = None):
        self._compiled_graph = None


    @property
    def graph(self):

        """
        返回图实例

        """
        logging.info("获取图实例")
        if self._compiled_graph is None:
             self._compiled_graph = self.build_graph() #创建图实例
        return self._compiled_graph


    @staticmethod
    def route_after_entry(state: ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "pdf"
        elif state.get("is_md_read_enabled"):
            return "md"
        else:
            logging.info("没有对应的格式，退出")
            return END



    def build_graph(self):
        """
        创建一个主图

        """
        graph = StateGraph(ImportGraphState)

        #1.注册节点
        graph.add_node("node_entry",NodeEntry())
        graph.add_node("node_pdf_to_md",NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())





        #2.节点边
        graph.set_entry_point("node_entry")
        ##条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "pdf":"node_pdf_to_md",
                "md":"node_md_img",
                END:END
            }
        )

        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)





        #3.编译图

        graph_compile = graph.compile()

        return graph_compile


    def run(self,state: ImportGraphState,stream:bool = False):
        if stream:
            return self.graph.stream(state,stream_mode="values")
        else:
            return self.graph.invoke(state)



if  __name__ == "__main__":
    setup_logging()
    workflow = KBImportWorkFlow()
    #graph = workflow.graph
    init_state = {
        "import_file_path" : r"D:\Anaconda3\pkgs\panel-1.5.2-py312haa95532_0\Lib\site-packages\panel\tests\test_data\sample.pdf"
    }
    for event in workflow.run(init_state,stream=True):
        print(f"state:{event}")

    #graph = workflow.run(init_state,stream=False)
    #print(json.dumps(graph,ensure_ascii=False,indent=4))
