import logging
import os
from datetime import datetime
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient, ASCENDING


class HistoryMongoTool:
    """
    mongoDB获取历史记录的工具类
    核心功能：封装mongdoDB连接工具，集合初始化，索引创建
    """
    def __init__(self):
        try:
            #获取mongoDB的URL
            self.mongo_url = os.getenv("MONGO_URL")
            #获取数据库名称
            self.db_name = os.getenv("MONGO_DB_NAME")
            #创建实例,返回一个mongoDB对象
            self.client = MongoClient(self.mongo_url)
            #获取指定名称的数据库
            self.db = self.client[self.db_name]
            #获取数据库的集合
            self.chat_message = self.db["chat_message"]
            #创建索引
            self.chat_message.create_index([("session_id",1),("ts",-1)])
            logging.info(f"成功连接数据库：{self.db_name}")
        except Exception as e:
            logging.error(f"连接失败：{e}")
            raise

#定义一个全局变量用于存放mongoDB实例,提前创建实例，防止后续调用时要重新调用，浪费时间
_history_mongo_tool = HistoryMongoTool()

def get_history_mongo_tool():

    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()

    return _history_mongo_tool








#未优化
def clear_history(session_id):
    mongo_tool = get_history_mongo_tool()
    result = mongo_tool.chat_message.delete_many({"session_id":session_id})
    logging.info(f"已删除{result.deleted_count}数据")
    return result.deleted_count




    return ""


def save_chat_message(
        session_id:str,
        role:str,
        text:str,
        rewritten_query:str="",
        item_names:list = None,
        image_urls:list = None,
        message_id:str = None,
):
    """
    保存更新消息内容
    :param session_id:会话id
    :param role: 角色
    :param text: 消息内容
    :param rewritten_query:重写的问题
    :param item_names: 商品名
    :param image_urls: 图片链接
    :param message_id: 消息id，有则更新，没有则新增
    :return:
    """
    #生成时间戳，方便查询
    ts = datetime.now().timestamp()

    #构造要传入的文档格式
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "image_urls": image_urls,
        "message_id": message_id,
        "ts": ts
    }

    #获取mongoDB实例
    mongo_tool = get_history_mongo_tool()
    #判断是否传入主键message_id，有主键则更新，没有主键则新增
    if message_id:
        result = mongo_tool.chat_message.update_one({"_id": ObjectId(message_id)}, {"$set": document})
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)












def update_message_item_names(ids,item_names)->int:

    mongo_tool = get_history_mongo_tool()

    object_ids = [ObjectId(i) for i in ids]

    result = mongo_tool.chat_message.update_one(
        {"_id": object_ids},
        {"$set": {"item_names": item_names}},
    )

    return result.modified_count















def get_recent_messages(session_id:str,limit:int = 10)->list[dict[str,Any]]:
    """
    获取最近的对话，初始默认是拿最近10条记录
    :param session_id:
    :param limit:
    :return:
    """
    try:
        #获取实例
        mongo_tool = get_history_mongo_tool()

        #构造查询条件
        query = {"session_id": session_id}

        #开始查询
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)

        #将查询结果给message
        message = list(cursor)
        return message
    except Exception as e:
        logging.error(f"获取最近消息失败：{e}")
        return []



if __name__ == '__main__':

    pass
