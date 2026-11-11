# processor/import_processor/nodes/node_md_img.py
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Deque

from minio import Minio

from Config import minio_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
import base64
from utils.llm_utils import get_llm_client
from Config.llm_config import llm_config
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):

        logging.info(f"{self.name}开始执行.....")
        #1.参数处理
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)



        #2.图片扫描
        target_images = self._step_2_scan_images(md_content, images_dir)#返回一个元组列表[(str,str,(str,str))]


        #3.视觉模型摘要（图片总结）
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)


        #4.上传minio，替换md
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)


        #5.保存备份md
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state




         #步骤1
    def _step_1_get_content(self,state: ImportGraphState):
        #校验
        md_path = state['md_path']
        if not md_path:
            raise StateFieldError(field_name='md_path',expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{md_path_obj}")
        md_content = state['md_content']
        images_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir
    #步骤二
    def _step_2_scan_images(self,md_content,images_dir):
        #返回结果
        target_images = []
        #扫描图片
        # os.listdir()拿出目录下文件的名字
        for image_file in os.listdir(images_dir):
            #splitext()拆分后缀
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持:{image_file}")
                continue
            img_path = images_dir / image_file
            #查找图片在引用中的上下文
            context = self._find_image_in_md(md_content, image_file)
            target_images.append((image_file,img_path,context))
        return target_images



        #步骤二方法一
    def _find_image_in_md(self,md_content:str,image_file:str,context_len:int = 100) -> Tuple[str, str]:
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        #找到对应的图片，返回一个对象，后续调用span函数方法
        match = pattern.search(md_content)
        if not match:
            return None
        #获取图片的前后两个字符位置start , end
        start , end = match.span()
        pre_text = md_content[max(0,start-context_len):start]
        post_text = md_content[end:min(len(md_content),end+context_len)]
        return pre_text , post_text





    def _step_3_generate_summaries(self,md_path_obj_stem:str, target_images:list[tuple[str,str,tuple[str,str]]]) -> dict[str, str]:
        """
        :param self:
        :param md_path_obj_stem:
        :param target_images:
        :return:
        """
        summaries = {}
        request_deque = deque()
        # 调用视觉模型
        for img_file ,img_path,context in target_images:
            self._apply_api_rate_limit(request_deque, max_requests=10)#时间窗口限流
            #调用大模型
            summaries[img_file] = self._summarize_image(img_path, root_folder=md_path_obj_stem, image_content=context)
        return summaries






    #步骤三方法一：时间窗口限流
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()
        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()
        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()
        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")





    #步骤三方法2（调用多模态大模型（如 Qwen-VL）来生成图片的中文摘要，作为 Markdown 图片的 Alt Text）
    def _summarize_image(self,img_path:str, root_folder : str, image_content : Tuple[str,str]) -> str:
        with open(img_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        try:
            chat_model = get_llm_client(llm_config.vl_model)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n", "")
        except Exception as e:
            self.logger.error(f"图像总结失败：{img_path}, 错误{e}")
            return "图片描述"




    def _step_4_upload_and_replace(self,md_path_obj_stem:str,target_images:list[tuple[str,str,tuple[str,str]]],summaries:dict[str, str],md_content:str):

        #0 获取minio客户端
        minio_client = get_minio_client()
        #构造上传目录
        minio_img_dir = minio_config.img_dir
        upload_dir = f"{minio_img_dir}/{md_path_obj_stem}"


        #1 清理minio目录
        self._clean_minio_directory(minio_client, upload_dir)

        #2 批量上传图片，获取minio的url

        urls = self.upload_images_batch(minio_client,upload_dir,target_images)

        #3 将摘要和url合并

        image_info = self._merge_summary_and_url(summaries, urls)


        #4 替换md文件的摘要和路径
        md_content = self._process_md_file(md_content, image_info)



        return md_content

    def _clean_minio_directory(minio_client:Minio, upload_dir:str):
        pass

    def upload_images_batch(minio_client, upload_dir, target_images):
        pass