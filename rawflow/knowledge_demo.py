import os.path

from ragflow_sdk import RAGFlow 

try:
    
    from rawflow.rag_config import _load_ragflow_env
except ModuleNotFoundError as exc:
    if exc.name != "rawflow":
        raise
    
    from rag_config import _load_ragflow_env

api_key,base_url =_load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key,base_url=base_url)

def create_knowledge_base(knowledge_base_name, description):
    """
    创建知识库，有个名字和描述!
    名字和描述一定要准确的写！agent调用哪个聊天助手看聊天助手的描述和他对应知识库的描述！
    :param knowledge_base_name: 名字
    :param description: 描述
    :return:
    """
    ds = ragflow_client.create_dataset(name=knowledge_base_name, description=description,embedding_model="text-embedding-v3@Tongyi-Qianwen")
    return ds.id

def upload_file_to_knowledge_base(kb_id, file_paths):
    """
    向知识库上传文件！ 文件可以是多个！！
    """

    datasets = ragflow_client.list_datasets(id=kb_id,page=1,page_size=10)
    dataset = datasets[0]
    
    document_list = []  
    for file_path in file_paths:  
        file_name = os.path.basename(file_path)  
        with open(file_path, 'rb') as f:  
            blob = f.read()  
            document_list.append({  
                "display_name": file_name,  
                "name": file_name,  
                "blob": blob  
            })
    
    dataset.upload_documents(document_list) 
