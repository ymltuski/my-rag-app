from zhipuai import ZhipuAI

class ZhipuAIEmbeddings:
    """`Zhipuai Embeddings` embedding models."""
    def __init__(self, api_key):
        """
        实例化 ZhipuAI 为 self.client

        Args:
            api_key (str): 智谱 AI 的 API Key.
        """
        self.client = ZhipuAI(api_key=api_key)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (list[str]): 要生成 embedding 的文本列表.

        Returns:
            list[list[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model="embedding-3",
                input=texts[i:i+64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result
    
    def embed_query(self, text: str) -> list[float]:
        """
        生成输入文本的 embedding.

        Args:
            text (str): 要生成 embedding 的文本.

        Return:
            embeddings (list[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        return self.embed_documents([text])[0]
