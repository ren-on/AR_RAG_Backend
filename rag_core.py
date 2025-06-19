
"""
應收款RAG系統核心模組 - 基於LangChain和FAISS的檢索增強生成系統
用於處理公司應收款和未收款項的查詢，提取備註中的原因，分析風險和異常指標
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from datetime import datetime, timedelta
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAI, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.embeddings import Embeddings
import zhconv  # 用於繁簡體轉換
import traceback


load_dotenv()


class SentenceTransformerEmbeddings(Embeddings):
    """包裝SentenceTransformer使其兼容LangChain的Embeddings接口"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0] # Ensure text is a list for encode
        return embedding.tolist()


class CompanyNameMatcher:
    """公司名稱精確匹配器，用於解決相似但不同公司名稱的識別問題"""
    
    def __init__(self, company_data: List[str], embedding_model):
        self.company_data = company_data
        self.model = embedding_model
        self.preprocess_companies()
        
    def preprocess_company_name(self, name: str) -> str:
        name = re.sub(r"\s+", "", name)
        name = re.sub(r"[^\w\u4e00-\u9fff]", "", name)
        return name.lower()
    
    def decompose_company_name(self, name: str) -> Dict[str, str]:
        locations = ["赣州", "江西", "珠海", "惠州", "苏州", "江门", "福建", "北京", "上海", "广州", "深圳"]
        location = None
        for loc in locations:
            if loc in name:
                location = loc
                break
        if location:
            main_name = name.replace(location, "")
        else:
            main_name = name
        return {"location": location, "main_name": main_name}
    
    def preprocess_companies(self):
        self.processed_names = {self.preprocess_company_name(name): name for name in self.company_data}
        self.decomposed_names = {name: self.decompose_company_name(name) for name in self.company_data}
        
        # 創建簡稱映射表，用於匹配簡稱
        self.short_names = {}
        for full_name in self.company_data:
            # 提取公司主體名稱（不含地區和"有限公司"等後綴）
            main_part = re.sub(r"(有限公司|股份有限公司|科技有限公司|电子科技有限公司|电路有限公司|材料股份有限公司)", "", full_name)
            if len(main_part) >= 2:  # 確保簡稱至少有2個字符
                self.short_names[main_part] = full_name
        
        if hasattr(self.model, "encode"):
            self.company_embeddings = self.model.encode(list(self.company_data))
        elif hasattr(self.model, "embed_documents"):
            self.company_embeddings = np.array(self.model.embed_documents(list(self.company_data)))
        else:
            raise ValueError("不支持的嵌入模型類型")
        self.dimension = self.company_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.company_embeddings)
        
    def match(self, query_name: str, threshold: float = 0.85) -> List[Tuple[str, float]]:
        processed_query = self.preprocess_company_name(query_name)
        
        # 1. 精確匹配完整名稱
        if processed_query in self.processed_names:
            return [(self.processed_names[processed_query], 1.0)]
        
        # 2. 匹配簡稱
        for short_name, full_name in self.short_names.items():
            if short_name in query_name:
                return [(full_name, 0.95)]
        
        # 3. 分解匹配（地區+主體名稱）
        decomposed_query = self.decompose_company_name(query_name)
        
        # 4. 向量相似度匹配
        if hasattr(self.model, "encode"):
            query_embedding = self.model.encode([query_name])
        elif hasattr(self.model, "embed_documents"):
            query_embedding = np.array(self.model.embed_documents([query_name]))
        else:
            raise ValueError("不支持的嵌入模型類型")
        distances, indices = self.index.search(query_embedding, len(self.company_data))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.company_data):
                distance = distances[0][i]
                max_distance = np.max(distances)
                if max_distance > 0:
                    normalized_distance = distance / max_distance
                else:
                    normalized_distance = 0
                similarity = 1 - normalized_distance
                if similarity >= threshold:
                    results.append((self.company_data[idx], similarity))
        return results


class ARDataProcessor:
    """應收款數據處理器，用於處理Excel數據並準備RAG系統"""
    
    def __init__(self, excel_path: str, embedding_model_name: str = "shibing624/text2vec-base-chinese"):
        self.excel_path = excel_path
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformerEmbeddings(embedding_model_name)
        self.st_model = SentenceTransformer(embedding_model_name)
        self.data = None
        self.company_matcher = None
        self.vector_store = None
        self.age_ranges = None
        self.load_data()
        self.define_age_ranges()
        
    def load_data(self):
        try:
            self.data = pd.read_excel(self.excel_path)
            print(f"成功加載數據，共 {len(self.data)} 行")
            if "客户名称" in self.data.columns:
                company_names = self.data["客户名称"].dropna().unique().tolist()
                self.company_matcher = CompanyNameMatcher(company_names, self.st_model)
                print(f"初始化公司名稱匹配器，共 {len(company_names)} 個公司")
            else:
                print("警告: 未找到客户名称列")
        except Exception as e:
            print(f"加載數據時出錯: {e}")
            raise
    
    def define_age_ranges(self):
        self.age_ranges = [
            {"column": "0-30天", "min_days": 0, "max_days": 30},
            {"column": "31~60天", "min_days": 31, "max_days": 60},
            {"column": "61~180天", "min_days": 61, "max_days": 180},
            {"column": "181~360天", "min_days": 181, "max_days": 360},
            {"column": "361天以上", "min_days": 361, "max_days": float("inf")}
        ]
        for age_range in self.age_ranges:
            column = age_range['column']
            if column not in self.data.columns:
                print(f"警告: 未找到賬齡段列 {column}")
    
    def preprocess_data(self):
        documents = []
        for idx, row in self.data.iterrows():
            company_name = row.get("客户名称", "")
            if not company_name or pd.isna(company_name):
                continue
            amount_info = {}
            for col in self.data.columns:
                if col not in ["客户名称", "备注", "异常指标"] and pd.notna(row[col]) and isinstance(row[col], (int, float)):
                    amount_info[col] = row[col]
            remarks = row.get("备注", "")
            if pd.isna(remarks):
                remarks = ""
            abnormal = row.get("异常指标", "")
            if pd.isna(abnormal):
                abnormal = "正常"
            content = f"公司名稱: {company_name}\n"
            content += f"應收金額: {amount_info.get('应收金额', 'N/A')}\n"
            content += f"到期-应收未收 合计: {amount_info.get('到期-应收未收 合计', 'N/A')}\n"
            for col, value in amount_info.items():
                if col not in ["应收金额", "到期-应收未收 合计"]:
                    content += f"{col}: {value}\n"
            content += f"異常指標: {abnormal}\n"
            content += f"備註: {remarks}\n"
            doc = Document(
                page_content=content,
                metadata={
                    "company_name": company_name,
                    "amounts": amount_info,
                    "remarks": remarks,
                    "abnormal": abnormal,
                    "row_idx": idx
                }
            )
            documents.append(doc)
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        split_docs = text_splitter.split_documents(documents)
        print(f"預處理完成，生成了 {len(split_docs)} 個文檔片段")
        return split_docs
    
    def create_vector_store(self):
        docs = self.preprocess_data()
        self.vector_store = LangchainFAISS.from_documents(docs, self.embedding_model)
        print("向量存儲創建完成")
        return self.vector_store
    
    def _extract_reason_from_remarks(self, remarks: str) -> str:
        if not remarks or pd.isna(remarks):
            return ""
        reason_keywords = ["原因", "未付款", "未收款", "延期", "拖欠", "欠款", "未结算", "未结清", "未支付", "未到账", "未回款"]
        sentences = re.split(r"[；;。\n]", remarks)
        reason_sentences = [s.strip() for s in sentences if any(keyword in s for keyword in reason_keywords)]
        return "；".join(reason_sentences) if reason_sentences else remarks
    
    def get_company_data(self, company_name: str) -> Dict[str, Any]:
        matches = self.company_matcher.match(company_name)
        if not matches:
            return {"error": f"未找到匹配的公司: {company_name}"}
        best_match, score = matches[0]
        company_data_row = self.data[self.data["客户名称"] == best_match]
        if len(company_data_row) == 0:
            return {"error": f"找到公司名稱 {best_match}，但無對應數據"}
        row = company_data_row.iloc[0]
        result = {
            "company_name": best_match,
            "match_score": score,
            "amounts": {col: row[col] for col in self.data.columns if col not in ["客户名称", "备注", "异常指标"] and pd.notna(row[col]) and isinstance(row[col], (int, float))},
            "remarks": row.get("备注", ""),
            "abnormal": row.get("异常指标", ""),
            "payment_reason": self._extract_reason_from_remarks(row.get("备注", ""))
        }
        return result

    def get_general_overdue_data(self) -> List[Dict[str, Any]]:
        results = []
        for idx, row in self.data.iterrows():
            company_name = row.get("客户名称", "")
            if pd.isna(company_name):
                continue
            total_overdue_amount = row.get("到期-应收未收 合计", 0)
            if pd.isna(total_overdue_amount):
                total_overdue_amount = 0
            
            if total_overdue_amount > 0:
                remarks = row.get("备注", "")
                payment_reason = self._extract_reason_from_remarks(remarks)
                results.append({
                    "company_name": company_name,
                    "overdue_amount": total_overdue_amount,
                    "payment_reason": payment_reason,
                    "abnormal": row.get("异常指标", ""),
                    "total_receivable": row.get("应收金额", 0) if pd.notna(row.get("应收金额", 0)) else 0
                })
        results = sorted(results, key=lambda x: x["overdue_amount"], reverse=True)
        return results

    def get_receivable_data(self) -> List[Dict[str, Any]]:
        results = []
        for idx, row in self.data.iterrows():
            company_name = row.get("客户名称", "")
            if pd.isna(company_name):
                continue
            receivable_amount = row.get("应收金额", 0)
            if pd.isna(receivable_amount):
                receivable_amount = 0
            
            if receivable_amount > 0:
                remarks = row.get("备注", "")
                payment_reason = self._extract_reason_from_remarks(remarks)
                results.append({
                    "company_name": company_name,
                    "receivable_amount": receivable_amount,
                    "overdue_amount": row.get("到期-应收未收 合计", 0) if pd.notna(row.get("到期-应收未收 合计", 0)) else 0,
                    "payment_reason": payment_reason,
                    "abnormal": row.get("异常指标", "")
                })
        results = sorted(results, key=lambda x: x["receivable_amount"], reverse=True)
        return results

    def get_overdue_data_by_specific_days(self, min_days: int) -> List[Dict[str, Any]]:
        results = []
        applicable_age_ranges = []
        for age_range in self.age_ranges:
            if age_range["min_days"] >= min_days:
                applicable_age_ranges.append(age_range)
        if not applicable_age_ranges:
            print(f"警告: 未找到適用於 {min_days} 天的賬齡段")
            return []
        print(f"適用於 {min_days} 天的賬齡段: {[ar['column'] for ar in applicable_age_ranges]}")
        
        # 遍歷每一行數據
        for idx, row in self.data.iterrows():
            company_name = row.get("客户名称", "")
            if pd.isna(company_name):
                continue
            
            # 計算特定天數的未收款金額
            specific_overdue_amount = 0
            overdue_details = {}
            
            # 只累加適用的賬齡段
            for age_range in applicable_age_ranges:
                col = age_range['column']
                if col in self.data.columns:
                    col_value = row[col]
                    if pd.notna(col_value):  # 確保值不是NaN
                        specific_overdue_amount += col_value
                        if col_value > 0:  # 只記錄大於0的金額
                            overdue_details[col] = col_value
            
            # 只添加有未收款的公司
            if specific_overdue_amount > 0:
                remarks = row.get("备注", "")
                payment_reason = self._extract_reason_from_remarks(remarks)
                results.append({
                    "company_name": company_name,
                    "overdue_amount": specific_overdue_amount,
                    "overdue_details": overdue_details,
                    "payment_reason": payment_reason,
                    "abnormal": row.get("异常指标", ""),
                    "total_receivable": row.get("应收金额", 0) if pd.notna(row.get("应收金额", 0)) else 0,
                    "total_actual_overdue": row.get("到期-应收未收 合计", 0) if pd.notna(row.get("到期-应收未收 合计", 0)) else 0
                })
        
        # 按未收款金額降序排序
        results = sorted(results, key=lambda x: x["overdue_amount"], reverse=True)
        return results

    def get_top_companies_by_amount(self, top_n: int = 5, query_type: str = "overdue", min_days: int = None) -> List[Dict[str, Any]]:
        if query_type == "overdue":
            all_data = self.get_general_overdue_data()
        elif query_type == "receivable":
            all_data = self.get_receivable_data()
        else:
            if min_days is None:
                min_days = 90
            all_data = self.get_overdue_data_by_specific_days(min_days)
        return all_data[:top_n] if top_n < len(all_data) else all_data


class RAGQueryEngine:
    """RAG查詢引擎，用於處理用戶查詢並生成回答"""
    
    def __init__(self, data_processor: ARDataProcessor):
        self.data_processor = data_processor
        self.vector_store = data_processor.create_vector_store()
        self.llm = None
        self.rag_chain = None
        self.retriever = None
        self.use_llm = self.setup_llm()
        self.setup_rag_chain()
        
    def setup_llm(self) -> bool:
        try:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_OPENAI_API_BASE")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            if api_key and api_base:
                self.llm = AzureChatOpenAI(
                    azure_deployment=deployment_name, openai_api_version=api_version,
                    azure_endpoint=api_base, api_key=api_key, temperature=0.1
                )
                print("大語言模型設置完成")
                return True
            else:
                print("警告: 未找到有效的Azure OpenAI API密鑰，將使用本地處理模式")
                self.llm = self._create_mock_llm()
                return False
        except Exception as e:
            print(f"設置大語言模型時出錯: {e}")
            print("將使用本地處理模式")
            self.llm = self._create_mock_llm()
            return False
    
    def _create_mock_llm(self) -> Callable:
        def mock_llm(prompt):
            context = prompt.get("context", "")
            question = prompt.get("question", "")
            return f"這是一個本地處理模式的回答（未使用GPT-4o）。\n\n根據查詢結果:\n{context}\n\n您的查詢: {question}"
        return RunnableLambda(mock_llm)
        
    def setup_rag_chain(self):
        template = """你是一個專業的財務分析助手，專門處理公司應收款和未收款項的查詢。
        請基於以下檢索到的信息，回答用戶的問題。如果無法從檢索信息中找到答案，請明確說明。

        檢索到的信息:
        {context}

        用戶問題: {question}

        請提供詳細、準確的回答，並在適當的情況下分析可能的風險和異常指標。回答必須使用繁體中文。
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        if self.use_llm and self.llm and not isinstance(self.llm, RunnableLambda):
            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt | self.llm | StrOutputParser()
            )
            print("RAG鏈設置完成")
        else:
            print("使用本地處理模式，不使用LLM")
            # 即使在本地模式，也創建一個可以調用的鏈
            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self._create_mock_llm()
            )
    
    def normalize_query(self, query: str) -> str:
        return zhconv.convert(query, "zh-cn")
    
    def extract_top_n(self, query: str) -> int:
        normalized_query = self.normalize_query(query)
        top_n_match = re.search(r"前(\d+|[一二三四五六七八九十百千万]+)(個|个|名|家|条|條)", normalized_query)
        if top_n_match:
            num_str = top_n_match.group(1)
            if re.match(r"\d+", num_str):
                return int(num_str)
            else:
                chinese_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000, "万": 10000}
                return chinese_num_map.get(num_str, 5)
        if re.search(r"(所有|全部)", normalized_query):
            return 100
        return 5
    
    def extract_days(self, query: str) -> int:
        normalized_query = self.normalize_query(query)
        days_match = re.search(r"(超過|超出|大於|多於|超过|超出|大于|多于)(\d+)天", normalized_query)
        if days_match:
            return int(days_match.group(2))
        return None

    def extract_company_names_from_query(self, query: str) -> List[str]:
        """從查詢中提取一個或多個公司名稱"""
        normalized_query = self.normalize_query(query)
        
        # 1. 檢查是否是多公司比較查詢
        comparison_keywords = ["和", "与", "以及", "跟", "对比", "比较", "VS", "vs", "對比", "比較"]
        
        # 如果查詢中包含比較關鍵詞，嘗試分割查詢並提取多個公司名稱
        if any(keyword in normalized_query for keyword in comparison_keywords):
            # 使用正則表達式分割查詢，找出可能的公司名稱部分
            for keyword in comparison_keywords:
                if keyword in normalized_query:
                    parts = normalized_query.split(keyword)
                    if len(parts) >= 2:
                        found_companies = []
                        for part in parts:
                            # 對每個部分嘗試提取公司名稱
                            company_matches = self._extract_single_company_name(part.strip())
                            if company_matches:
                                found_companies.extend(company_matches)
                        
                        if len(found_companies) >= 2:
                            return list(set(found_companies))  # 去重
        
        # 2. 如果不是多公司比較查詢或無法提取多個公司，嘗試提取單個公司
        return self._extract_single_company_name(normalized_query)
    
    def _extract_single_company_name(self, text: str) -> List[str]:
        """從文本中提取單個公司名稱"""
        # 嘗試匹配已知的公司名稱
        known_companies = self.data_processor.company_matcher.company_data
        found_companies = []
        
        # 優先匹配較長的公司名稱
        sorted_known_companies = sorted(known_companies, key=len, reverse=True)
        
        # 1. 完整名稱匹配
        for company in sorted_known_companies:
            if company in text:
                found_companies.append(company)
                break  # 找到最長匹配即可
        
        if found_companies:
            return found_companies
        
        # 2. 簡稱匹配
        for short_name, full_name in self.data_processor.company_matcher.short_names.items():
            if short_name in text and len(short_name) >= 2:  # 確保簡稱至少有2個字符
                found_companies.append(full_name)
        
        if found_companies:
            return list(set(found_companies))  # 去重
        
        # 3. 使用更通用的正則表達式（可能不太準確）
        company_patterns = re.findall(r"([\u4e00-\u9fa5]+(?:科技|电子|公司|集团|股份|制造|精密|电路|光电|材料))", text)
        if company_patterns:
            # 嘗試用公司匹配器驗證提取的名稱
            validated_companies = []
            for pattern_match in company_patterns:
                matches = self.data_processor.company_matcher.match(pattern_match)
                if matches:
                    validated_companies.append(matches[0][0])  # 取最佳匹配
            if validated_companies:
                return list(set(validated_companies))
        
        return []

    def analyze_query_type(self, query: str) -> Dict[str, Any]:
        normalized_query = self.normalize_query(query)
        result = {
            "type": "unknown",
            "top_n": self.extract_top_n(normalized_query),
            "days": self.extract_days(normalized_query),
            "company_names": []  # 改為列表以支持多公司
        }

        # 1. 提取公司名稱 (最優先，因為很多查詢都可能包含公司名)
        extracted_names = self.extract_company_names_from_query(normalized_query)
        result["company_names"] = extracted_names
        
        # 2. 檢查是否是多公司比較查詢
        comparison_keywords = ["和", "与", "以及", "跟", "对比", "比较", "VS", "vs", "對比", "比較", "哪個", "哪家"]
        if len(extracted_names) > 1 and any(keyword in normalized_query for keyword in comparison_keywords):
            result["type"] = "multi_company_comparison"
            return result
        
        # 3. 檢查是否是單一特定公司查詢
        if len(extracted_names) == 1:
            # 避免將聚合查詢誤判為公司查詢
            aggregate_keywords = ["最多", "最大", "最高", "排名", "前", "所有", "全部", "超过", "超過"]
            
            # 如果查詢中包含聚合關鍵詞，但也明確提到了特定公司，優先判斷為公司查詢
            # 例如："江西科翔电子科技最新应收款情况如何"
            if not any(keyword in normalized_query for keyword in aggregate_keywords) or "情况" in normalized_query or "情況" in normalized_query:
                result["type"] = "company"
                return result
        
        # 4. 檢查是否是欠款最多的公司查詢
        top_overdue_patterns = [
            r"(.*?)(欠|欠款|欠錢|欠钱)(最多|最大|最高|排名|前|前\d+)(.*?)",
            r"(.*?)(未收款|未付款)(最多|最大|最高|排名|前|前\d+)(.*?)",
            r"(.*?)(欠款最多|欠钱最多|欠錢最多)(.*?)",
            r"(.*?)(未收款金額最大|未收款金额最大)(.*?)",
            r"(查询|查詢|列出|顯示|显示)(.*?)(欠款|欠|未收款)(.*?)(最多|最大|最高|排名)(.*?)",
            r"(列出|列举|列舉|显示|顯示)(.*?)(欠)(.*?)(最多|最大)(.*?)(公司|企业|企業)",
            r"(列出|列举|列舉|显示|顯示)(.*?)(欠我们钱|欠我們錢)(.*?)(前|最多)(.*?)",
            r"(.*?)(未收款金额最大的客户|未收款金額最大的客戶)(.*?)"
        ]
        for pattern in top_overdue_patterns:
            if re.search(pattern, normalized_query):
                result["type"] = "top_overdue"
                return result
        
        # 5. 檢查是否是應收款最多的公司查詢
        receivable_keywords = ["應收款", "应收款", "應收金額", "应收金额", "应收账款", "應收賬款"]
        top_receivable_patterns = [
            r"(.*?)(應收款|应收款|應收金額|应收金额|应收账款|應收賬款)(最多|最大|最高|排名|前|前\d+)(.*?)",
            r"(.*?)(應收款最多|应收款最多|應收金額最多|应收金额最多)(.*?)",
            r"(查询|查詢|列出|顯示|显示)(.*?)(應收款|应收款|應收金額|应收金额|应收账款|應收賬款)(.*?)(最多|最大|最高|排名)(.*?)",
            r"(列出|列举|列舉|显示|顯示)(.*?)(应收款最多|應收款最多)(.*?)(公司|企业|企業)",
            r"(哪家|哪些)(.*?)(公司|企业|企業)(.*?)(应收账款最高|應收賬款最高)"
        ]
        for pattern in top_receivable_patterns:
            if re.search(pattern, normalized_query) or any(f"{keyword}最多" in normalized_query for keyword in receivable_keywords):
                result["type"] = "top_receivable"
                return result
        
        # 6. 檢查是否是超過特定天數的未收款查詢
        overdue_days_patterns = [
            r"(超過|超出|大於|多於|超过|超出|大于|多于)(\d+)天(.*?)(未收款|欠款|未付款|拖欠)(.*?)",
            r"(.*?)(超過|超出|大於|多於|超过|超出|大于|多于)(\d+)天(.*?)",
            r"(哪些|哪家|什么|什麼)(.*?)(欠款|欠|拖欠|未付款|未收款)(超過|超出|大於|多於|超过|超出|大于|多于)(\d+)天",
            r"(列出|列举|列舉|显示|顯示)(.*?)(超過|超过)(\d+)天(.*?)(未付款|未收款|欠款)(.*?)(公司|客户|客戶)",
            r"(显示|顯示)(所有|全部)(超過|超过)(\d+)天(.*?)(未付款|未收款)(.*?)(公司|客户|客戶)"
        ]
        for pattern in overdue_days_patterns:
            overdue_days_match = re.search(pattern, normalized_query)
            if overdue_days_match or result["days"] is not None:
                result["type"] = "overdue_days"
                if overdue_days_match:
                    for group_idx in range(1, len(overdue_days_match.groups()) + 1):
                        group = overdue_days_match.group(group_idx)
                        if group and group.isdigit():
                            result["days"] = int(group)
                            break
                if not result["days"]:
                    result["days"] = 90
                return result
        
        # 7. 檢查是否是列出所有未收款客戶
        list_all_patterns = [
            r"(列出|列舉|顯示|查詢|列举|显示|查询)(所有|全部)(.*?)(未收款|欠款|未付款|拖欠)(.*?)",
            r"(.*?)(未收款項|未收款客户|未付款客户|欠款客户|欠公司钱)(.*?)",
            r"(所有|全部)(.*?)(未收款|欠款|未付款|拖欠)(.*?)",
            r"(显示|顯示)(.*?)(未收款|未付款)(.*?)(客户|客戶)",
            r"(哪些|哪家)(.*?)(公司|客户|客戶)(.*?)(欠款|欠|未付款|未收款)"
        ]
        for pattern in list_all_patterns:
            if re.search(pattern, normalized_query):
                result["type"] = "general_overdue"
                return result
        
        # 8. 如果仍然無法確定類型，但提取到了公司名，則認為是特定公司查詢
        if len(extracted_names) >= 1 and result["type"] == "unknown":
            if len(extracted_names) > 1:
                result["type"] = "multi_company_comparison"
            else:
                result["type"] = "company"
            return result
            
        # 9. 最終回退邏輯
        if result["type"] == "unknown":
            if "欠款" in normalized_query or "欠钱" in normalized_query or "欠錢" in normalized_query:
                result["type"] = "top_overdue"
            elif any(keyword in normalized_query for keyword in receivable_keywords):
                result["type"] = "top_receivable"
            elif "未收款" in normalized_query or "未付款" in normalized_query:
                result["type"] = "general_overdue"
        
        return result
        
    def process_query(self, query: str) -> str:
        try:
            query_analysis = self.analyze_query_type(query)
            query_type = query_analysis["type"]
            top_n = query_analysis["top_n"]
            days = query_analysis["days"]
            company_names = query_analysis["company_names"]
            
            print(f"查詢分析結果: 類型={query_type}, 數量={top_n}, 天數={days}, 公司名稱列表={company_names}")
            
            if query_type == "company" and company_names:
                return self._handle_company_query(company_names[0], query) # 取第一個公司名
            elif query_type == "multi_company_comparison" and len(company_names) > 1:
                return self._handle_multi_company_comparison_query(company_names, query)
            elif query_type == "overdue_days":
                return self._handle_overdue_days_query_specific(days, query, top_n)
            elif query_type == "general_overdue":
                return self._handle_general_overdue_query(query, top_n)
            elif query_type == "top_overdue":
                return self._handle_top_overdue_companies_query(top_n, query)
            elif query_type == "top_receivable":
                return self._handle_top_receivable_companies_query(top_n, query)
            else:
                if self.use_llm and self.rag_chain:
                    self.retriever.search_kwargs["k"] = top_n
                    return self.rag_chain.invoke(query)
                else:
                    # 本地模式下，如果無法識別類型，提供通用幫助或返回錯誤
                    return "無法識別您的查詢類型。請嘗試更明確的查詢，例如：\'列出所有未收款客戶\' 或 \'查询赣州科翔电子的未收款情况\'。" 
        except Exception as e:
            error_msg = f"處理查詢時出錯: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"處理查詢時出錯，請檢查系統日誌獲取詳細信息。錯誤類型: {type(e).__name__}"
    
    def _handle_company_query(self, company_name: str, original_query: str) -> str:
        company_data = self.data_processor.get_company_data(company_name)
        if "error" in company_data:
            # 如果LLM可用，讓LLM嘗試回答，否則返回錯誤
            return self.rag_chain.invoke(original_query) if self.use_llm and self.rag_chain and not isinstance(self.llm, RunnableLambda) else f"未找到匹配的公司: {company_name}"
        
        context = f"公司名稱: {company_data['company_name']}\n"
        context += f"總應收金額: {company_data['amounts'].get('应收金额', 'N/A')}\n"
        context += f"總未收款金額 (到期-应收未收 合计): {company_data['amounts'].get('到期-应收未收 合计', 'N/A')}\n\n賬齡分佈:\n"
        for col, value in company_data["amounts"].items():
            if col not in ["应收金额", "到期-应收未收 合计"]:
                context += f"{col}: {value}\n"
        context += f"\n未付款原因: {company_data['payment_reason']}\n"
        context += f"異常指標: {company_data['abnormal']}\n"
        
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"公司查詢結果 (本地處理模式):\n\n{context}"

    def _handle_multi_company_comparison_query(self, company_names: List[str], original_query: str) -> str:
        """處理多個公司的比較查詢"""
        context = "多公司比較查詢結果:\n\n"
        all_found = True
        for name in company_names:
            company_data = self.data_processor.get_company_data(name)
            if "error" in company_data:
                context += f"未找到公司 {name} 的數據。\n\n"
                all_found = False
                continue
            
            context += f"--- 公司: {company_data['company_name']} ---\n"
            context += f"總應收金額: {company_data['amounts'].get('应收金额', 'N/A')}\n"
            context += f"總未收款金額 (到期-应收未收 合计): {company_data['amounts'].get('到期-应收未收 合计', 'N/A')}\n"
            context += "賬齡分佈:\n"
            for col, value in company_data["amounts"].items():
                if col not in ["应收金额", "到期-应收未收 合计"]:
                    context += f"  {col}: {value}\n"
            context += f"未付款原因: {company_data['payment_reason']}\n"
            context += f"異常指標: {company_data['abnormal']}\n\n"
        
        if not all_found and len(company_names) > 1:
            context += "部分公司數據未找到，比較可能不完整。\n"
            
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"{context}"

    def _handle_general_overdue_query(self, original_query: str, top_n: int = 5) -> str:
        overdue_data = self.data_processor.get_general_overdue_data()
        if not overdue_data:
            return "沒有發現任何未收款項。"
        context = "一般未收款查詢結果 (使用「到期-应收未收 合计」列):\n\n"
        display_data = overdue_data[:top_n] if len(overdue_data) > top_n else overdue_data
        context += f"找到 {len(overdue_data)} 條記錄，顯示前 {len(display_data)} 條:\n\n"
        for item in display_data:
            context += f"公司名稱: {item['company_name']}\n"
            context += f"總未收款金額: {item['overdue_amount']}\n"
            context += f"總應收金額: {item['total_receivable']}\n"
            context += f"未付款原因: {item['payment_reason']}\n"
            context += f"異常指標: {item['abnormal']}\n\n"
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"{context}"

    def _handle_overdue_days_query_specific(self, days: int, original_query: str, top_n: int = 5) -> str:
        overdue_data = self.data_processor.get_overdue_data_by_specific_days(days)
        if not overdue_data:
            return f"沒有發現超過 {days} 天的未收款項。"
        context = f"超過 {days} 天未收款查詢結果 (累加特定賬齡段):\n\n"
        display_data = overdue_data[:top_n] if top_n < len(overdue_data) else overdue_data
        context += f"找到 {len(overdue_data)} 條記錄，顯示前 {len(display_data)} 條:\n\n"
        for item in display_data:
            context += f"公司名稱: {item['company_name']}\n"
            context += f"超過{days}天的未收款金額: {item['overdue_amount']}\n"
            context += f"總未收款金額: {item.get('total_actual_overdue', 'N/A')}\n"
            if "overdue_details" in item and item["overdue_details"]:
                details_str = ""
                for age_col, amount in item["overdue_details"].items():
                    details_str += f"{age_col}: {amount}, "
                context += f"賬齡明細: {details_str.rstrip(', ')}\n"
            context += f"未付款原因: {item['payment_reason']}\n"
            context += f"異常指標: {item['abnormal']}\n\n"
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"{context}"
    
    def _handle_top_overdue_companies_query(self, top_n: int, original_query: str) -> str:
        top_companies = self.data_processor.get_top_companies_by_amount(top_n, "overdue")
        if not top_companies:
            return "沒有發現任何未收款數據。"
        context = f"欠款金額最多的前 {len(top_companies)} 個公司 (基於總未收款金額):\n\n"
        for i, item in enumerate(top_companies):
            context += f"排名 {i+1}: {item['company_name']}\n"
            context += f"未收款金額: {item['overdue_amount']}\n"
            context += f"總應收金額: {item.get('total_receivable', 'N/A')}\n"
            context += f"未付款原因: {item['payment_reason']}\n"
            context += f"異常指標: {item['abnormal']}\n\n"
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"{context}"
    
    def _handle_top_receivable_companies_query(self, top_n: int, original_query: str) -> str:
        top_companies = self.data_processor.get_top_companies_by_amount(top_n, "receivable")
        if not top_companies:
            return "沒有發現任何應收款數據。"
        context = f"應收款金額最多的前 {len(top_companies)} 個公司 (基於總應收款金額):\n\n"
        for i, item in enumerate(top_companies):
            context += f"排名 {i+1}: {item['company_name']}\n"
            context += f"應收款金額: {item['receivable_amount']}\n"
            context += f"未收款金額: {item.get('overdue_amount', 'N/A')}\n"
            context += f"未付款原因: {item['payment_reason']}\n"
            context += f"異常指標: {item['abnormal']}\n\n"
        return self.rag_chain.invoke({"context": context, "question": original_query}) if self.use_llm and self.rag_chain else f"{context}"


# 創建RAG系統實例的工廠函數
def create_rag_system(excel_path: str):
    """創建RAG系統實例的工廠函數"""
    data_processor = ARDataProcessor(excel_path)
    rag_engine = RAGQueryEngine(data_processor)
    return rag_engine
