from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, Any
import os
from rag_core import create_rag_system

app = FastAPI(title="RAG应收帐系统API")

# 允许所有CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_rag_system = None

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    global current_rag_system
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="只支持.xlsx文件")
    try:
        file_path = "temp_ar_data.xlsx"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        current_rag_system = create_rag_system(file_path)
        df = pd.read_excel(file_path)
        total_companies = len(df["客户名称"].unique())
        total_amount = df["应收金额"].sum()
        return {
            "message": "文件上传成功",
            "total_companies": total_companies,
            "total_amount": total_amount
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")

@app.post("/api/query")
async def process_query(query: Dict[str, str]) -> Dict[str, Any]:
    global current_rag_system
    if not current_rag_system:
        raise HTTPException(status_code=400, detail="请先上传Excel文件")
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="查询不能为空")
        response = current_rag_system.process_query(user_query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 