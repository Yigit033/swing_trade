"""
tools.py - LLM Function/Tool Calling tanımları ve callback fonksiyonları.

Bu modül, asistanın (Agentic AI) kullanabileceği araçların JSON şemalarını ve
gerçek hayattaki işlevlerini (Python callback'lerini) içerir.
"""

import json
import logging
from typing import Dict, Any
from swing_trader.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Tool Tanımları (JSON Schema formatında)
# -----------------------------------------------------------------------------

GET_LIVE_STOCK_DATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_live_stock_data",
        "description": "Fetch real-time stock data including the latest price, volume, moving averages (MA20, MA50) and company information for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., AAPL, TSLA, NVDA)"
                }
            },
            "required": ["ticker"]
        }
    }
}

# -----------------------------------------------------------------------------
# 2. Tool Çalıştırma (Callback) Fonksiyonları
# -----------------------------------------------------------------------------

def execute_get_live_stock_data(ticker: str) -> str:
    """
    LLM 'get_live_stock_data' aracını çağırdığında bu fonksiyon çalışır.
    DataFetcher kullanarak canlı veriyi çeker ve LLM'e özet bir string döndürür.
    """
    logger.info(f"Tool Execute: get_live_stock_data for {ticker}")
    fetcher = DataFetcher()
    
    try:
        # 1. Hisse temel bilgilerini çek
        info = fetcher.get_stock_info(ticker)
        
        # 2. Son 3 aylık fiyat hareketini çek (MA hesaplaması için)
        df = fetcher.fetch_stock_data(ticker, period="3mo")
        
        if df is None or df.empty:
            return f"Error: Could not fetch market data for {ticker}. The ticker might be invalid or data is unavailable."

        # 3. Teknik göstergeleri hesapla
        last_close = df['Close'].iloc[-1]
        last_vol = df['Volume'].iloc[-1]
        
        ma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
        ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        
        # 4. LLM'in okuması için temiz bir özet (Markdown) hazırla
        summary = f"### Canlı Piyasa Verisi: {ticker}\n"
        if info:
            summary += f"- **Şirket:** {info.get('name', 'N/A')}\n"
            summary += f"- **Sektör:** {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}\n"
        
        summary += f"- **Son Fiyat:** ${last_close:.2f}\n"
        summary += f"- **Son Hacim:** {int(last_vol):,}\n"
        
        if ma20:
            summary += f"- **MA20 (20 Günlük Ort.):** ${ma20:.2f}\n"
        if ma50:
            summary += f"- **MA50 (50 Günlük Ort.):** ${ma50:.2f}\n"
            
        logger.info(f"Tool Execute Başarılı: {ticker} -> Son Fiyat: ${last_close:.2f}")
        return summary

    except Exception as e:
        logger.error(f"Tool Execute Hatası ({ticker}): {e}")
        return f"Error occurred while fetching data for {ticker}: {str(e)}"
