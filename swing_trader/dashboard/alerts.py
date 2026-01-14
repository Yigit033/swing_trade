"""
Alert system for email and Telegram notifications.
"""

import logging
from typing import List, Dict, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Manages alerts via email and Telegram.
    
    Attributes:
        config (Dict): Configuration dictionary
        email_enabled (bool): Whether email alerts are enabled
        telegram_enabled (bool): Whether Telegram alerts are enabled
    """
    
    def __init__(self, config: Dict):
        """
        Initialize AlertSystem.
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config
        alert_config = config.get('alerts', {})
        
        self.email_enabled = alert_config.get('email_enabled', False)
        self.telegram_enabled = alert_config.get('telegram_enabled', False)
        
        # Email settings
        self.email_from = alert_config.get('email_from', '')
        self.email_to = alert_config.get('email_to', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
        # Telegram settings
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        logger.info(f"AlertSystem initialized (Email: {self.email_enabled}, Telegram: {self.telegram_enabled})")
    
    def send_email(self, subject: str, body: str) -> bool:
        """
        Send email alert.
        
        Args:
            subject: Email subject
            body: Email body (can be HTML)
        
        Returns:
            True if sent successfully, False otherwise
        
        Example:
            >>> alerts = AlertSystem(config)
            >>> alerts.send_email("New Signal", "AAPL buy signal detected!")
        """
        if not self.email_enabled:
            logger.debug("Email alerts disabled")
            return False
        
        if not all([self.email_from, self.email_to, self.email_password]):
            logger.warning("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            
            # Add body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send via Gmail SMTP
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return False
    
    def send_telegram(self, message: str) -> bool:
        """
        Send Telegram alert.
        
        Args:
            message: Message text (supports Markdown)
        
        Returns:
            True if sent successfully, False otherwise
        
        Example:
            >>> alerts = AlertSystem(config)
            >>> alerts.send_telegram("🔔 New signal: AAPL")
        """
        if not self.telegram_enabled:
            logger.debug("Telegram alerts disabled")
            return False
        
        if not all([self.telegram_token, self.telegram_chat_id]):
            logger.warning("Telegram credentials not configured")
            return False
        
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram message sent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
            return False
    
    def format_signal_alert(self, signal: Dict) -> str:
        """
        Format trading signal for alert.
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Formatted message string
        """
        ticker = signal.get('ticker', 'UNKNOWN')
        score = signal.get('score', 0)
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        target_1 = signal.get('target_1', 0)
        rsi = signal.get('rsi', 0)
        adx = signal.get('adx', 0)
        volume_surge = signal.get('volume_surge', 0)
        
        message = f"""
🔔 NEW SIGNAL: {ticker}

📊 Score: {score}/10
💰 Entry: ${entry_price:.2f}
🛡️ Stop Loss: ${stop_loss:.2f}
🎯 Target: ${target_1:.2f}

📈 Indicators:
  • RSI: {rsi:.1f}
  • ADX: {adx:.1f}
  • Volume Surge: {volume_surge:.2f}x

Risk/Reward: {((target_1 - entry_price) / (entry_price - stop_loss)):.1f}:1
        """.strip()
        
        return message
    
    def send_signal_alert(self, signal: Dict) -> bool:
        """
        Send alert for a trading signal.
        
        Args:
            signal: Signal dictionary
        
        Returns:
            True if sent successfully, False otherwise
        
        Example:
            >>> alerts = AlertSystem(config)
            >>> alerts.send_signal_alert(signal)
        """
        message = self.format_signal_alert(signal)
        
        success = True
        
        if self.email_enabled:
            subject = f"🔔 Trading Signal: {signal.get('ticker', 'UNKNOWN')}"
            html_body = f"<pre>{message}</pre>"
            success = success and self.send_email(subject, html_body)
        
        if self.telegram_enabled:
            success = success and self.send_telegram(message)
        
        return success
    
    def send_daily_summary(self, signals: List[Dict], date: str) -> bool:
        """
        Send daily summary of signals.
        
        Args:
            signals: List of signal dictionaries
            date: Date string
        
        Returns:
            True if sent successfully, False otherwise
        
        Example:
            >>> alerts = AlertSystem(config)
            >>> alerts.send_daily_summary(signals, '2024-01-15')
        """
        if not signals:
            logger.debug("No signals to report")
            return False
        
        # Sort by score
        signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)
        top_signals = signals[:5]
        
        message = f"📊 Daily Trading Summary - {date}\n\n"
        message += f"Total Signals: {len(signals)}\n"
        message += f"Avg Score: {sum(s.get('score', 0) for s in signals) / len(signals):.1f}/10\n\n"
        message += "Top 5 Signals:\n"
        
        for i, signal in enumerate(top_signals, 1):
            ticker = signal.get('ticker', 'UNKNOWN')
            score = signal.get('score', 0)
            entry = signal.get('entry_price', 0)
            message += f"{i}. {ticker}: ${entry:.2f} (Score: {score}/10)\n"
        
        success = True
        
        if self.email_enabled:
            subject = f"📊 Daily Summary - {date} ({len(signals)} signals)"
            html_body = f"<pre>{message}</pre>"
            success = success and self.send_email(subject, html_body)
        
        if self.telegram_enabled:
            success = success and self.send_telegram(message)
        
        return success
    
    def send_performance_report(self, metrics: Dict, period: str = "Weekly") -> bool:
        """
        Send performance report.
        
        Args:
            metrics: Performance metrics dictionary
            period: Report period (e.g., "Weekly", "Monthly")
        
        Returns:
            True if sent successfully, False otherwise
        
        Example:
            >>> alerts = AlertSystem(config)
            >>> alerts.send_performance_report(metrics, "Weekly")
        """
        message = f"📈 {period} Performance Report\n\n"
        
        message += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
        message += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        message += f"Win Rate: {metrics.get('win_rate', 0):.1%}\n"
        message += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        message += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        message += f"Max Drawdown: {metrics.get('max_drawdown_percent', 0):.1f}%\n"
        
        success = True
        
        if self.email_enabled:
            subject = f"📈 {period} Performance Report"
            html_body = f"<pre>{message}</pre>"
            success = success and self.send_email(subject, html_body)
        
        if self.telegram_enabled:
            success = success and self.send_telegram(message)
        
        return success

