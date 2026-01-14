# 🔧 Installation Guide - Swing Trading System

Complete step-by-step installation guide for Windows, Linux, and Mac.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection
- 2GB free disk space

## Step 1: Verify Python Installation

```bash
# Check Python version
python --version
# or
python3 --version

# Should show Python 3.8.x or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt-get install python3 python3-pip`
- **Mac**: `brew install python3`

## Step 2: Download Project

```bash
# Navigate to desired location
cd C:\  # Windows
# or
cd ~/  # Linux/Mac

# If you have git:
git clone <repository-url> swing_trade

# Or extract downloaded ZIP file to swing_trade folder
```

## Step 3: Create Virtual Environment (Recommended)

### Windows
```powershell
cd swing_trade
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac
```bash
cd swing_trade
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Step 4: Install Dependencies

### Standard Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### If Installation Fails

**Issue**: TA-Lib installation error on Windows

**Solution**:
1. Download pre-built TA-Lib wheel from:
   - [TA-Lib Wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Choose correct version for your Python (e.g., `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl` for Python 3.11 64-bit)

2. Install wheel:
```bash
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

3. Continue with other packages:
```bash
pip install -r requirements.txt
```

**Alternative**: Use pandas-ta instead (already in requirements.txt, no TA-Lib needed)

## Step 5: Run Setup Script

```bash
python setup.py --full-setup
```

This will:
- ✅ Create necessary directories
- ✅ Initialize database
- ✅ Create .env template
- ✅ Verify dependencies
- ✅ Check configuration

Expected output:
```
====================================================
SWING TRADING SYSTEM SETUP
====================================================

[1/5] Creating directories...
[2/5] Creating .env file...
[3/5] Checking dependencies...
[4/5] Verifying configuration...
[5/5] Initializing database...

✅ SETUP COMPLETE!
```

## Step 6: Configure (Optional)

### Basic Configuration (No API Keys Needed)

The system works out-of-the-box with `yfinance` (free, no API key required).

### Advanced Configuration (Optional Alerts)

Edit `.env` file for email/Telegram alerts:

```bash
# Open .env in text editor
notepad .env  # Windows
nano .env     # Linux
open -e .env  # Mac
```

Add your credentials:
```
# Email alerts (Gmail)
EMAIL_PASSWORD=your_app_password_here

# Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Getting Email App Password (Gmail)**:
1. Go to Google Account settings
2. Security → 2-Step Verification
3. App passwords → Generate
4. Copy password to .env

**Getting Telegram Bot Token**:
1. Message @BotFather on Telegram
2. Send `/newbot` and follow instructions
3. Copy token to .env
4. Get your chat_id from @userinfobot

## Step 7: Test Installation

```bash
# Check if everything works
python -c "import pandas, numpy, yfinance, streamlit; print('✅ All imports successful')"
```

## Step 8: Download Initial Data

```bash
# Download S&P 500 historical data (last 250 days)
python main.py --download-data --days=250
```

This takes **10-20 minutes** depending on internet speed.

Expected output:
```
====================================================
DOWNLOADING INITIAL DATA
====================================================
Fetching S&P 500 ticker list...
Found 503 tickers
Downloading stocks: 100%|████████| 503/503 [15:23<00:00]
Successfully downloaded: 485/503 stocks
```

## Step 9: Verify Installation

```bash
# Run a test scan
python main.py --daily-scan --portfolio-value=10000
```

If you see signals in the output, installation is complete!

## Step 10: Launch Dashboard

```bash
streamlit run swing_trader/dashboard/app.py
```

Browser should automatically open to `http://localhost:8501`

## Troubleshooting

### Issue: "Python not found"
**Windows**: Add Python to PATH during installation  
**Linux/Mac**: Use `python3` instead of `python`

### Issue: "pip not found"
```bash
# Windows
python -m ensurepip --upgrade

# Linux
sudo apt-get install python3-pip
```

### Issue: "Permission denied"
```bash
# Linux/Mac: Use sudo
sudo pip install -r requirements.txt

# Or use user installation
pip install --user -r requirements.txt
```

### Issue: "Module not found after installation"
```bash
# Make sure virtual environment is activated
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Reinstall in venv
pip install -r requirements.txt
```

### Issue: "Database locked"
```bash
# Close all Python processes
# Delete database and reinitialize
rm data/stocks.db  # Linux/Mac
del data\stocks.db  # Windows
python setup.py --init-db
```

### Issue: "Port 8501 already in use"
```bash
# Use different port
streamlit run swing_trader/dashboard/app.py --server.port 8502
```

## System Requirements

### Minimum
- CPU: Dual-core 2.0 GHz
- RAM: 4GB
- Storage: 2GB free space
- OS: Windows 10, Ubuntu 18.04, macOS 10.14+

### Recommended
- CPU: Quad-core 2.5 GHz+
- RAM: 8GB+
- Storage: 5GB+ free space
- SSD for faster database operations

## Next Steps

After successful installation:

1. ✅ **Review Configuration**: Edit `config.yaml` if needed
2. ✅ **Learn the Strategy**: Read strategy section in README
3. ✅ **Run Backtest**: Test on historical data first
4. ✅ **Paper Trade**: Practice with virtual money (3+ months)
5. ✅ **Daily Scans**: Set up automated daily scans

## Automated Daily Scans (Optional)

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 5:30 PM
4. Action: Start Program
   - Program: `C:\swing_trade\venv\Scripts\python.exe`
   - Arguments: `main.py --daily-scan`
   - Start in: `C:\swing_trade`

### Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Add line (runs Mon-Fri at 5:30 PM)
30 17 * * 1-5 cd /path/to/swing_trade && ./venv/bin/python main.py --daily-scan
```

## Updating the System

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Pull latest changes (if using git)
git pull

# Update dependencies
pip install --upgrade -r requirements.txt

# Update database schema if needed
python setup.py --init-db
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove directory
rm -rf swing_trade  # Linux/Mac
rmdir /s swing_trade  # Windows
```

---

**Need Help?** Check README.md troubleshooting section or review code comments.

