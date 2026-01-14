"""
Setup script for initializing the swing trading system.
"""

import argparse
import yaml
import os
import sys
import logging
from pathlib import Path

from swing_trader.data.storage import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'logs',
        'output',
        'swing_trader/data',
        'swing_trader/indicators',
        'swing_trader/strategy',
        'swing_trader/backtesting',
        'swing_trader/dashboard',
        'swing_trader/tests'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")


def create_env_file():
    """Create .env file from template."""
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if env_path.exists():
        logger.info(".env file already exists")
        return
    
    if env_example_path.exists():
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        logger.info("Created .env file from template")
        logger.warning("⚠️  Please edit .env file and add your API keys!")
    else:
        logger.warning(".env.example not found")


def init_database(config_path: str = 'config.yaml'):
    """Initialize database."""
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        db_path = config['data']['database_path']
        db = DatabaseManager(db_path)
        
        logger.info(f"Initializing database: {db_path}")
        success = db.initialize_database()
        
        if success:
            logger.info("✅ Database initialized successfully")
        else:
            logger.error("❌ Failed to initialize database")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)


def verify_config(config_path: str = 'config.yaml'):
    """Verify configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'filters', 'indicators', 'risk', 'strategy', 'backtesting', 'alerts']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing required config section: {section}")
                sys.exit(1)
        
        logger.info("✅ Configuration file verified")
        return config
        
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"❌ Invalid YAML in configuration file: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas',
        'numpy',
        'yfinance',
        'pandas_ta',
        'streamlit',
        'plotly',
        'yaml',
        'scipy'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"❌ Missing required packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("✅ All required dependencies installed")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Swing Trading System')
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create configuration template'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies'
    )
    
    parser.add_argument(
        '--full-setup',
        action='store_true',
        help='Perform full setup (all steps)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SWING TRADING SYSTEM SETUP")
    print("="*60 + "\n")
    
    if args.full_setup or not any(vars(args).values()):
        # Full setup
        logger.info("Starting full setup...")
        
        # 1. Create directories
        logger.info("\n[1/5] Creating directories...")
        create_directories()
        
        # 2. Create .env file
        logger.info("\n[2/5] Creating .env file...")
        create_env_file()
        
        # 3. Check dependencies
        logger.info("\n[3/5] Checking dependencies...")
        check_dependencies()
        
        # 4. Verify config
        logger.info("\n[4/5] Verifying configuration...")
        verify_config()
        
        # 5. Initialize database
        logger.info("\n[5/5] Initializing database...")
        init_database()
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys (if using alerts)")
        print("2. Review config.yaml and adjust parameters if needed")
        print("3. Download data: python main.py --download-data")
        print("4. Run daily scan: python main.py --daily-scan")
        print("5. Launch dashboard: streamlit run swing_trader/dashboard/app.py")
        print("\n")
        
    else:
        if args.check_deps:
            check_dependencies()
        
        if args.create_config:
            logger.info("config.yaml already exists")
        
        if args.init_db:
            init_database()


if __name__ == '__main__':
    main()

