#!/usr/bin/env python3
"""
Setup Kaggle API and download the fruit ripeness dataset
"""

import os
from pathlib import Path

def setup_kaggle_instructions():
    """
    Print step-by-step instructions for setting up Kaggle API
    """
    print("🔐 KAGGLE API SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("To download Kaggle datasets automatically, you need to:")
    print("1. Create a Kaggle account (if you don't have one)")
    print("2. Get your API token")
    print("3. Place it in the right location")
    print()
    
    print("STEP 1: Get Your Kaggle API Token")
    print("-" * 40)
    print("1. Go to https://www.kaggle.com")
    print("2. Login to your account (or create one)")
    print("3. Click on your profile picture (top right)")
    print("4. Select 'Account' from the dropdown menu")
    print("5. Scroll down to 'API' section")
    print("6. Click 'Create New API Token'")
    print("7. This will download a file named 'kaggle.json'")
    print()
    
    print("STEP 2: Place the Token File")
    print("-" * 40)
    print("1. The downloaded kaggle.json file should contain:")
    print('   {"username":"your_username","key":"your_api_key"}')
    print()
    print("2. Move it to the correct location:")
    print("   mkdir -p ~/.kaggle")
    print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
    print()
    print("3. Set proper permissions:")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print()
    
    print("STEP 3: Download the Dataset")
    print("-" * 40)
    print("Once the token is set up, run:")
    print("   kaggle datasets download -d dudinurdiyansah/fruit-ripeness-dataset")
    print()
    print("Then unzip it:")
    print("   unzip fruit-ripeness-dataset.zip -d ~/Downloads/fruit-ripeness-dataset")
    print()
    
    print("STEP 4: Extract Strawberries")
    print("-" * 40)
    print("Finally, extract only strawberry images:")
    print("   cd /home/user/machine-learning/GitHubRepos/strawberryPicker/model")
    print("   python3 training/extract_strawberries_from_kaggle.py")
    print()
    
    print("=" * 60)
    print("ALTERNATIVE: Manual Download")
    print("=" * 60)
    print()
    print("If you prefer not to use the Kaggle API:")
    print("1. Go to: https://www.kaggle.com/datasets/dudinurdiyansah/fruit-ripeness-dataset")
    print("2. Click 'Download' button")
    print("3. Unzip the downloaded file")
    print("4. Run the extraction script and provide the path")
    print()

def check_kaggle_setup():
    """Check if Kaggle is properly set up"""
    kaggle_dir = Path.home() / ".kaggle"
    token_file = kaggle_dir / "kaggle.json"
    
    if token_file.exists():
        print("✅ Kaggle token found!")
        print(f"   Location: {token_file}")
        
        # Check permissions
        import stat
        file_stat = token_file.stat()
        if file_stat.st_mode & stat.S_IROTH:
            print("⚠️  Warning: Token file is readable by others!")
            print("   Run: chmod 600 ~/.kaggle/kaggle.json")
        else:
            print("✅ Token file permissions are correct")
        
        return True
    else:
        print("❌ Kaggle token not found")
        print("   Follow the instructions above to set it up")
        return False

if __name__ == "__main__":
    print("KAGGLE SETUP HELPER")
    print("=" * 60)
    print()
    
    # Check current setup
    if check_kaggle_setup():
        print()
        print("🎉 Kaggle is ready to use!")
        print("   You can now download datasets")
    else:
        print()
        setup_kaggle_instructions()
    
    print()
    print("=" * 60)
    print("Need help? Visit: https://github.com/Kaggle/kaggle-api")
    print("=" * 60)