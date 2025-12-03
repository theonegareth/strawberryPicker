#!/usr/bin/env python3
"""
Simple Web Interface for Labeling Strawberry Crops
Run this script and open http://localhost:5000 in your browser
"""

import os
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for
import random

app = Flask(__name__)

# Configuration
DATASET_PATH = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/ripeness_manual_dataset")
TO_LABEL_DIR = DATASET_PATH / "to_label"
CLASS_DIRS = {
    "unripe": DATASET_PATH / "unripe",
    "ripe": DATASET_PATH / "ripe",
    "overripe": DATASET_PATH / "overripe"
}

# Ensure directories exist
for dir_path in CLASS_DIRS.values():
    dir_path.mkdir(exist_ok=True)

def get_remaining_images():
    """Get list of images still to label"""
    return sorted([f for f in TO_LABEL_DIR.glob("*.jpg") if f.is_file()])

def get_progress():
    """Get labeling progress"""
    total = 889  # Known total
    remaining = len(get_remaining_images())
    labeled = total - remaining
    return labeled, total, (labeled / total * 100) if total > 0 else 0

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Strawberry Ripeness Labeling</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .progress {
            background-color: #e0e0e0;
            border-radius: 10px;
            padding: 5px;
            margin: 20px 0;
        }
        .progress-bar {
            background-color: #4CAF50;
            height: 30px;
            border-radius: 5px;
            text-align: center;
            line-height: 30px;
            color: white;
            font-weight: bold;
            transition: width 0.3s;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border: 2px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            min-width: 150px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .btn-unripe {
            background-color: #90EE90;
            color: #333;
        }
        .btn-ripe {
            background-color: #FF6B6B;
            color: white;
        }
        .btn-overripe {
            background-color: #8B4513;
            color: white;
        }
        .btn-skip {
            background-color: #FFA500;
            color: white;
        }
        .guide {
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .stat {
            text-align: center;
            padding: 10px;
        }
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .completed {
            text-align: center;
            padding: 50px;
        }
        .completed h2 {
            color: #4CAF50;
            font-size: 48px;
        }
        .shortcuts {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        {% if completed %}
            <div class="completed">
                <h2>🎉 Labeling Complete!</h2>
                <p>You've successfully labeled all {{ total }} strawberry images.</p>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">{{ counts.unripe }}</div>
                        <div class="stat-label">Unripe</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{{ counts.ripe }}</div>
                        <div class="stat-label">Ripe</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{{ counts.overripe }}</div>
                        <div class="stat-label">Overripe</div>
                    </div>
                </div>
                <p><a href="{{ url_for('index') }}" style="color: #4CAF50; font-size: 20px;">View Summary</a></p>
            </div>
        {% else %}
            <h1>🍓 Strawberry Ripeness Labeling</h1>
            
            <div class="progress">
                <div class="progress-bar" style="width: {{ progress }}%">
                    {{ progress|round(1) }}% ({{ labeled }}/{{ total }})
                </div>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{{ remaining }}</div>
                    <div class="stat-label">Remaining</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{{ counts.unripe }}</div>
                    <div class="stat-label">Unripe</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{{ counts.ripe }}</div>
                    <div class="stat-label">Ripe</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{{ counts.overripe }}</div>
                    <div class="stat-label">Overripe</div>
                </div>
            </div>
            
            <div class="shortcuts">
                <strong>Keyboard Shortcuts:</strong> 
                U = Unripe | R = Ripe | O = Overripe | S = Skip
            </div>
            
            <div class="guide">
                <h3>Labeling Guide:</h3>
                <p><strong>Unripe (Green button):</strong> Green, white, or pale pink strawberries. Small, firm, not ready to pick.</p>
                <p><strong>Ripe (Red button):</strong> Bright red, full size, firm but not hard. Perfect for picking!</p>
                <p><strong>Overripe (Brown button):</strong> Dark red, soft, mushy, or wrinkled. Too late for picking.</p>
            </div>
            
            <div class="image-container">
                <img src="{{ image_url }}" alt="Strawberry to label">
            </div>
            
            <form method="POST" action="{{ url_for('label') }}">
                <input type="hidden" name="image" value="{{ current_image }}">
                <div class="buttons">
                    <button type="submit" name="class" value="unripe" class="btn btn-unripe">
                        🟢 Unripe
                    </button>
                    <button type="submit" name="class" value="ripe" class="btn btn-ripe">
                        🔴 Ripe
                    </button>
                    <button type="submit" name="class" value="overripe" class="btn btn-overripe">
                        ⚫ Overripe
                    </button>
                    <button type="submit" name="class" value="skip" class="btn btn-skip">
                        ⏭️ Skip
                    </button>
                </div>
            </form>
        {% endif %}
    </div>
    
    <script>
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'u' || e.key === 'U') {
                document.querySelector('.btn-unripe').click();
            } else if (e.key === 'r' || e.key === 'R') {
                document.querySelector('.btn-ripe').click();
            } else if (e.key === 'o' || e.key === 'O') {
                document.querySelector('.btn-overripe').click();
            } else if (e.key === 's' || e.key === 'S') {
                document.querySelector('.btn-skip').click();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main labeling interface"""
    labeled, total, progress = get_progress()
    remaining = len(get_remaining_images())
    
    # Get counts per class
    counts = {
        "unripe": len(list(CLASS_DIRS["unripe"].glob("*.jpg"))),
        "ripe": len(list(CLASS_DIRS["ripe"].glob("*.jpg"))),
        "overripe": len(list(CLASS_DIRS["overripe"].glob("*.jpg")))
    }
    
    if remaining == 0:
        return render_template_string(HTML_TEMPLATE, 
                                    completed=True,
                                    total=total,
                                    counts=counts)
    
    # Get next image
    images = get_remaining_images()
    current_image = images[0]
    image_url = f"/image/{current_image.name}"
    
    return render_template_string(HTML_TEMPLATE,
                                completed=False,
                                current_image=current_image.name,
                                image_url=image_url,
                                labeled=labeled,
                                total=total,
                                remaining=remaining,
                                progress=get_progress()[2],
                                counts=counts)

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve image file"""
    return send_from_directory(TO_LABEL_DIR, filename)

@app.route('/label', methods=['POST'])
def label():
    """Handle labeling action"""
    image_name = request.form['image']
    class_name = request.form['class']
    
    src_path = TO_LABEL_DIR / image_name
    
    if class_name == 'skip':
        # Move to back of queue (rename temporarily)
        src_path.rename(TO_LABEL_DIR / f"_skipped_{image_name}")
    else:
        # Move to appropriate class folder
        dest_path = CLASS_DIRS[class_name] / image_name
        src_path.rename(dest_path)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🍓 Starting Strawberry Labeling Interface...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nKeyboard shortcuts:")
    print("  U = Unripe")
    print("  R = Ripe")
    print("  O = Overripe")
    print("  S = Skip")
    print("\nClick the buttons or use keyboard shortcuts to label images.")
    app.run(host='0.0.0.0', port=5000, debug=True)