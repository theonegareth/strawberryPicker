from roboflow import Roboflow

# Replace with your Roboflow API key and project details
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version("YOUR_VERSION").download("folder")

print("Dataset downloaded to dataset/")