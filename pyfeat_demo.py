import gradio as gr
import matplotlib.pyplot as plt
from feat import Detector
from PIL import Image


def detect_and_plot(image: Image.Image, face_model: str, landmark_model: str, au_model: str, emotion_model: str, facepose_model: str) -> plt.Figure:
    # 画像を RGB に変換 (透過 PNG 対策)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert("RGB")

    # 一時ファイルに保存
    image_path = "temp_image.png"
    image.save(image_path)

    # Detectorの初期化
    detector = Detector(
        face_model=face_model,
        landmark_model=landmark_model,
        au_model=au_model,
        emotion_model=emotion_model,
        facepose_model=facepose_model,
    )

    # 感情検出
    result = detector.detect_image(image_path)

    # プロット
    result.plot_detections(poses=True)

    return plt


# Gradioインターフェースの設定
face_model_options = ['retinaface', 'mtcnn', 'faceboxes', 'img2pose', 'img2pose-c']
landmark_model_options = ['mobilefacenet', 'mobilenet', 'pfld']
au_model_options = ['xgb', 'svm']
emotion_model_options = ['resmasknet', 'svm']
facepose_model_options = ['img2pose', 'img2pose-c']
interface = gr.Interface(
    fn=detect_and_plot,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=face_model_options, value=face_model_options[0], label="Face Model"),
        gr.Dropdown(choices=landmark_model_options, value=landmark_model_options[0], label="Landmark Model"),
        gr.Dropdown(choices=au_model_options, value=au_model_options[0], label="AU Model"),
        gr.Dropdown(choices=emotion_model_options, value=emotion_model_options[0], label="Emotion Model"),
        gr.Dropdown(choices=facepose_model_options, value=facepose_model_options[0], label="Facepose Model")
    ],
    outputs=gr.Plot(),
    title="Emotion Detection",
    description="Upload an image to detect emotions and display the plot. Select the face model, landmark model, AU model, emotion model, and facepose model from the dropdown menus."
)

# インターフェースの起動
if __name__ == "__main__":
    interface.launch()
