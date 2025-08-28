import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import math

# ページ設定
st.set_page_config(
    page_title="YOLO姿勢推定アプリ",
    layout="wide"
)

# タイトル
st.title("YOLO姿勢推定アプリ")
st.markdown("写真をアップロードすると、AIが姿勢を推定します")

# YOLOの簡単な説明
st.markdown("""
**YOLO（You Only Look Once）**は、リアルタイム物体検出で世界的に有名な深層学習技術です。  
このアプリでは、YOLOの姿勢推定版を使用して人体の17個のキーポイントを高精度で検出し、詳細な姿勢分析を行います。
""")

# タブの作成
tab1, tab2 = st.tabs(["姿勢推定", "📋READ ME"])

@st.cache_resource
def load_model():
    """YOLOモデルの読み込み"""
    try:
        model = YOLO('yolov8n-pose.pt')
        return model
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None

def fix_image_orientation(image):
    """EXIF情報に基づいて画像の向きを自動修正"""
    try:
        # PIL.ImageOpsのexif_transposeを使用してEXIF情報に基づいて自動回転
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        # EXIF情報がない場合やエラーの場合は元の画像を返す
        return image

def hex_to_bgr(hex_color):
    """HEX色をBGR形式に変換"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb  # RGB形式のままで返す

def calculate_angle(p1, p2, p3):
    """3点から角度を計算（内角を返す）"""
    try:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    except:
        return None

def detect_posture_orientation(keypoints):
    """姿勢の向き（正面/横向き）を自動判定"""
    try:
        if len(keypoints) == 0:
            return "unknown"
        
        kpts = keypoints[0]
        
        # 左右の肩、腰の座標を取得
        left_shoulder = kpts[5][:2]   # 左肩
        right_shoulder = kpts[6][:2]  # 右肩
        left_hip = kpts[11][:2]       # 左腰
        right_hip = kpts[12][:2]      # 右腰
        
        # 信頼度をチェック
        if (kpts[5][2] < 0.5 or kpts[6][2] < 0.5 or 
            kpts[11][2] < 0.5 or kpts[12][2] < 0.5):
            return "unknown"
        
        # 肩と腰の幅を計算
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        
        # 体の高さを計算
        body_height = abs((left_shoulder[1] + right_shoulder[1])/2 - (left_hip[1] + right_hip[1])/2)
        
        # 幅と高さの比率で判定
        width_to_height_ratio = (shoulder_width + hip_width) / 2 / body_height
        
        # 閾値で判定（調整可能）
        if width_to_height_ratio > 0.8:
            return "front"  # 正面
        else:
            return "side"   # 横向き
            
    except Exception as e:
        return "unknown"

def analyze_front_posture(keypoints):
    """正面姿勢の分析"""
    try:
        kpts = keypoints[0]
        results = {}
        
        # キーポイントのインデックス
        nose = kpts[0][:2]           # 鼻
        left_shoulder = kpts[5][:2]   # 左肩
        right_shoulder = kpts[6][:2]  # 右肩
        left_hip = kpts[11][:2]       # 左腰
        right_hip = kpts[12][:2]      # 右腰
        
        # 重心位置（肩と腰の中点）
        shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                          (left_shoulder[1] + right_shoulder[1])/2]
        hip_center = [(left_hip[0] + right_hip[0])/2, 
                     (left_hip[1] + right_hip[1])/2]
        
        center_of_gravity = [(shoulder_center[0] + hip_center[0])/2,
                           (shoulder_center[1] + hip_center[1])/2]
        
        results["重心位置"] = f"X: {center_of_gravity[0]:.1f}, Y: {center_of_gravity[1]:.1f}"
        
        # 頭の傾き（鼻と肩中心の水平からの角度）
        if kpts[0][2] > 0.5:  # 鼻の信頼度チェック
            head_tilt = math.degrees(math.atan2(nose[1] - shoulder_center[1], 
                                              nose[0] - shoulder_center[0]) - math.pi/2)
            results["頭の傾き"] = f"{head_tilt:.1f}°"
        
        # 肩の左右差
        shoulder_diff = left_shoulder[1] - right_shoulder[1]
        shoulder_angle = math.degrees(math.atan2(shoulder_diff, 
                                               right_shoulder[0] - left_shoulder[0]))
        results["肩の左右差"] = f"{shoulder_angle:.1f}° ({'右下がり' if shoulder_diff > 0 else '左下がり'})"
        
        # 骨盤の傾き
        hip_diff = left_hip[1] - right_hip[1]
        hip_angle = math.degrees(math.atan2(hip_diff, right_hip[0] - left_hip[0]))
        results["骨盤の傾き"] = f"{hip_angle:.1f}° ({'右下がり' if hip_diff > 0 else '左下がり'})"
        
        return results
        
    except Exception as e:
        return {"エラー": f"正面姿勢分析エラー: {str(e)}"}

def analyze_side_posture(keypoints):
    """横向き姿勢の分析"""
    try:
        kpts = keypoints[0]
        results = {}
        
        # キーポイントのインデックス（左右どちらか見えている方を使用）
        nose = kpts[0][:2]
        left_ear = kpts[3][:2]
        right_ear = kpts[4][:2]
        left_shoulder = kpts[5][:2]
        right_shoulder = kpts[6][:2]
        left_hip = kpts[11][:2]
        right_hip = kpts[12][:2]
        left_knee = kpts[13][:2]
        right_knee = kpts[14][:2]
        left_ankle = kpts[15][:2]
        right_ankle = kpts[16][:2]
        
        # より信頼度の高い方を選択
        ear = left_ear if kpts[3][2] > kpts[4][2] else right_ear
        shoulder = left_shoulder if kpts[5][2] > kpts[6][2] else right_shoulder
        hip = left_hip if kpts[11][2] > kpts[12][2] else right_hip
        knee = left_knee if kpts[13][2] > kpts[14][2] else right_knee
        ankle = left_ankle if kpts[15][2] > kpts[16][2] else right_ankle
        
        # 頭の前後傾斜（耳と鼻の関係）
        if kpts[0][2] > 0.5 and max(kpts[3][2], kpts[4][2]) > 0.5:
            head_angle = math.degrees(math.atan2(nose[1] - ear[1], nose[0] - ear[0]))
            results["頭の前後傾斜"] = f"{head_angle:.1f}°"
        
        # 体幹の傾き（肩と腰を結ぶ線の垂直からの角度）
        if min(kpts[5][2], kpts[6][2], kpts[11][2], kpts[12][2]) > 0.5:
            trunk_angle = math.degrees(math.atan2(hip[0] - shoulder[0], hip[1] - shoulder[1]))
            results["体幹の傾き"] = f"{trunk_angle:.1f}°"
        
        # 膝の角度（腰-膝-足首の角度）- 修正版
        if (max(kpts[11][2], kpts[12][2]) > 0.5 and 
            max(kpts[13][2], kpts[14][2]) > 0.5 and 
            max(kpts[15][2], kpts[16][2]) > 0.5):
            
            # 内角を計算（腰-膝-足首の角度）
            inner_angle = calculate_angle(hip, knee, ankle)
            if inner_angle:
                # 膝の屈曲角度は内角そのもの
                # 180度 = 完全伸展（直立）、90度 = 90度屈曲、0度 = 完全屈曲
                extension_angle = inner_angle  # 伸展角度
                flexion_angle = 180 - inner_angle  # 屈曲角度
                
                results["膝の角度"] = f"{extension_angle:.1f}° (屈曲 {flexion_angle:.1f}°)"
        
        # 骨盤の前後傾（簡易版：腰と膝の関係から推定）
        if (max(kpts[11][2], kpts[12][2]) > 0.5 and 
            max(kpts[13][2], kpts[14][2]) > 0.5):
            pelvic_angle = math.degrees(math.atan2(knee[0] - hip[0], knee[1] - hip[1]))
            results["骨盤の前後傾"] = f"{pelvic_angle:.1f}°"
        
        return results
        
    except Exception as e:
        return {"エラー": f"横向き姿勢分析エラー: {str(e)}"}

def draw_custom_pose(image, results, thickness, color_bgr):
    """カスタム描画での姿勢表示"""
    try:
        img = image.copy()
        
        if results.keypoints is None or len(results.keypoints.data) == 0:
            return img
            
        keypoints = results.keypoints.data[0].cpu().numpy()
        
        # COCO pose のスケルトン接続定義
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # 線を描画
        for connection in skeleton:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
            
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                keypoints[pt1_idx][2] > 0.5 and keypoints[pt2_idx][2] > 0.5):
                
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                
                cv2.line(img, pt1, pt2, color_bgr, thickness)
        
        # キーポイントを描画
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.5:
                center = (int(kpt[0]), int(kpt[1]))
                cv2.circle(img, center, thickness + 2, color_bgr, -1)
                cv2.circle(img, center, thickness + 2, (255, 255, 255), 1)
        
        return img
        
    except Exception as e:
        st.error(f"描画エラー: {e}")
        return image

def process_image(image, model, confidence, thickness, color_hex):
    """画像の姿勢推定処理"""
    try:
        img_array = np.array(image)
        color_bgr = hex_to_bgr(color_hex)
        
        # 姿勢推定実行
        results = model(img_array, conf=confidence)
        
        # カスタム描画
        annotated_img = draw_custom_pose(img_array, results[0], thickness, color_bgr)
        
        return annotated_img, results[0]
    
    except Exception as e:
        st.error(f"画像処理エラー: {e}")
        return None, None

# タブ1: 姿勢推定
with tab1:
    # サイドバー設定
    st.sidebar.title("設定")

    # 描画設定
    st.sidebar.subheader("描画設定")
    line_thickness = st.sidebar.slider("線の太さ", 1, 10, 3, 1)

    # 色の選択
    color_options = {
        "緑": "#00FF00",
        "白": "#FFFFFF", 
        "黒": "#000000",
        "赤": "#FF0000",
        "青": "#0000FF"
    }
    selected_color_name = st.sidebar.selectbox("線の色", list(color_options.keys()))
    line_color = color_options[selected_color_name]

    # 分析設定
    st.sidebar.subheader("分析設定")
    posture_type = st.sidebar.selectbox(
        "姿勢タイプ",
        ["自動判定", "正面姿勢", "横向き姿勢"]
    )
    
    confidence_threshold = st.sidebar.slider("信頼度閾値", 0.1, 1.0, 0.5, 0.1)

    # メインコンテンツ
    model = load_model()
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader(
        "画像をアップロードしてください",
        type=['jpg', 'jpeg', 'png'],
        help="JPG、JPEG、PNG形式に対応しています"
    )
    
    if uploaded_file is not None:
        # 画像読み込みと向き修正
        image = Image.open(uploaded_file)
        image = fix_image_orientation(image)  # EXIF情報に基づいて自動回転
        
        # 画像情報表示
        st.caption(f"画像サイズ: {image.size[0]} × {image.size[1]} px")
        
        with st.spinner("AIが姿勢を分析しています..."):
            processed_img, results = process_image(image, model, confidence_threshold, 
                                                 line_thickness, line_color)
        
        if processed_img is not None and results.keypoints is not None:
            # 結果画像表示
            st.subheader("姿勢推定結果")
            st.image(processed_img, use_container_width=True)
            
            # 姿勢分析
            st.subheader("詳細分析")
            
            keypoints_data = results.keypoints.data.cpu().numpy()
            
            if len(keypoints_data) > 0:
                # 姿勢タイプの判定
                if posture_type == "自動判定":
                    detected_orientation = detect_posture_orientation(keypoints_data)
                    orientation_text = {'front': '正面', 'side': '横向き', 'unknown': '不明'}
                    st.info(f"検出された姿勢: **{orientation_text[detected_orientation]}**")
                    analysis_type = detected_orientation
                elif posture_type == "正面姿勢":
                    analysis_type = "front"
                else:
                    analysis_type = "side"
                
                # 分析実行と結果表示
                col1, col2 = st.columns(2)
                
                if analysis_type == "front":
                    analysis_results = analyze_front_posture(keypoints_data)
                    st.markdown("#### 正面姿勢分析")
                elif analysis_type == "side":
                    analysis_results = analyze_side_posture(keypoints_data)
                    st.markdown("#### 横向き姿勢分析")
                else:
                    analysis_results = {"状態": "姿勢の向きを判定できませんでした"}
                    st.warning("姿勢の向きを判定できませんでした。手動でタイプを選択してください。")
                
                # 結果をカラムに分けて表示
                metrics_keys = list(analysis_results.keys())
                if len(metrics_keys) > 1:
                    mid_point = len(metrics_keys) // 2
                    
                    with col1:
                        for key in metrics_keys[:mid_point]:
                            if key != "エラー":
                                st.metric(key, analysis_results[key])
                            else:
                                st.error(analysis_results[key])
                    
                    with col2:
                        for key in metrics_keys[mid_point:]:
                            if key != "エラー":
                                st.metric(key, analysis_results[key])
                            else:
                                st.error(analysis_results[key])
                else:
                    # エラーや単一結果の場合
                    for key, value in analysis_results.items():
                        if key != "エラー":
                            st.metric(key, value)
                        else:
                            st.error(value)
                
            else:
                st.warning("人物が検出されませんでした")
                
            st.success("姿勢推定が完了しました")
            st.info("結果をスクリーンショットで保存できます")
            
        else:
            st.error("姿勢推定に失敗しました。別の画像をお試しください。")
    
    else:
        # 使い方のヒント
        st.info("上のエリアに画像をアップロードして開始してください")
        
        st.subheader("推奨される写真")
        st.markdown("""
        - **明るい場所**で撮影された写真
        - **全身**が写っている写真  
        - **背景**がシンプルな写真
        - **正面向き**または**真横向き**の写真
        - **自然な立ち姿勢**の写真
        """)

def readme_tab_components():
    st.info("このアプリケーションは機械学習モデル**YOLO（You Only Look Once）**を用いて姿勢を推定し、結果を表示します。")
    
    st.subheader("使い方")
    st.markdown("""
    1. **設定を調整** - サイドバーで線の太さ、色、分析タイプを設定
    2. **画像をアップロード** - 姿勢推定タブで画像をアップロード
    3. **結果を確認** - AI分析結果とスケルトン表示を確認
    4. **必要に応じて保存** - スクリーンショット等で結果を保存
    
    ※ ブラウザを閉じると結果はクリアされます。
    """)
 
    st.subheader("🔒 個人情報保護に関する注意事項")
    st.markdown("""
    - アップロードされた画像は姿勢推定のみに使用され、**サーバーに保存されません**
    - 個人を特定できる情報が含まれる画像のアップロードは避けてください
    - 処理は全てローカル環境で実行されます
    """)
    
    st.subheader("⚠️ 免責事項")
    st.markdown("""
    このアプリケーションは**教育および参考目的のみ**で提供されています。
    医療診断や専門的な姿勢分析にはご使用いただけません。
    """)

# タブ2: 使い方
with tab2:
    readme_tab_components()

if __name__ == "__main__":
    pass