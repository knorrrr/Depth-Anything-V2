import cv2
import os
import numpy as np
def blend_images(rgb_path, event_path, output_path, alpha=0.5):
    """
    イベント画像の黒色部分(R=0,G=0,B=0)を無視して、RGB画像とイベント画像をアルファブレンドで合成する関数。

    Parameters:
    -----------
    rgb_path : str
        RGB画像のファイルパス
    event_path : str
        イベント画像のファイルパス
    output_path : str
        合成結果を保存するファイルパス
    alpha : float
        RGB画像とイベント画像をブレンドする割合(0~1)
        （値が大きいほどRGB画像が強く表示され、小さいほどイベント画像が強く表示される）
    """
    # 画像を読み込み
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    event_image = cv2.imread(event_path, cv2.IMREAD_COLOR)
    
    if rgb_image is None:
        raise FileNotFoundError(f"RGB画像が見つかりませんでした: {rgb_path}")
    if event_image is None:
        raise FileNotFoundError(f"イベント画像が見つかりませんでした: {event_path}")
    
    # イベント画像をRGB画像と同じ大きさにリサイズ（必要なら）
    if rgb_image.shape[:2] != event_image.shape[:2]:
        event_image = cv2.resize(event_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # イベント画像の黒色部分（R=0, G=0, B=0）を判定するマスクを作成
    # inRange(イベント画像, 最小値, 最大値) で黒色部分だけを取り出す
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([0, 0, 0], dtype=np.uint8)
    black_mask = cv2.inRange(event_image, lower_black, upper_black)  # 黒色部分 -> 255、それ以外 -> 0

    # マスクを反転して黒色「以外」の領域を抽出
    not_black_mask = cv2.bitwise_not(black_mask)  # 黒色以外 -> 255、黒色 -> 0

    # 合成結果を格納するためにRGB画像をコピー
    blended = rgb_image.copy()

    # 黒色「以外」の部分だけアルファブレンドして合成
    # マスクが255（黒色以外）の画素について、(alpha * RGB + (1-alpha) * event) を計算
    indices = np.where(not_black_mask == 255)
    blended[indices] = cv2.addWeighted(
        rgb_image[indices], alpha,
        event_image[indices], 1 - alpha,
        0
    )

    # 合成結果を保存
    cv2.imwrite(output_path, blended)
    print(f"合成画像を保存しました: {output_path}")

def process_folder(rgb_dir, event_dir, output_dir, alpha=0.0):
    """
    rgb_dir 内の画像と event_dir 内の画像をファイル名単位でペアにして合成し、
    output_dir に保存する関数。
    
    alpha=0.0 の場合はイベント画像のみ表示になるので、
    必要に応じてパラメータを調整してください。
    """
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # RGBフォルダ内のファイルを走査
    for filename in os.listdir(rgb_dir):
        if filename.lower().endswith(".png"):
            # RGB画像パス
            rgb_path = os.path.join(rgb_dir, filename)
            # イベント画像パス（同じファイル名を想定）
            event_path = os.path.join(event_dir, filename)

            # イベント画像が存在する場合のみ処理
            if os.path.exists(event_path):
                output_path = os.path.join(output_dir, filename)
                blend_images(rgb_path, event_path, output_path, alpha=alpha)
            else:
                print(f"イベント画像が見つかりませんでした: {event_path}")

# メイン部分（スクリプトとして実行する場合の例）
if __name__ == "__main__":
    # 読み込みたいRGB画像とイベント画像のパス、保存先パスを指定してください
    rgb_dir = "/home/hermes-22/Depth-Anything-V2/SEVD/M-TE2/ego0/images/rgb_camera/rgb_camera-front"
    event_dir = "/home/hermes-22/Depth-Anything-V2/SEVD/M-TE2/ego0/images/dvs_camera/dvs_camera-front"
    output_dir = "/home/hermes-22/Depth-Anything-V2/SEVD/M-TE2/ego0/rgb_ev"   

    process_folder(rgb_dir, event_dir, output_dir, alpha=0.0)

