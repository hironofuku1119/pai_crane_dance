# 引用と参考

## 引用（直接使ったもの）

- 手法：T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", 2023（arXiv:2304.13705）。ACT の提案論文
- LeRobot 0.6.1（Hugging Face）：ACT の実装・学習の入口・データ形式（LeRobotDataset）
- PyTorch 2.11.0：学習（MPS）
- librosa 1.0.0：曲の拍の検出。NumPy・SciPy：数値計算
- ROS 2 Humble・Gazebo Sim 6.16（Fortress）
- crane_plus（rt-net）
- ros2_control：講座の Docker イメージに入っているものを使用
- 講座内 Docker イメージ：airobotbook/ros2-desktop-ai-robot-book-humble（Docker Hub）。
- teleop：matsuolab/physicalai_teleop
  - https://github.com/matsuolab/physicalai_teleop.git
  - 使った commit：0ef2a00fd6d7
  - キーフレームの 4 つの姿勢を作るのに使用

## 参考（操作のしかたや内容の理解に使ったもの）

- 東京大学 松尾・岩澤研究室「Physical AI 講座」基礎編・応用編1　講義資料一式
- LeRobot（ACT・データ形式）：
  - https://huggingface.co/docs/lerobot/act
  - https://huggingface.co/docs/lerobot/v0.6.1/en/lerobot-dataset-v3
- PyTorch（Apple の GPU・MPS）：
  - https://docs.pytorch.org/docs/2.11/notes/mps.html
- librosa（拍の検出）：
  - https://librosa.org/doc/latest/api/generated/librosa.beat.beat_track.html
  - https://librosa.org/doc/latest/api/generated/librosa.onset.onset_strength.html
  - https://librosa.org/doc/latest/api/generated/librosa.frames_to_time.html
  - https://librosa.org/doc/latest/api/generated/librosa.load.html
- ROS 2（時計とシム時刻・通信の設定・記録）：
  - https://design.ros2.org/articles/clock_and_time.html
  - https://design.ros2.org/articles/qos.html
  - https://index.ros.org/r/rosbag2/
  - https://index.ros.org/p/sensor_msgs/
- noVNC（ブラウザでの画面表示）：
  - https://novnc.com/noVNC/docs/EMBEDDING.html

## AI の利用

- Claude Code：解析の道具・コード・文書の作成の補助
- Codex CLI：コードのレビュー
- Suno：音楽生成
- Gemini：公開用の動画の一部を生成
