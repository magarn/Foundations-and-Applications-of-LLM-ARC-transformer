import os
import soundfile as sf

# 你的数据根目录
root_dir = "./data/miniLibriSpeech/dev-clean-2"

# 输出文件
audio_path_file = "./data/miniLibriSpeech/dev-clean-2.paths"
text_file = "./data/miniLibriSpeech/dev-clean-2.text"
lengths_file = "./data/miniLibriSpeech/dev-clean-2.lengths"

# 可选：transcription 文件（每个说话人文件夹下会有 .txt 文件）
# 例：19/198/19-198.trans.txt
def load_transcripts(dir_path):
    trans = {}
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".trans.txt"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fin:
                    for line in fin:
                        utt_id, *words = line.strip().split()
                        trans[utt_id] = " ".join(words)
    return trans

transcripts = load_transcripts(root_dir)

paths, texts, lengths = [], [], []

# 遍历 flac 文件
for root, _, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".flac"):
            path = os.path.join(root, f)
            utt_id = os.path.splitext(f)[0]  # 如 19-198-0000
            paths.append(path)
            # 读取音频长度（秒）
            audio, sr = sf.read(path)
            length = len(audio) / sr
            lengths.append(f"{length:.2f}")
            # 文字（如果有）
            text = transcripts.get(utt_id, "")
            texts.append(text)

# 写入文件
with open(audio_path_file, "w", encoding="utf-8") as f:
    f.write("\n".join(paths))
with open(text_file, "w", encoding="utf-8") as f:
    f.write("\n".join(texts))
with open(lengths_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lengths))

print(f"✅ 已生成 {len(paths)} 条数据")
print("Paths file:", audio_path_file)
print("Text file:", text_file)
print("Lengths file:", lengths_file)
