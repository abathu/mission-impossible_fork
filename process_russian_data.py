import os
from utils import gpt2_original_tokenizer

# 正确的输入路径
input_dirs = [
    "/work/tc067/tc067/s2678328/babyLM/Russian/translated_file/Russain",
    "/work/tc067/tc067/s2678328/babyLM/Russian/translated_dev_file/Russain"
]

# 统一的输出目录
output_dir = "/work/tc067/tc067/s2678328/babyLM/Russian/processed_data"
os.makedirs(output_dir, exist_ok=True)

# 遍历两个路径
for data_dir in input_dirs:
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            input_file = os.path.join(subdir_path, "text.txt")
            if not os.path.exists(input_file):
                print(f"⚠️ Warning: {input_file} does not exist, skipping.")
                continue

            dataset_name = os.path.basename(data_dir)  # translated_file or translated_dev_file
            output_file = os.path.join(output_dir, f"{dataset_name}_{subdir}_processed.txt")

            # 读取和处理每行
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            processed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    tokens = gpt2_original_tokenizer.encode(line)
                    processed_line = " ".join(map(str, tokens))
                    processed_lines.append(processed_line)

            # 写入输出
            with open(output_file, "w", encoding="utf-8") as f:
                for line in processed_lines:
                    f.write(line + "\n")

            print(f"✅ Processed: {input_file} → {output_file}")
