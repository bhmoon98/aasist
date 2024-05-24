def count_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

# 사용 예시
train_list = '/media/NAS/DATASET/1mDFDC/filelist/val_videos.txt'
line_count = count_lines(train_list)
print(f"The file {train_list} has {line_count} lines.")
