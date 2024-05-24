# 파일 위치 변경 파일임

# 입력 파일 경로
file_path = '/media/NAS/DATASET/1mDFDC/filelist/test_audio.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

with open(file_path, 'w') as file:
    for line in lines:
        # 각 라인의 'test/' 부분 삭제
        new_line = line.replace('test/', '')
        file.write(new_line)

print("변환 완료!")

