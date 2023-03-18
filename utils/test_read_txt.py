with open('/data1/ABAW/ABAW5/Aff-Wild2/annotations/AU_Detection_Challenge/Validation_Set/video61.txt', 'r') as f:
    lines = f.readlines()
cnt = 0
for i, line in enumerate(lines):
    if i == 0:
        continue
    l = line.strip('\n')
    au = [int(x) for x in l.split(',')]
    print(au)
    cnt += 1
print(cnt)

