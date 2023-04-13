def process_one_image_boxes(output_box):
    if len(output_box) < 3:
        return output_box
    new_output_box = [output_box[0]]
    for i in range(1, len(output_box) - 1):
        if output_box[i][0] < output_box[i-1][2] and output_box[i][2] > output_box[i+1][0]:
            continue
        else:
            new_output_box.append(output_box[i])
    new_output_box.append(output_box[-1])
    return new_output_box

def convert_boxs_to_notes(output_boxes, hop, scale_h, scale_w, width,height):
    notes = []
    last_state = 0 # 0 for end 1 for open
    last_pos = 0
    for i, output_box in enumerate(output_boxes):
        offset_pixel = hop * i
        output_box = output_box.cpu().numpy().tolist()
        output_box.sort(key=lambda x: x[0])
        output_box = process_one_image_boxes(output_box)
        
        first = True
        for j, (x1, y1, x2, y2, conf) in enumerate(output_box):
            if x1 < last_pos - 12 and x2 < last_pos + 7:
                continue
            onset = (x1 + offset_pixel) * scale_w
            offset = (x2 + offset_pixel) * scale_w
            # pitch = (height-y2) * scale_h + 21.0
            pitch = round(height-y2, 3)#直接存储像素点
            # 上一个音符open #或者 上一个音符open但是last_state有误
            if first and last_state and x1 <= last_pos + 7:# or x1 < last_pos and x2 > last_pos):
                # if len(notes) > 1 and notes[-1][0] < notes[-2][1]  and j > 0 and output_box[j][0] > output_box[j-1][2]:
                #     notes[-1][0] = onset
                notes[-1][1] = offset
                notes[-1][2] = (notes[-1][2] + pitch) / 2
                #先暂时不要conf
                # notes[-1][-1] = (notes[-1][-1] + conf) / 2
                first = False
            else:
                # notes.append([onset, offset, pitch, conf])
                notes.append([onset, offset, pitch])

            if j == len(output_box) - 1:
                last_state = 0 if x2 < width - 4 else 1
                last_pos = x2 - hop if x2 < width - 4 else x1 - hop
            
    return notes