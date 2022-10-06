import os
import json
import random
import glob
# with open('./mpii_trb_train.json','r') as f:
#     p = json.load(f)
# print(p.keys())
# print(p['size'])
# print(os.listdir(os.getcwd()))
def json_making(json):
    image_size = json['size']
    image_id = json['objects'][0]['id']
    num_joints = len(json['objects'][0]['nodes'])
    keypoints = list(json['objects'][0]['nodes'].values())
    center = json['objects'][1]['points']['exterior'][0]
    image_h,image_w = image_size.values()
    bbox = json['objects'][2]['points']['exterior']
    id_ = int(str(json['objects'][0]['id'])[-6:])
    x1,y1 = bbox[0]
    x2,y2 = bbox[1]
    width = x2-x1
    height = y2-y1
    target_size = (width * height) / (image_h * image_w)
    kp_dic = {}
    kp_list = []
    for kp in keypoints:
        a,b = kp['loc']
        kp_list.append(a)
        kp_list.append(b)
        source_point = random.randint(1,2)
        kp_list.append(source_point)
    kp_dic['num_joints'] = num_joints
    kp_dic['keypoints'] = kp_list
    kp_dic['image_id'] = image_id
    kp_dic['center'] = center
    kp_dic['size'] = target_size
    kp_dic['category_id'] = 1
    kp_dic['id'] = id_
    kp_dic['iscrowd'] = 0
    return kp_dic, image_h,image_w, id_

# json 형식은 무조건 맞춰야됌 D:\taskfrom3_10\Avatar_human\mmpose\ds0\ann\00000.jpg.json 이거로
# print(glob.glob('./ds0/ann/*.json'))
if __name__ == '__main__':
    dic_list_i = []
    dic_list_a = []
    final_dic = {}
    file_path = './annotation/' 
    file_list = os.listdir('./annotation/')
    print(file_list)
    for file in os.listdir('./annotation/'):
        with open(os.path.join('./annotation/' + file),'r') as f:
            p = json.load(f)
        f_inner_dic = {}
        kp_dic_,image_h,image_w,id_ = json_making(p)
        f_inner_dic['file_name'] = file
        f_inner_dic['height'] = image_h
        f_inner_dic['width'] = image_w
        f_inner_dic['id'] = id_
        dic_list_i.append(f_inner_dic)
        dic_list_a.append(kp_dic_)
    final_dic['images'] = dic_list_i
    final_dic['annotations'] = dic_list_a
    print(final_dic)
    print(file_list)