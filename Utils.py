from scipy.spatial import distance
import cv2
import numpy as np

'''
Written by B
'''

# ------------------- Body Figure Caculation -------------------
def point_distance_mpii(pose_result):
    '''
    Caculate Distance of Human Body Key Point(Mpii Format)
    '''
    keypoint = pose_result[0]['keypoints']
    Right_leg = distance.euclidean((keypoint[0][0],keypoint[0][1]),(keypoint[1][0],keypoint[1][1])) + distance.euclidean((keypoint[1][0],keypoint[1][1]),(keypoint[2][0],keypoint[2][1]))
    print('Right Leg:{}'.format(Right_leg))

    Left_leg = distance.euclidean((keypoint[3][0],keypoint[3][1]),(keypoint[4][0],keypoint[4][1])) + distance.euclidean((keypoint[4][0],keypoint[4][1]),(keypoint[5][0],keypoint[5][1]))
    print('Left Leg:{}'.format(Left_leg))

    Right_Arm = distance.euclidean((keypoint[10][0],keypoint[10][1]),(keypoint[11][0],keypoint[11][1])) + distance.euclidean((keypoint[11][0],keypoint[11][1]),(keypoint[12][0],keypoint[12][1]))
    print('Right Arm:{}'.format(Right_Arm))

    Left_Arm = distance.euclidean((keypoint[13][0],keypoint[13][1]),(keypoint[14][0],keypoint[14][1])) + distance.euclidean((keypoint[14][0],keypoint[14][1]),(keypoint[15][0],keypoint[15][1]))
    print('Left Arm:{}'.format(Left_Arm))

    Shoulder = distance.euclidean((keypoint[12][0],keypoint[12][1]),(keypoint[13][0],keypoint[13][1]))
    print('Shoulder:{}'.format(Shoulder))
    
    Head = distance.euclidean((keypoint[8][0],keypoint[8][1]),(keypoint[9][0],keypoint[9][1]))
    print('Head:{}'.format(Head))

    bbox = list(map(float,pose_result[0]['bbox']))
    Height = bbox[3] - bbox[1]
    print('Height: {}'.format(Height))

    Body_Ratio = Height//Head
    print('Whole Body Ratio: {}'.format(Body_Ratio))
    Shoulder_to_Body = Shoulder/Height
    print('Shoulder to Body Ratio: {}'.format(round(Shoulder_to_Body,3)))
    return Right_leg, Left_leg, Right_Arm, Left_Arm, Shoulder, Head, Height, Body_Ratio, Shoulder_to_Body

def point_distance_mpii_trb(pose_result):
    '''
    Caculate distance of keypoint (Mpii TRB Format)

    Return : dict() , Body Information
    '''

    body_information_dict = {}

    keypoint = pose_result[0].get('keypoints')
    bbox = pose_result[0].get('bbox')

    Shoulder = distance.euclidean((keypoint[0][0],keypoint[0][1]),(keypoint[1][0],keypoint[1][1]))
    body_information_dict['Shoulder'] = Shoulder
    print('Shoulder:{}'.format(Shoulder))

    Left_Arm = distance.euclidean((keypoint[1][0],keypoint[1][1]),(keypoint[2][0],keypoint[2][1])) + distance.euclidean((keypoint[2][0],keypoint[2][1]),(keypoint[4][0],keypoint[4][1]))
    body_information_dict['Left_Arm'] = Left_Arm
    print('Left Arm:{}'.format(Left_Arm))

    Right_Arm = distance.euclidean((keypoint[0][0],keypoint[0][1]),(keypoint[3][0],keypoint[3][1])) + distance.euclidean((keypoint[3][0],keypoint[3][1]),(keypoint[5][0],keypoint[5][1]))
    body_information_dict['Right_Arm'] = Right_Arm
    print('Right Arm:{}'.format(Right_Arm))

    Left_leg = distance.euclidean((keypoint[6][0],keypoint[6][1]),(keypoint[8][0],keypoint[8][1])) + distance.euclidean((keypoint[8][0],keypoint[8][1]),(keypoint[10][0],keypoint[10][1]))
    body_information_dict['Left_leg'] = Left_leg
    print('Left Leg:{}'.format(Left_leg))

    Right_leg = distance.euclidean((keypoint[7][0],keypoint[7][1]),(keypoint[9][0],keypoint[9][1])) + distance.euclidean((keypoint[9][0],keypoint[9][1]),(keypoint[11][0],keypoint[11][1]))
    body_information_dict['Right_leg'] = Right_leg
    print('Right Leg:{}'.format(Right_leg))

    Neck_to_Center = distance.euclidean((keypoint[13][0],keypoint[13][1]),(keypoint[28][0],keypoint[28][1])) 
    body_information_dict['Neck_to_Center'] = Neck_to_Center
    print('Neck to Center:{}'.format(Neck_to_Center))

    Left_Shoulder_Thick = distance.euclidean((keypoint[22][0],keypoint[22][1]),(keypoint[23][0],keypoint[23][1]))
    body_information_dict['L_Shoulder_Thick'] = Left_Shoulder_Thick
    print('Left Shoulder Thick:{}'.format(Left_Shoulder_Thick))

    Right_Shoulder_Thick = distance.euclidean((keypoint[16][0],keypoint[16][1]),(keypoint[17][0],keypoint[17][1]))
    body_information_dict['R_Shoulder_Thick'] = Right_Shoulder_Thick
    print('Right Shoulder Thick:{}'.format(Right_Shoulder_Thick))

    Left_Thigh_Thick = distance.euclidean((keypoint[28][0],keypoint[28][1]),(keypoint[35][0],keypoint[35][1]))
    body_information_dict['L_Thigh_Thick'] = Left_Thigh_Thick
    print('L_Thigh Thick:{}'.format(Left_Thigh_Thick))

    Right_Thigh_Thick = distance.euclidean((keypoint[28][0],keypoint[28][1]),(keypoint[29][0],keypoint[29][1]))
    body_information_dict['R_Thigh_Thick'] = Right_Thigh_Thick
    print('R_Thigh Thick:{}'.format(Right_Thigh_Thick))

    Left_Knee_Thick = distance.euclidean((keypoint[36][0],keypoint[36][1]),(keypoint[37][0],keypoint[37][1]))
    body_information_dict['L_Knee_Thick'] = Left_Knee_Thick
    print('L_Knee Thick:{}'.format(Left_Knee_Thick))

    Right_Knee_Thick = distance.euclidean((keypoint[30][0],keypoint[30][1]),(keypoint[31][0],keypoint[31][1]))
    body_information_dict['R_Knee_Thick'] = Right_Knee_Thick
    print('R_Knee Thick:{}'.format(Right_Knee_Thick))

    Left_Ankle_Thick = distance.euclidean((keypoint[38][0],keypoint[38][1]),(keypoint[39][0],keypoint[39][1]))
    body_information_dict['L_Ankle_Thick'] = Left_Ankle_Thick
    print('L_Ankle Thick:{}'.format(Left_Ankle_Thick))

    Right_Ankle_Thick = distance.euclidean((keypoint[32][0],keypoint[32][1]),(keypoint[33][0],keypoint[33][1]))
    body_information_dict['R_Ankle_Thick'] = Right_Ankle_Thick
    print('R_Ankle Thick:{}'.format(Right_Ankle_Thick))

    height = bbox[3] - bbox[1]
    body_information_dict['Height'] = height
    print('Height : {}'.format(height))
    return body_information_dict

def _img_size(image_name):
    img = cv2.imread(image_name,cv2.IMREAD_COLOR)
    img_h , img_w , img_c = img.shape
    return img_h,img_w,img_c

def target_person_size(pose_result,image_name):
    '''
    Ratio of box: Bounding box와 Image의 비율을 계산하여 대상 인물이 이미지내에서 어느 정도의 크기를 가졌는지 계산
    '''
    bbox = pose_result[0].get('bbox')
    h,w,c = _img_size(image_name)
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    ratio_of_box = (h * w) // (width * height)
    print('Ratio of Bbox : {}'.format(ratio_of_box))
    return ratio_of_box

def Ratio_Caculate(smpl_scr_value,style_trg_value1,style_trg_value2):
    '''
    Input : smpl_scr_value = smpl의 고정된 수치(Head)
            style_trg_value = style input에서 추출된 keypoint distance

    Output : Target Distance = 메쉬가 해당 값으로 변경되어야 함
    '''
    big_v, small_v = max([style_trg_value1,style_trg_value2]) , min([style_trg_value1,style_trg_value2])
    return round((smpl_scr_value * small_v) / big_v, 3)
    

    
  

# print(dir(target_person_size))

# ------------------------- 3D Mesh Processing ------------------------------

import bpy

def setup_scene(model_path, fps_target):
    scene = bpy.data.scenes['Scene']

    ###########################
    # Engine independent setup
    ###########################

    scene.render.fps = fps_target

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Import gender specific .fbx template file
    bpy.ops.import_scene.fbx(filepath=model_path)

def Shapekey_processing(model_name,target_model_name):
    '''
    Mesh data transfer Using Shapekey of Mesh
    
    ShapeKey가 존재하지 않을 경우에 실행

    Input: model_name / target_model_name , type: String

    output : Mesh Data Transfer Shapekey Value
    '''
    model = bpy.data.objects[model_name]
    model.select_set(True)
    model.mesh_data_transfer_object.mesh_object_space = "UVS" 
    model.mesh_data_transfer_object.search_method = "CLOSEST" 
    model.mesh_data_transfer_object.mesh_source = bpy.data.objects[target_model_name] 
    model.mesh_data_transfer_object.transfer_shape_as_key = True 
    bpy.ops.object.transfer_shape_data()
    key_value = bpy.data.shape_keys["Key"].key_blocks[target_model_name + ".Transferred"].value 
    return model , key_value

def Export_data(model, output_path):
    '''
    Mesh File Export
    Input -> model: bpy.data.objects[model_name] , output_path : file Directory + file name (string)
    '''
    model.select_set(True)
    print('Exporting to FBX binary (.fbx)')
    bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False)  
    return

def limbs_deform(model,Right_arm_move,Left_arm_move,Right_leg_move,Left_leg_move):
    '''
    limbs Deformation
    Input -> model: bpy.data.objects[model_name] , _move : Vector Value (float)

    Vertex Groups Index
    0 : Right Hand
    1: Left Hand
    2: Right Leg
    3 : Left Leg

    Shape Key Index(수정 예정)
    0: Basis
    1: limbs
    '''
    # Edit Mode
    bpy.ops.object.editmode_toggle()

    # Model Initialize
    model.select_set(True)
    # Shape Key Active (Limbs) : Shape Key의 인덱스를 잘 설정해 줘야함 현재 값: 1
    model.active_shape_key_index = 1
    if model.active_shape_key_index == None:
        print('Shape Key is Missing')

    # Right Arm Deform
    model.vertex_groups.active_index = 0
    bpy.ops.object.vertex_group_select()
    bpy.ops.transform.translate(value=(-Right_arm_move, 0, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=0.564474, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.editmode_toggle()
    # Left Arm Deform
    model.vertex_groups.active_index = 1
    bpy.ops.object.vertex_group_select()
    bpy.ops.transform.translate(value=(Left_arm_move, 0, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=True, use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=0.564474, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    bpy.ops.object.vertex_group_deselect()
    bpy.ops.object.editmode_toggle()

# print('smpl_woman' + '.Transferred')

# bpy.data.shape_keys["Key"].key_blocks['limbs']