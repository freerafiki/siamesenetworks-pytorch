import os 
import shutil 

rendered_folder = '/media/lucap/big_data/datasets/repair/triplets/siamese_repair'
images_folder = os.path.join(rendered_folder, 'seg')
output_dir = os.path.join(rendered_folder, 'data')
imgs_paths = os.listdir(images_folder)

for img_path in imgs_paths:

    # g_ind = img_path.index('Group')+5
    # group_num = img_path[g_ind:g_ind+2]

    id_ind = img_path.index('RPf')+4
    rpf_id = img_path[id_ind:id_ind+5]

    target_folder = os.path.join(output_dir, f'{rpf_id}')
    os.makedirs(target_folder, exist_ok=True)   
    source_path = os.path.join(images_folder, img_path)
    target_path = os.path.join(target_folder, img_path)
    shutil.copy(source_path, target_path)

output_folders = os.listdir(output_dir)
print(f'there are now {len(output_folders)} folder in {output_dir}')
for fold in output_folders:
    #print(f'copied {source_path} to {target_path}')
    print(f'there are {len(os.listdir(os.path.join(output_dir, fold)))} in {fold}')