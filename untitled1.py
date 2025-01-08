# import os
# import random
# import shutil

# # Define the source directory and the destination directory
# source_dir = r'D:\Program Files\Aisino\365Server\FPCY_img\00'
# destination_dir = r"./test"

# # Ensure the destination directory exists
# os.makedirs(destination_dir, exist_ok=True)

# # Get a list of all files in the source directory
# files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# # Randomly select 30 files from the list
# selected_files = random.sample(files, 30)

# # Copy the selected files to the destination directory
# for file in selected_files:
#     shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))



# # path = r'D:\S\yzm\Mark\train_all'
# # for filename in os.listdir(path):
# #     os.remove(os.path.join(path, filename))

