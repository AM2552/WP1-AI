import os

folder_path = '/Users/Xandi/Desktop/herpe/amphibien/107_gelbbauchunke'

counter = 1
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    new_filename = f'107_gelbbauchunke_{counter}.png'
    new_file_path = os.path.join(folder_path, new_filename)
    
    os.rename(file_path, new_file_path)
    counter += 1