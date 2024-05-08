read -p "folder_name: (e.g. custom): " folder_name
read -p "image_name: (e.g. 000.jpg): " image_name

# python -m iid.geometry_prediction logger=console data.input_path=data/$folder_name/im/$image_name output.folder=data/$folder_name
# python -m iid.material_diffusion logger=console data.input_path=data/$folder_name/im/$image_name output.folder=data/$folder_name
# python -m iid.lighting_optimization logger=console folder_name=$folder_name callbacks.file_copy.dst=data/$folder_name/lighting/0.ckpt
mkdir -p output/rendering/$folder_name
python -m iid.test logger=console folder_name=$folder_name ckpt_path=data/$folder_name/lighting/0.ckpt